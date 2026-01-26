from xgboost import XGBClassifier, XGBRegressor
from ..base import model
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

from utils import load_args, parse_timestamp, prob_positive, save_args, save_valid_metrics

'''
class model:
    def __init__(self):
        pass
    def train(self, data:pd.DataFrame) -> None:
        pass
    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        pass
'''

class xg(model):
    def __init__(
        self,
        type: str = "classifier",
        **kwargs
    ):
        super().__init__()
        
        kwargs.pop("type", None)

        if type == 'classifier':
            self.model = XGBClassifier(**kwargs)
        elif type == 'regressor':
            self.model = XGBRegressor(**kwargs)
        else:
            raise ValueError("악! 입력값이 너무 많지 말입니다! 이게 해병 악기바리?")
        
        self.args = kwargs # 모델 파라미터 저장
        self.type = type
        self.is_trained = False






    def load(self, model_path:str, args_path:str = None) -> None:
        model_name = Path(model_path).name

        if model_name.endswith('_cls.json') or model_name.endswith('cls.json'):
            self.type = 'classifier'
        elif model_name.endswith('_reg.json') or model_name.endswith('reg.json'):
            self.type = 'regressor'


        if self.type == 'classifier':
            self.model = XGBClassifier()
        else:
            self.model = XGBRegressor()
        self.model.load_model(model_path)


        if args_path is not None:
            self.args = load_args(args_path)

        self.timestamp = parse_timestamp(model_name, "xgboost_model_")

        self.is_trained = True






    def train(self, 
                data_train: pd.DataFrame, 
                data_valid: pd.DataFrame = None, 

                save_dir: str = None, 
                save_group: str = None, 

                id_label: str = None, 
                target_label: str = None,


                **kwargs) -> None:

        X_train = data_train.drop(columns=[id_label, target_label])
        y_train = data_train[target_label]

        if data_valid is not None:
            X_valid = data_valid.drop(columns=[id_label, target_label])
            y_valid = data_valid[target_label]
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                **kwargs
            )
        else:
            self.model.fit(
                X_train, y_train,
                **kwargs
            )

        self.is_trained = True
        
        if save_group is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir = save_dir / "xgboost"
            if save_group is not None:
                save_dir = save_dir / str(save_group)
            save_dir = save_dir / f"{self.timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if self.type == 'classifier':
                model_path = save_dir / f"xgboost_model_{self.timestamp}_cls.json"
            else:
                model_path = save_dir / f"xgboost_model_{self.timestamp}_reg.json"
            self.model.save_model(model_path)

            args_path = save_dir / f"xgboost_args_{self.timestamp}.yaml"
            save_args(self.args, args_path)

            if data_valid is not None:
                save_valid_metrics(
                    trained_model=self,
                    data_valid=data_valid,
                    id_label=id_label,
                    target_label=target_label,
                    artifact_dir=save_dir,
                    file_prefix="xgboost_valid_metrics",
                )






    def predict(
        self,
        input_data: pd.DataFrame,
        save_dir=None,
        save_group: str = None,
        threshold: float | None = 0.5,
    ) -> pd.DataFrame:
        '''
        ID : 샘플별 고유 ID. input_data의 ID column
        completed : (TARGET) 수료 여부(0, 1). predict 결과
        형식으로 csv 저장.
        '''
        if not self.is_trained:
            raise Exception("학습도 안하고 모델을 쓰려고 하다니... 기열!")
        

        X = input_data.drop(columns=["ID"])
        if threshold is None:
            preds = self.model.predict(X)
        else:
            score = None
            if hasattr(self.model, "predict_proba"):
                score = prob_positive(self.model.predict_proba(X))
            if score is None:
                score = np.asarray(self.model.predict(X)).reshape(-1)
            preds = (score >= float(threshold)).astype(int)
        

        output = pd.DataFrame({
            'ID': input_data['ID'],
            'completed': preds
        })
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir = save_dir / "xgboost"
            if save_group is not None:
                save_dir = save_dir / str(save_group)
            save_dir = save_dir / f"{self.timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"xgboost_predictions_{self.timestamp}.csv"
            output.to_csv(output_path, index=False)

        return output
