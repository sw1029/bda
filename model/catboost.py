from catboost import CatBoostClassifier, CatBoostRegressor
from base import model
import pandas as pd
from pathlib import Path
from datetime import datetime

from utils import parse_timestamp

'''
class model:
    def __init__(self):
        pass
    def train(self, data:pd.DataFrame) -> None:
        pass
    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        pass
'''

class cat(model):
    def __init__(
        self,
        type: str = "classifier",
        **kwargs
    ):
        super().__init__()
        
        if type == 'classifier':
            self.model = CatBoostClassifier(**kwargs)
        elif type == 'regressor':
            self.model = CatBoostRegressor(**kwargs)
        else:
            raise ValueError("악! 입력값이 너무 많지 말입니다! 이게 해병 악기바리?")
        
        self.args = kwargs # 모델 파라미터 저장
        self.type = type
        self.is_trained = False

    def load(self, model_path:str, args_path:str = None) -> None:
        model_name = Path(model_path).name

        if model_name.endswith('_cls.cbm') or model_name.endswith('cls.cbm'):
            self.type = 'classifier'
        elif model_name.endswith('_reg.cbm') or model_name.endswith('reg.cbm'):
            self.type = 'regressor'


        if self.type == 'classifier':
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRegressor()
        self.model.load_model(model_path)


        if args_path is not None:
            self.load_args(args_path)

        self.timestamp = parse_timestamp(model_name, "catboost_model_")

        self.is_trained = True

    def train(self,X_train, y_train, save_dir:str = None ) -> None:
        self.model.fit(X_train, y_train)

        self.is_trained = True
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if self.type == 'classifier':
                model_path = save_dir / f"catboost_model_{self.timestamp}_cls.cbm"
            else:
                model_path = save_dir / f"catboost_model_{self.timestamp}_reg.cbm"
            self.model.save_model(model_path)

            args_path = save_dir / f"catboost_args_{self.timestamp}.yaml"
            self.save_args(self.args, args_path)

    def predict(self, input_data: pd.DataFrame, save_dir = None) -> pd.DataFrame:
        '''
        ID : 샘플별 고유 ID. input_data의 ID column
        completed : (TARGET) 수료 여부(0, 1). predict 결과
        형식으로 csv 저장.
        '''
        if not self.is_trained:
            raise Exception("학습도 안하고 모델을 쓰려고 하다니... 기열!")
        

        if self.type == 'classifier':
            preds = self.model.predict(input_data.drop(columns=['ID']), prediction_type="Class")
        else:
            preds = self.model.predict(input_data.drop(columns=['ID']))
        

        output = pd.DataFrame({
            'ID': input_data['ID'],
            'completed': preds
        })
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"catboost_predictions_{self.timestamp}.csv"
            output.to_csv(output_path, index=False)

        return output