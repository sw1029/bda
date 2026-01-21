from catboost import CatBoostClassifier, CatBoostRegressor
from base import model
import pandas as pd
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

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

    def train(self,X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, input_data: pd.DataFrame, save_dir = None) -> pd.DataFrame:
        '''
        ID : 샘플별 고유 ID. input_data의 ID column
        completed : (TARGET) 수료 여부(0, 1). predict 결과
        형식으로 csv 저장.
        '''
        if not self.is_trained:
            raise Exception("학습도 안하고 모델을 쓰려고 하다니... 기열!")
        
        preds = self.model.predict(input_data.drop(columns=['ID']), prediction_type="Class")
        output = pd.DataFrame({
            'ID': input_data['ID'],
            'completed': preds
        })
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = save_dir / f"catboost_predictions_{timestamp}.csv"
            output.to_csv(output_path, index=False)

            args = self.args
            args_path = save_dir / f"catboost_args_{timestamp}.yaml"
            with open(args_path, 'w') as f:
                for key, value in args.items():
                    f.write(f"{key}: {value}\n")
        return output

    