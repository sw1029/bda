'''
모델 관련 모듈로 사용하기 위한 파일
어지간하면 base.py에 정의된 클래스를 상속하는 방식으로 진행
가능하면 pytorch lightning과 같이 model.train() 으로 학습 자체도 호출 가능하도록
config는 config/config.yaml를 hydra로 불러와서 사용
'''

from .models.catboost import cat
from .models.mlp import mlp

__all__ = ['cat', 'factory', 'mlp']
