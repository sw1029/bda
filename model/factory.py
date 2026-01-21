'''
모델 객체 던져주는 팩토리 패턴
근데 이걸 팩토리라고 불러도 되나 싶긴 한데
아무튼 팩토리로 치기로 함

객체지향의 봄은 온다...
'''

from model.base import model

def get_model(**kwargs) -> model:
    if kwargs.get("model_name") == "catboost":
        from model.catboost import cat
        return cat(**kwargs)
    elif kwargs.get("model_name") == "mlp":
        from model.mlp import mlp
        return mlp(**kwargs)