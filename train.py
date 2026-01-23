import hydra
from omegaconf import DictConfig
from model import factory # 바이브 코딩이 좋더라
from data import *
from utils import set_seed
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import random

@hydra.main(version_base=None, config_path="config", config_name="config") # hydra 데코레이터를 통해 실험 설정 관리
def main(cfg: DictConfig) -> None:
    
    model = factory.get_model(
        model_cfg=cfg.model,
        registry_cfg=cfg.registry,
        unknown_key_policy=cfg.get("unknown_key_policy", "error"),
    )

    data = pd.read_csv(cfg.train_csv) # 여기서 데이터 전처리 함수, 결과물 호출

    with set_seed(cfg.seed): # 시드 설정/원복용

        # 데이터도 특정 시드에서 train/valid 분할 진행
        if cfg.get("need_valid"):
            data_train, data_valid = train_test_split(
                data,
                test_size=cfg.get("valid_ratio", 0.2),
                random_state=cfg.seed
            )
        else:
            data_train, data_valid = data, None

        model.train(data_train=data_train, data_valid=data_valid, 
                    id_label=cfg.id_label, target_label=cfg.target_label) 
        # 데이터 안넣어주면 에러남. 데이터 지정해줄것.
        # valid 데이터가 None이면 내부에서 알아서 처리함. 최종 inference 목적으로 쓰는걸 추천.
    
    
    if cfg.get("do_inference"):
        pred = model.predict(input_data=pd.read_csv(cfg.inference_csv),
                        save_dir=cfg.log_dir) # save dir 경로가 None 혹은 지정 안해주는 경우 추론 결과 저장 안됨.

if __name__ == "__main__":
    main()


'''
실제 사용 시 python train.py model.depth=10 와 같이 커맨드창에서 설정 오버라이드 가능
'''