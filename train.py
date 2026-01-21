import hydra
from omegaconf import DictConfig
from model import factory # 바이브 코딩이 좋더라
from data import *

@hydra.main(version_base=None, config_path="config", config_name="config") # hydra 데코레이터를 통해 실험 설정 관리
def main(cfg: DictConfig):
    
    model = factory.get_model(
        model_cfg=cfg.model,
        registry_cfg=cfg.registry,
        unknown_key_policy=cfg.get("unknown_key_policy", "error"),
    )
    model.train(data_train=data_train, data_valid=data_valid, id_label=cfg.id_label, target_label=cfg.target_label, seed=cfg.seed) # 데이터 안넣어주면 에러남. 데이터 지정해줄것.

if __name__ == "__main__":
    main()


'''
실제 사용 시 python train.py model.depth=10 와 같이 커맨드창에서 설정 오버라이드 가능
'''