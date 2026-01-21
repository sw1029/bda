import hydra
from omegaconf import DictConfig
from model import cat # 호출 예시. 추후 factory 패턴으로 model 모듈 개선 예정

@hydra.main(version_base=None, config_path="config", config_name="catboost") # hydra 데코레이터를 통해 실험 설정 관리
def main(cfg: DictConfig):
    model = cat(type=cfg.type, **cfg.params)

if __name__ == "__main__":
    main()


'''
실제 사용 시 python train.py model.depth=10 와 같이 커맨드창에서 설정 오버라이드 가능
'''