from __future__ import annotations

import hydra
from omegaconf import DictConfig

from data import PreprocessConfig, preprocess_data
from model import factory  # 바이브 코딩이 좋더라
from utils import save_preprocess_config_artifact, set_seed, to_dict

from sklearn.model_selection import train_test_split

@hydra.main(version_base=None, config_path="config", config_name="config") # hydra 데코레이터를 통해 실험 설정 관리
def main(cfg: DictConfig) -> None:

    model = factory.get_model(
        model_cfg=cfg.model,
        registry_cfg=cfg.registry,
        unknown_key_policy=cfg.get("unknown_key_policy", "error"),
    )

    data_args = to_dict(cfg.get("preprocess"))
    data_cfg = PreprocessConfig(**data_args) if data_args else PreprocessConfig()
    # --- 데이터 전처리 부분 ---
    data, y, cat_features, text_features = preprocess_data(
        cfg.train_csv,
        is_train=True,
        id_label=cfg.id_label,
        target_label=cfg.target_label,
        config=data_cfg,
    )
    # --- 데이터 전처리 완료 ---

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
                    id_label=cfg.id_label, target_label=cfg.target_label,
                    save_dir=cfg.log_dir) 
        # 데이터 안넣어주면 에러남. 데이터 지정해줄것.
        # valid 데이터가 None이면 내부에서 알아서 처리함. 최종 inference 목적으로 쓰는걸 추천.

    if cfg.get("log_dir"):
        save_preprocess_config_artifact(
            log_dir=cfg.log_dir,
            model_cfg=cfg.model,
            trained_model=model,
            preprocess_config=data_cfg,
        )
    
    
    if cfg.get("do_inference"):
        # 추론용 데이터 전처리
        # train.py 추론 쪽 (데이터 전처리 부분으로 간주)
        data_test, _, _, _ = preprocess_data(
            cfg.inference_csv, is_train=False, 
            train_cols=data.columns, 
            id_label=cfg.id_label
        )
        args = {}
        threshold = cfg.get("threshold", 0.5)
        if threshold is not None:
            args["threshold"] = float(threshold)
        pred = model.predict(input_data=data_test, save_dir=cfg.log_dir, **args)

if __name__ == "__main__":
    main()


'''
실제 사용 시 python train.py model.model.depth=10 preprocess.text_mode=T2 와 같이 커맨드창에서 설정 오버라이드 가능
'''
