from __future__ import annotations

import hydra
from omegaconf import DictConfig
from pathlib import Path

import numpy as np
import pandas as pd

from data import PreprocessConfig, preprocess_data
from model import factory  # 바이브 코딩이 좋더라
from utils import evaluate_for_optuna, prob_positive, save_preprocess_config_artifact, set_seed, threshold_metric, to_dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

        def predict_scores(input_df: pd.DataFrame) -> np.ndarray:
            model_obj = getattr(model, "model", None) or model
            drop_cols = [cfg.id_label]
            if cfg.target_label in input_df.columns:
                drop_cols.append(cfg.target_label)
            X = input_df.drop(columns=drop_cols, errors="ignore")
            score = None
            if hasattr(model_obj, "predict_proba"):
                try:
                    score = prob_positive(model_obj.predict_proba(X))
                except Exception:
                    score = None
            if score is None:
                score = np.asarray(model_obj.predict(X)).reshape(-1)
            return np.asarray(score).reshape(-1).astype(float)

        def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, fallback: float) -> float:
            thresholds = np.unique(scores)
            best_t = float(fallback)
            best = -1.0
            for t in thresholds:
                pred_local = (scores >= float(t)).astype(int)
                f1 = f1_score(y_true, pred_local, zero_division=0)
                if f1 > best:
                    best = float(f1)
                    best_t = float(t)
            return best_t

        args = {}
        threshold = cfg.get("threshold", 0.5)
        optuna_group = getattr(model, "_optuna_group", None)
        best_params = getattr(model, "best_params", None)
        best_threshold = getattr(model, "best_threshold", None)
        if optuna_group and best_params is not None and best_threshold is None and data_valid is not None:
            try:
                model_cfg = to_dict(cfg.get("model"))
                inner_model_cfg = model_cfg.get("model") if isinstance(model_cfg, dict) else {}
                if not isinstance(inner_model_cfg, dict):
                    inner_model_cfg = {}
                optuna_cfg = inner_model_cfg.get("optuna") if isinstance(inner_model_cfg.get("optuna"), dict) else None
                threshold_cfg = optuna_cfg.get("threshold") if isinstance(optuna_cfg, dict) else None
                th_metric = threshold_metric(threshold_cfg)
                _, best_th = evaluate_for_optuna(
                    model,
                    data_valid,
                    cfg.id_label,
                    cfg.target_label,
                    th_metric,
                    threshold_cfg=threshold_cfg,
                    return_threshold=True,
                )
                if best_th is not None:
                    best_threshold = float(best_th)
                    setattr(model, "best_threshold", best_threshold)
            except Exception:
                pass
        if optuna_group and best_params is not None and best_threshold is not None:
            threshold = float(best_threshold)

        cluster_cfg = to_dict(cfg.get("cluster_threshold")) if cfg.get("cluster_threshold") is not None else {}
        cluster_enabled = bool(cluster_cfg.get("enabled", False))
        cluster_col = str(cluster_cfg.get("cluster_col", "missing_cluster_id") or "missing_cluster_id")
        min_cluster_size = int(cluster_cfg.get("min_cluster_size", 30) or 30)
        global_threshold = float(threshold) if threshold is not None else 0.5

        cluster_thresholds: dict[int, float] = {}
        if (
            cluster_enabled
            and data_valid is not None
            and cluster_col in data_valid.columns
            and cluster_col in data_test.columns
        ):
            try:
                scores_valid = predict_scores(data_valid)
                y_valid = np.asarray(data_valid[cfg.target_label]).astype(int).reshape(-1)
                clusters_valid = np.asarray(data_valid[cluster_col]).reshape(-1)
                clusters_valid = np.round(clusters_valid).astype(int)

                for k in np.unique(clusters_valid):
                    mask = clusters_valid == int(k)
                    if int(mask.sum()) < min_cluster_size:
                        continue
                    cluster_thresholds[int(k)] = best_f1_threshold(y_valid[mask], scores_valid[mask], global_threshold)
            except Exception:
                cluster_thresholds = {}

        if cluster_thresholds:
            scores_test = predict_scores(data_test)
            clusters_test = np.asarray(data_test[cluster_col]).reshape(-1)
            clusters_test = np.round(clusters_test).astype(int)
            th_arr = np.full(len(scores_test), float(global_threshold), dtype=float)
            for k, t in cluster_thresholds.items():
                th_arr[clusters_test == int(k)] = float(t)
            preds = (scores_test >= th_arr).astype(int)
            pred = pd.DataFrame({cfg.id_label: data_test[cfg.id_label], cfg.target_label: preds})
        else:
            if threshold is not None:
                args["threshold"] = float(threshold)
            pred = model.predict(input_data=data_test, save_dir=cfg.log_dir, **args)

        if optuna_group and best_params is not None and cfg.get("log_dir"):
            model_cfg = to_dict(cfg.get("model"))
            inner_model_cfg = model_cfg.get("model") if isinstance(model_cfg, dict) else {}
            if not isinstance(inner_model_cfg, dict):
                inner_model_cfg = {}
            model_name = inner_model_cfg.get("name") or inner_model_cfg.get("model_name") or "model"
            model_name = str(model_name).strip() or "model"
            best_dir = Path(str(cfg.log_dir)) / model_name / str(optuna_group)
            best_dir.mkdir(parents=True, exist_ok=True)
            pred.to_csv(best_dir / "submission.csv", index=False)

if __name__ == "__main__":
    main()


'''
실제 사용 시 python train.py model.model.depth=10 preprocess.text_mode=T2 와 같이 커맨드창에서 설정 오버라이드 가능
'''
