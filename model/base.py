from datetime import datetime
from pathlib import Path
import random
from typing import Any, Optional

import pandas as pd
from omegaconf import OmegaConf
import numpy as np

try:
    import torch
except Exception:
    torch = None

class model:
    def __init__(self):
        self.timestamp = None
        self.args = None
    def train(self, data_train:pd.DataFrame, data_valid:pd.DataFrame = None, **kwargs) -> None:
        pass
    def predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        # 귀찮은데 inference 기능은 여기 통합하는걸로
        pass
    def load(self, model_path:str) -> None:
        pass


    def _as_1d_array(self, values: Any) -> np.ndarray:
        arr = np.asarray(values)
        return arr.reshape(-1)

    def _prob_positive(self, proba: Any) -> np.ndarray:
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].reshape(-1)
        return proba.reshape(-1)

    def evaluate_metrics(
        self,
        *,
        data_valid: pd.DataFrame,
        id_label: Optional[str],
        target_label: str,
    ) -> dict:
        if target_label is None:
            raise ValueError("target_label이 필요합니다.")
        if target_label not in data_valid.columns:
            raise ValueError(f"data_valid에 target_label({target_label}) 컬럼이 없습니다.")

        drop_cols = [target_label]
        if id_label is not None and id_label in data_valid.columns:
            drop_cols.insert(0, id_label)

        X_valid = data_valid.drop(columns=drop_cols)
        y_true = self._as_1d_array(data_valid[target_label])

        model_obj = getattr(self, "model", None)
        if model_obj is None:
            raise ValueError("self.model이 설정되지 않았습니다.")

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )

        is_classifier = getattr(self, "type", None) == "classifier" or hasattr(model_obj, "predict_proba")
        metrics: dict[str, Any] = {"n_valid": int(len(data_valid))}

        if is_classifier:
            y_proba_raw = None
            if hasattr(model_obj, "predict_proba"):
                y_proba_raw = model_obj.predict_proba(X_valid)

            y_prob_pos = None
            if y_proba_raw is not None:
                y_prob_pos = self._prob_positive(y_proba_raw)

            y_pred = None
            if hasattr(model_obj, "predict"):
                y_pred_raw = np.asarray(model_obj.predict(X_valid))
                if y_pred_raw.ndim == 2:
                    if y_pred_raw.shape[1] == 1:
                        y_pred_raw = y_pred_raw[:, 0]
                    else:
                        y_pred_raw = np.argmax(y_pred_raw, axis=1)
                y_pred_raw = y_pred_raw.reshape(-1)
                if y_pred_raw.dtype.kind in {"f", "c"}:
                    finite = np.isfinite(y_pred_raw)
                    if not finite.any():
                        y_pred = None
                    else:
                        y_min = float(y_pred_raw[finite].min())
                        y_max = float(y_pred_raw[finite].max())
                        if 0.0 <= y_min and y_max <= 1.0:
                            y_pred = (y_pred_raw >= 0.5).astype(int)
                        else:
                            y_pred = np.rint(y_pred_raw).astype(int)
                else:
                    try:
                        y_pred = y_pred_raw.astype(int)
                    except Exception:
                        y_pred = y_pred_raw

            if y_proba_raw is not None:
                try:
                    if np.asarray(y_proba_raw).ndim == 1:
                        metrics["logloss"] = float(log_loss(y_true, y_prob_pos, labels=[0, 1]))
                    else:
                        metrics["logloss"] = float(log_loss(y_true, y_proba_raw))
                except Exception:
                    metrics["logloss"] = None

            if y_prob_pos is not None:
                try:
                    metrics["auc"] = float(roc_auc_score(y_true, y_prob_pos))
                except Exception:
                    metrics["auc"] = None

            if y_pred is not None:
                try:
                    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                except Exception:
                    metrics["accuracy"] = None
                try:
                    metrics["f1"] = float(f1_score(y_true, y_pred))
                except Exception:
                    metrics["f1"] = None

            return metrics

        if not hasattr(model_obj, "predict"):
            raise ValueError("regression metric 계산을 위해 predict가 필요합니다.")

        y_pred = self._as_1d_array(model_obj.predict(X_valid))
        metrics["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        return metrics

    def save_valid_metrics(
        self,
        *,
        data_valid: pd.DataFrame,
        id_label: Optional[str],
        target_label: str,
        artifact_dir: str | Path,
        file_prefix: str = "valid_metrics",
    ) -> dict:
        if data_valid is None:
            return {}

        timestamp = self.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "timestamp": str(timestamp),
            "model_type": getattr(self, "type", None),
        }
        try:
            payload["metrics"] = self.evaluate_metrics(
                data_valid=data_valid,
                id_label=id_label,
                target_label=target_label,
            )
        except Exception as exc:
            payload["metrics"] = {}
            payload["error"] = str(exc)

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = artifact_dir / f"{file_prefix}_{timestamp}.yaml"
        OmegaConf.save(OmegaConf.create(payload), metrics_path)
        return payload

    
    def save_args(self, args: dict, args_path: str) -> None:
        if args is None:
                return
        OmegaConf.save(OmegaConf.create(args), args_path)

    def load_args(self, args_path: str) -> dict:
        cfg = OmegaConf.load(args_path)
        self.args = OmegaConf.to_container(cfg, resolve=True)
        return self.args

    def set_seed(self, seed: int) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
