'''
잡다하게 쓰는 함수는 여기다 모아놓을것

가시성 없는 코드는 죽음을 의미한다...
'''
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import random
import numpy as np
try:
    import torch
except Exception:
    torch = None

from contextlib import contextmanager


# 경로에서 타임스탬프 파싱하는 함수
def parse_timestamp(name: str, prefix: str) -> str:
        try:
            return name.split(prefix)[1].split('_')[0]
        except Exception:
            return datetime.now().strftime("%Y%m%d_%H%M%S")


# 시드 세팅용
@contextmanager
def set_seed(seed: int):
    if seed is None:
        # No-op context manager when no seed is provided.
        yield
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


    yield



    # 시드 복원
    random.seed()
    np.random.seed()
    if torch is not None:
        torch.manual_seed(torch.initial_seed())
        torch.cuda.manual_seed_all(torch.initial_seed())
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print(f"seed {seed} 에서의 실험 완료.")


def as_1d_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    return arr.reshape(-1)


def prob_positive(proba: Any) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1].reshape(-1)
    return proba.reshape(-1)


def save_args(args: dict | None, args_path: str | Path) -> None:
    if args is None:
        return
    OmegaConf.save(OmegaConf.create(args), str(args_path))


def load_args(args_path: str | Path) -> dict:
    cfg = OmegaConf.load(str(args_path))
    return OmegaConf.to_container(cfg, resolve=True)


def evaluate_metrics(
    *,
    trained_model: Any,
    data_valid: Any,
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
    y_true = as_1d_array(data_valid[target_label])

    model_obj = getattr(trained_model, "model", None)
    if model_obj is None:
        raise ValueError("trained_model.model이 설정되지 않았습니다.")

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    is_classifier = getattr(trained_model, "type", None) == "classifier" or hasattr(model_obj, "predict_proba")
    metrics: dict[str, Any] = {"n_valid": int(len(data_valid))}

    if is_classifier:
        y_proba_raw = None
        if hasattr(model_obj, "predict_proba"):
            y_proba_raw = model_obj.predict_proba(X_valid)

        y_prob_pos = None
        if y_proba_raw is not None:
            y_prob_pos = prob_positive(y_proba_raw)

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
                if finite.any():
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

    y_pred = as_1d_array(model_obj.predict(X_valid))
    metrics["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


def save_valid_metrics(
    *,
    trained_model: Any,
    data_valid: Any,
    id_label: Optional[str],
    target_label: str,
    artifact_dir: str | Path,
    file_prefix: str = "valid_metrics",
) -> dict:
    if data_valid is None:
        return {}

    timestamp = getattr(trained_model, "timestamp", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    payload: dict[str, Any] = {
        "timestamp": str(timestamp),
        "model_type": getattr(trained_model, "type", None),
    }
    try:
        payload["metrics"] = evaluate_metrics(
            trained_model=trained_model,
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


def to_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return dict(cfg)


def unwrap_model_cfg(model_cfg: dict) -> dict:
    if "name" not in model_cfg and "model" in model_cfg:
        inner = model_cfg.get("model") or {}
        if isinstance(inner, dict):
            return inner
    return model_cfg


def unwrap_registry_cfg(registry_cfg: dict) -> dict:
    registry = registry_cfg.get("registry")
    if isinstance(registry, dict):
        return registry
    return registry_cfg


def save_preprocess_config_artifact(
    *,
    log_dir: str | Path | None,
    model_cfg: Any,
    trained_model: Any,
    preprocess_config: Any,
    filename: str = "preprocess_config.yaml",
) -> Optional[Path]:
    if not log_dir:
        return None

    model_name = unwrap_model_cfg(to_dict(model_cfg)).get("name") or "model"
    model_name = str(model_name).strip() or "model"

    timestamp_val = getattr(trained_model, "timestamp", None)
    timestamp = str(timestamp_val).strip() if timestamp_val else ""
    if not timestamp:
        return None

    save_group_val = getattr(trained_model, "_optuna_group", None)
    save_group = str(save_group_val).strip() if save_group_val else ""

    artifact_dir = Path(log_dir) / model_name
    if save_group:
        artifact_dir = artifact_dir / save_group
    artifact_dir = artifact_dir / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = preprocess_config
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(preprocess_config):
            payload = asdict(preprocess_config)
    except Exception:
        pass

    save_args(to_dict(payload), artifact_dir / filename)
    return artifact_dir / filename


def load_factory(factory_path: str):
    if ":" not in factory_path:
        raise ValueError(f"factory 경로 오류: {factory_path}")
    module_path, attr = factory_path.split(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr)


def apply_unknown_policy(unknown_keys: list[str], policy: str, model_name: str) -> None:
    if not unknown_keys:
        return
    if policy == "drop":
        return
    raise ValueError(f"모델 파라미터 오류({model_name}): {sorted(unknown_keys)}")


def build_kwargs(model_cfg: dict, entry: dict, unknown_key_policy: str) -> dict:
    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    kwargs.pop("model_name", None)

    allow = set(entry.get("allow") or [])
    params_policy = entry.get("params_policy")

    if params_policy == "merge_dict":
        params = kwargs.pop("params", None) or {}
        if not isinstance(params, dict):
            raise ValueError("params는 dict 여야 합니다.")

        if allow:
            extra_keys = [k for k in kwargs.keys() if k not in allow]
            for key in extra_keys:
                params[key] = kwargs.pop(key)

        kwargs.update(params)
        return kwargs

    if allow:
        unknown = [k for k in kwargs.keys() if k not in allow]
        apply_unknown_policy(unknown, unknown_key_policy, model_cfg.get("name"))
        kwargs = {k: v for k, v in kwargs.items() if k in allow}

    return kwargs


def is_optuna_enabled(optuna_cfg: Optional[dict]) -> bool:
    if not isinstance(optuna_cfg, dict):
        return False
    return bool(optuna_cfg.get("enabled") or optuna_cfg.get("enable"))


def optuna_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_optuna_direction(metric: str) -> str:
    maximize = {"auc", "accuracy", "f1", "f1_score", "r2"}
    return "maximize" if (metric or "").lower() in maximize else "minimize"


def suggest_optuna_params(trial: Any, suggest_cfg: dict) -> dict:
    if not isinstance(suggest_cfg, dict):
        raise ValueError("optuna.suggest는 dict 여야 합니다.")

    params: dict = {}
    for name, spec in suggest_cfg.items():
        if not isinstance(spec, dict):
            raise ValueError(f"optuna.suggest.{name}는 dict 여야 합니다.")
        method = (spec.get("type") or spec.get("method") or spec.get("suggest") or "").lower()

        if method in {"int", "suggest_int"}:
            low = spec["low"]
            high = spec["high"]
            step = spec.get("step")
            log = bool(spec.get("log", False))
            if step is None:
                params[name] = trial.suggest_int(name, low, high, log=log)
            else:
                params[name] = trial.suggest_int(name, low, high, step=step, log=log)
            continue

        if method in {"float", "suggest_float"}:
            low = spec["low"]
            high = spec["high"]
            step = spec.get("step")
            log = bool(spec.get("log", False))
            if step is None:
                params[name] = trial.suggest_float(name, low, high, log=log)
            else:
                params[name] = trial.suggest_float(name, low, high, step=step, log=log)
            continue

        if method in {"categorical", "suggest_categorical"}:
            choices = spec.get("choices")
            if choices is None:
                raise ValueError(f"optuna.suggest.{name}.choices가 필요합니다.")
            params[name] = trial.suggest_categorical(name, choices)
            continue

        raise ValueError(f"지원하지 않는 optuna suggest 타입: {method} (param={name})")

    return params


def threshold_search_config(cfg: Any) -> tuple[bool, Optional[float], Optional[float], float]:
    if cfg is None:
        return True, None, None, 0.01
    if isinstance(cfg, bool):
        return cfg, None, None, 0.01
    if not isinstance(cfg, dict):
        return True, None, None, 0.01

    enabled = cfg.get("enabled")
    if enabled is None:
        enabled = cfg.get("enable")
    enabled = True if enabled is None else bool(enabled)

    low = cfg.get("low")
    high = cfg.get("high")
    step = float(cfg.get("step", 0.01))
    return enabled, low, high, step


def threshold_metric(cfg: Any, *, default: str = "f1") -> str:
    if isinstance(cfg, dict):
        metric = cfg.get("metric") or cfg.get("threshold_metric")
        if metric is not None:
            return str(metric)
    return default


def best_threshold(
    *,
    y_true: Any,
    y_score: Any,
    metric: str,
    threshold_cfg: Any = None,
) -> tuple[Optional[float], Optional[float]]:
    from sklearn.metrics import accuracy_score, f1_score

    metric = (metric or "").lower()
    if metric not in {"accuracy", "acc", "f1", "f1_score"}:
        return None, None

    enabled, low, high, step = threshold_search_config(threshold_cfg)
    if not enabled:
        return None, None

    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true/y_score 길이가 다릅니다.")

    score_min = float(np.nanmin(y_score)) if np.isfinite(np.nanmin(y_score)) else 0.0
    score_max = float(np.nanmax(y_score)) if np.isfinite(np.nanmax(y_score)) else 1.0

    if low is None:
        low = 0.0 if score_min >= 0.0 and score_max <= 1.0 else score_min
    if high is None:
        high = 1.0 if score_min >= 0.0 and score_max <= 1.0 else score_max

    low = float(low)
    high = float(high)
    if step <= 0:
        raise ValueError("threshold step은 0보다 커야 합니다.")

    if high < low:
        low, high = high, low

    thresholds = np.arange(low, high + step * 0.5, step, dtype=float)
    if thresholds.size == 0:
        return None, None

    best_th = None
    best_val = None
    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        if metric in {"accuracy", "acc"}:
            val = float(accuracy_score(y_true, y_pred))
        else:
            val = float(f1_score(y_true, y_pred, zero_division=0))
        if best_val is None or val > best_val:
            best_val = val
            best_th = float(th)

    return best_th, best_val


def evaluate_for_optuna(
    trained_model: Any,
    data_valid: Any,
    id_label: str,
    target_label: str,
    metric: str,
    *,
    threshold_cfg: Any = None,
    return_threshold: bool = False,
) -> float | tuple[float, Optional[float]]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    metric = (metric or "").lower()
    if not metric:
        raise ValueError("optuna.metric이 필요합니다.")

    if id_label is None or target_label is None:
        raise ValueError("id_label/target_label이 필요합니다.")

    X_valid = data_valid.drop(columns=[id_label, target_label])
    y_true = data_valid[target_label]

    is_classifier = getattr(trained_model, "type", None) == "classifier"
    model_obj = getattr(trained_model, "model", None)
    if model_obj is None:
        raise ValueError("trained_model.model이 없습니다.")

    if is_classifier:
        proba = None
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(X_valid)
        if proba is not None:
            proba = np.asarray(proba)
            y_prob = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
        else:
            y_prob = None

        y_pred = None
        if hasattr(model_obj, "predict"):
            y_pred = model_obj.predict(X_valid)
            y_pred = np.asarray(y_pred).reshape(-1)

        if metric in {"logloss", "log_loss"}:
            if y_prob is None:
                raise ValueError("logloss 계산을 위해 predict_proba가 필요합니다.")
            value = float(log_loss(y_true, y_prob))
            return (value, None) if return_threshold else value
        if metric in {"auc", "roc_auc"}:
            if y_prob is None:
                raise ValueError("auc 계산을 위해 predict_proba가 필요합니다.")
            value = float(roc_auc_score(y_true, y_prob))
            return (value, None) if return_threshold else value
        if metric in {"accuracy", "acc"}:
            if y_prob is not None:
                th, val = best_threshold(y_true=y_true, y_score=y_prob, metric=metric, threshold_cfg=threshold_cfg)
                if val is not None:
                    return (val, th) if return_threshold else val
            if y_pred is None:
                raise ValueError("accuracy 계산을 위해 predict가 필요합니다.")
            value = float(accuracy_score(y_true, y_pred))
            return (value, None) if return_threshold else value
        if metric in {"f1", "f1_score"}:
            if y_prob is not None:
                th, val = best_threshold(y_true=y_true, y_score=y_prob, metric=metric, threshold_cfg=threshold_cfg)
                if val is not None:
                    return (val, th) if return_threshold else val
            if y_pred is None:
                raise ValueError("f1 계산을 위해 predict가 필요합니다.")
            value = float(f1_score(y_true, y_pred))
            return (value, None) if return_threshold else value

        raise ValueError(f"지원하지 않는 metric: {metric}")

    # regressor
    if not hasattr(model_obj, "predict"):
        raise ValueError("regression metric 계산을 위해 predict가 필요합니다.")
    y_pred = model_obj.predict(X_valid)
    y_pred = np.asarray(y_pred).reshape(-1)

    if metric in {"auc", "roc_auc"}:
        value = float(roc_auc_score(y_true, y_pred))
        return (value, None) if return_threshold else value
    if metric in {"accuracy", "acc", "f1", "f1_score"}:
        th, val = best_threshold(y_true=y_true, y_score=y_pred, metric=metric, threshold_cfg=threshold_cfg)
        if val is None:
            raise ValueError("threshold 탐색 실패: y_pred를 확인하세요.")
        return (val, th) if return_threshold else val

    if metric in {"rmse"}:
        value = float(mean_squared_error(y_true, y_pred, squared=False))
        return (value, None) if return_threshold else value
    if metric in {"mae"}:
        value = float(mean_absolute_error(y_true, y_pred))
        return (value, None) if return_threshold else value
    if metric in {"r2"}:
        value = float(r2_score(y_true, y_pred))
        return (value, None) if return_threshold else value

    raise ValueError(f"지원하지 않는 metric: {metric}")
