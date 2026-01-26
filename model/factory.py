'''

yaml만 던져주고 모델 객체를 반환받기 위한 factory 함수
모델 설정은 config/registry.yaml에 정의된 내용을 참고

귀찮아서 바이브코딩 하긴 했는데, 유지보수 생각하면 별로 좋지는 않은듯

'''

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from model.base import model


def _to_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return dict(cfg)


def _unwrap_model_cfg(model_cfg: dict) -> dict:
    if "name" not in model_cfg and "model" in model_cfg:
        inner = model_cfg.get("model") or {}
        if isinstance(inner, dict):
            return inner
    return model_cfg


def _unwrap_registry_cfg(registry_cfg: dict) -> dict:
    registry = registry_cfg.get("registry")
    if isinstance(registry, dict):
        return registry
    return registry_cfg


def _load_factory(factory_path: str):
    if ":" not in factory_path:
        raise ValueError(f"factory 경로 오류: {factory_path}")
    module_path, attr = factory_path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _apply_unknown_policy(unknown_keys, policy: str, model_name: str) -> None:
    if not unknown_keys:
        return
    if policy == "drop":
        return
    raise ValueError(f"모델 파라미터 오류({model_name}): {sorted(unknown_keys)}")


def _build_kwargs(model_cfg: dict, entry: dict, unknown_key_policy: str) -> dict:
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
        _apply_unknown_policy(unknown, unknown_key_policy, model_cfg.get("name"))
        kwargs = {k: v for k, v in kwargs.items() if k in allow}

    return kwargs


def _is_optuna_enabled(optuna_cfg: Optional[dict]) -> bool:
    if not isinstance(optuna_cfg, dict):
        return False
    return bool(optuna_cfg.get("enabled") or optuna_cfg.get("enable"))


def _optuna_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _infer_direction(metric: str) -> str:
    maximize = {"auc", "accuracy", "f1", "f1_score", "r2"}
    return "maximize" if (metric or "").lower() in maximize else "minimize"


def _suggest_params(trial: Any, suggest_cfg: dict) -> dict:
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


def _evaluate(trained_model: Any, data_valid, id_label: str, target_label: str, metric: str) -> float:
    import numpy as np
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
    if is_classifier:
        proba = None
        if hasattr(trained_model, "model") and hasattr(trained_model.model, "predict_proba"):
            proba = trained_model.model.predict_proba(X_valid)
        if proba is not None:
            proba = np.asarray(proba)
            y_prob = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
        else:
            y_prob = None

        y_pred = None
        if hasattr(trained_model, "model") and hasattr(trained_model.model, "predict"):
            y_pred = trained_model.model.predict(X_valid)
            y_pred = np.asarray(y_pred).reshape(-1)

        if metric in {"logloss", "log_loss"}:
            if y_prob is None:
                raise ValueError("logloss 계산을 위해 predict_proba가 필요합니다.")
            return float(log_loss(y_true, y_prob))
        if metric in {"auc", "roc_auc"}:
            if y_prob is None:
                raise ValueError("auc 계산을 위해 predict_proba가 필요합니다.")
            return float(roc_auc_score(y_true, y_prob))
        if metric in {"accuracy", "acc"}:
            if y_pred is None:
                raise ValueError("accuracy 계산을 위해 predict가 필요합니다.")
            return float(accuracy_score(y_true, y_pred))
        if metric in {"f1", "f1_score"}:
            if y_pred is None:
                raise ValueError("f1 계산을 위해 predict가 필요합니다.")
            return float(f1_score(y_true, y_pred))

        raise ValueError(f"지원하지 않는 metric: {metric}")

    # regressor
    if not hasattr(trained_model, "model") or not hasattr(trained_model.model, "predict"):
        raise ValueError("regression metric 계산을 위해 predict가 필요합니다.")
    y_pred = trained_model.model.predict(X_valid)
    y_pred = np.asarray(y_pred).reshape(-1)

    if metric in {"rmse"}:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    if metric in {"mae"}:
        return float(mean_absolute_error(y_true, y_pred))
    if metric in {"r2"}:
        return float(r2_score(y_true, y_pred))

    raise ValueError(f"지원하지 않는 metric: {metric}")


class OptunaTunedModel:
    def __init__(
        self,
        *,
        inner: model,
        model_name: str,
        factory: Callable[..., model],
        base_kwargs: dict,
        optuna_cfg: dict,
    ) -> None:
        self._inner = inner
        self._model_name = model_name
        self._factory = factory
        self._base_kwargs = dict(base_kwargs)
        self._optuna_cfg = optuna_cfg or {}
        self._optuna_group: Optional[str] = None
        self.best_params: Optional[dict] = None

    def __getattr__(self, item: str):
        return getattr(self._inner, item)

    def train(self, data_train, data_valid=None, save_dir: str = None, id_label: str = None, target_label: str = None, **kwargs) -> None:
        if not _is_optuna_enabled(self._optuna_cfg) or data_valid is None:
            self._inner.train(
                data_train=data_train,
                data_valid=data_valid,
                save_dir=save_dir,
                id_label=id_label,
                target_label=target_label,
                **kwargs,
            )
            return

        suggest_cfg = self._optuna_cfg.get("suggest") or {}
        if not suggest_cfg:
            self._inner.train(
                data_train=data_train,
                data_valid=data_valid,
                save_dir=save_dir,
                id_label=id_label,
                target_label=target_label,
                **kwargs,
            )
            return

        import optuna

        metric = self._optuna_cfg.get("metric")
        if metric is None:
            metric = "logloss" if getattr(self._inner, "type", None) == "classifier" else "rmse"

        direction = self._optuna_cfg.get("direction") or _infer_direction(metric)
        n_trials = int(self._optuna_cfg.get("n_trials", 20))
        seed = self._optuna_cfg.get("seed")

        sampler_name = str(self._optuna_cfg.get("sampler", "tpe")).lower()
        if sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)

        optuna_ts = _optuna_run_timestamp()
        self._optuna_group = f"optuna_{optuna_ts}"

        def objective(trial: Any) -> float:
            suggested = _suggest_params(trial, suggest_cfg)
            candidate_kwargs = dict(self._base_kwargs)
            candidate_kwargs.update(suggested)

            candidate = self._factory(**candidate_kwargs)
            candidate.train(
                data_train=data_train,
                data_valid=data_valid,
                save_dir=save_dir,
                save_group=self._optuna_group,
                id_label=id_label,
                target_label=target_label,
                **kwargs,
            )
            return _evaluate(candidate, data_valid, id_label, target_label, metric)

        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        self.best_params = dict(study.best_trial.params)

        if save_dir is not None:
            best_dir = Path(save_dir) / self._model_name / self._optuna_group
            best_dir.mkdir(parents=True, exist_ok=True)
            from omegaconf import OmegaConf

            OmegaConf.save(OmegaConf.create(self.best_params), best_dir / "best_params.yaml")

        final_kwargs = dict(self._base_kwargs)
        final_kwargs.update(self.best_params)
        final_model = self._factory(**final_kwargs)
        final_model.train(
            data_train=data_train,
            data_valid=data_valid,
            save_dir=save_dir,
            save_group=self._optuna_group,
            id_label=id_label,
            target_label=target_label,
            **kwargs,
        )
        self._inner = final_model

    def predict(self, input_data, save_dir: str = None, **kwargs):
        if self._optuna_group is not None and "save_group" not in kwargs:
            kwargs["save_group"] = self._optuna_group
        return self._inner.predict(input_data=input_data, save_dir=save_dir, **kwargs)


def get_model(*, model_cfg=None, registry_cfg=None, unknown_key_policy: str = "error", **kwargs) -> model:
    if model_cfg is None:
        raise ValueError("model_cfg가 필요합니다.")

    model_cfg = _unwrap_model_cfg(_to_dict(model_cfg))
    registry_cfg = _unwrap_registry_cfg(_to_dict(registry_cfg))

    if unknown_key_policy is None:
        unknown_key_policy = registry_cfg.get("unknown_key_policy", "error")

    optuna_cfg = None
    if isinstance(model_cfg, dict):
        optuna_cfg = model_cfg.pop("optuna", None)

    model_name = model_cfg.get("name") or model_cfg.get("model_name")
    if not model_name:
        raise ValueError("model name이 필요합니다.")

    entry = registry_cfg.get(model_name)
    if not entry:
        raise ValueError(f"registry에 없는 모델: {model_name}")

    factory = _load_factory(entry.get("factory", ""))
    kwargs.update(_build_kwargs(model_cfg, entry, unknown_key_policy))

    instance = factory(**kwargs)

    if _is_optuna_enabled(optuna_cfg):
        return OptunaTunedModel(
            inner=instance,
            model_name=model_name,
            factory=factory,
            base_kwargs=kwargs,
            optuna_cfg=optuna_cfg,
        )

    return instance
