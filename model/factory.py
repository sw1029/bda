'''

yaml만 던져주고 모델 객체를 반환받기 위한 factory 함수
모델 설정은 config/registry.yaml에 정의된 내용을 참고

귀찮아서 바이브코딩 하긴 했는데, 유지보수 생각하면 별로 좋지는 않은듯

'''

from pathlib import Path
from typing import Any, Callable, Optional

from model.base import model
from utils import (
    build_kwargs as _build_kwargs,
    evaluate_for_optuna as _evaluate,
    infer_optuna_direction as _infer_direction,
    is_optuna_enabled as _is_optuna_enabled,
    load_factory as _load_factory,
    optuna_run_timestamp as _optuna_run_timestamp,
    suggest_optuna_params as _suggest_params,
    threshold_metric as _threshold_metric,
    threshold_search_config as _threshold_search_config,
    to_dict as _to_dict,
    unwrap_model_cfg as _unwrap_model_cfg,
    unwrap_registry_cfg as _unwrap_registry_cfg,
)


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
        self.best_threshold: Optional[float] = None

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
        threshold_cfg = self._optuna_cfg.get("threshold")
        threshold_metric = _threshold_metric(threshold_cfg)
        threshold_enabled, _, _, _ = _threshold_search_config(threshold_cfg)

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
            value = None
            if threshold_enabled and (threshold_metric or "").lower() == (metric or "").lower():
                value, best_th = _evaluate(
                    candidate,
                    data_valid,
                    id_label,
                    target_label,
                    metric,
                    threshold_cfg=threshold_cfg,
                    return_threshold=True,
                )
                if best_th is not None:
                    trial.set_user_attr("best_threshold", float(best_th))
                    trial.set_user_attr("best_threshold_metric", str(threshold_metric))
                    trial.set_user_attr("best_threshold_value", float(value))
                return float(value)

            value = float(
                _evaluate(
                    candidate,
                    data_valid,
                    id_label,
                    target_label,
                    metric,
                    threshold_cfg=threshold_cfg,
                    return_threshold=False,
                )
            )

            if threshold_enabled:
                try:
                    th_value, best_th = _evaluate(
                        candidate,
                        data_valid,
                        id_label,
                        target_label,
                        threshold_metric,
                        threshold_cfg=threshold_cfg,
                        return_threshold=True,
                    )
                    if best_th is not None:
                        trial.set_user_attr("best_threshold", float(best_th))
                        trial.set_user_attr("best_threshold_metric", str(threshold_metric))
                        trial.set_user_attr("best_threshold_value", float(th_value))
                except Exception:
                    pass

            return float(value)

        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        self.best_params = dict(study.best_trial.params)
        self.best_threshold = study.best_trial.user_attrs.get("best_threshold")

        if save_dir is not None:
            best_dir = Path(save_dir) / self._model_name / self._optuna_group
            best_dir.mkdir(parents=True, exist_ok=True)
            from omegaconf import OmegaConf

            OmegaConf.save(OmegaConf.create(self.best_params), best_dir / "best_params.yaml")
            if self.best_threshold is not None:
                OmegaConf.save(OmegaConf.create({"best_threshold": float(self.best_threshold)}), best_dir / "best_threshold.yaml")
            try:
                import pandas as pd

                rows = []
                for t in study.trials:
                    row = {
                        "trial": t.number,
                        "value": t.value,
                        "best_threshold": t.user_attrs.get("best_threshold"),
                        "best_threshold_metric": t.user_attrs.get("best_threshold_metric"),
                        "best_threshold_value": t.user_attrs.get("best_threshold_value"),
                    }
                    row.update(t.params)
                    rows.append(row)
                pd.DataFrame(rows).to_csv(best_dir / "trials.csv", index=False)
            except Exception:
                pass

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
        if self.best_threshold is not None and "threshold" not in kwargs:
            kwargs["threshold"] = float(self.best_threshold)
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
