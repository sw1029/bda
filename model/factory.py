'''

yaml만 던져주고 모델 객체를 반환받기 위한 factory 함수
모델 설정은 config/registry.yaml에 정의된 내용을 참고

귀찮아서 바이브코딩 하긴 했는데, 유지보수 생각하면 별로 좋지는 않은듯

'''

import importlib
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


def get_model(*, model_cfg=None, registry_cfg=None, unknown_key_policy: str = "error", **kwargs) -> model:
    if model_cfg is None:
        raise ValueError("model_cfg가 필요합니다.")

    model_cfg = _unwrap_model_cfg(_to_dict(model_cfg))
    registry_cfg = _unwrap_registry_cfg(_to_dict(registry_cfg))

    if unknown_key_policy is None:
        unknown_key_policy = registry_cfg.get("unknown_key_policy", "error")

    model_name = model_cfg.get("name") or model_cfg.get("model_name")
    if not model_name:
        raise ValueError("model name이 필요합니다.")

    entry = registry_cfg.get(model_name)
    if not entry:
        raise ValueError(f"registry에 없는 모델: {model_name}")

    factory = _load_factory(entry.get("factory", ""))
    kwargs.update(_build_kwargs(model_cfg, entry, unknown_key_policy))

    return factory(**kwargs)
