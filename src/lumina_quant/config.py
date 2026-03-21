"""Lightweight compatibility facade for runtime config access."""

from __future__ import annotations

import importlib

_RUNTIME_ACCESS_MODULE = "lumina_quant.configuration.runtime_access"
_PUBLIC_NAMES = {
    "BaseConfig",
    "BacktestConfig",
    "LiveConfig",
    "OptimizationConfig",
    "current_market_data_runtime_settings",
    "export_runtime_dict",
    "load_current_runtime_config",
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load raw YAML config."""
    from lumina_quant.configuration.loader import load_yaml_config

    return load_yaml_config(config_path=config_path)


def _runtime_access_module():
    return importlib.import_module(_RUNTIME_ACCESS_MODULE)


def __getattr__(name: str):
    if name in _PUBLIC_NAMES:
        return getattr(_runtime_access_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PUBLIC_NAMES)


__all__ = ["load_config", *sorted(_PUBLIC_NAMES)]
