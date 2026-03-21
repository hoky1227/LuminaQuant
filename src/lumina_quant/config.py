"""Compatibility facade for runtime config access."""

from __future__ import annotations

import importlib
import os
import sys

from lumina_quant.configuration.loader import load_yaml_config
_runtime_access_module_name = "lumina_quant.configuration.runtime_access"
if _runtime_access_module_name in sys.modules:
    _runtime_access = importlib.reload(sys.modules[_runtime_access_module_name])
else:
    _runtime_access = importlib.import_module(_runtime_access_module_name)

BaseConfig = _runtime_access.BaseConfig
BacktestConfig = _runtime_access.BacktestConfig
LiveConfig = _runtime_access.LiveConfig
OptimizationConfig = _runtime_access.OptimizationConfig
current_market_data_runtime_settings = _runtime_access.current_market_data_runtime_settings
export_runtime_dict = _runtime_access.export_runtime_dict
load_current_runtime_config = _runtime_access.load_current_runtime_config
seed_runtime_env_defaults = _runtime_access.seed_runtime_env_defaults


def load_config(config_path: str = "config.yaml") -> dict:
    """Load raw YAML config."""
    return load_yaml_config(config_path=config_path)


__all__ = [
    "BacktestConfig",
    "BaseConfig",
    "LiveConfig",
    "OptimizationConfig",
    "current_market_data_runtime_settings",
    "export_runtime_dict",
    "load_config",
    "load_current_runtime_config",
    "os",
    "seed_runtime_env_defaults",
]
