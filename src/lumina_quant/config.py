"""Compatibility facade for runtime config access."""

from __future__ import annotations

import os

from lumina_quant.configuration.loader import load_yaml_config
from lumina_quant.configuration.runtime_access import (
    BaseConfig,
    BacktestConfig,
    LiveConfig,
    OptimizationConfig,
    current_market_data_runtime_settings,
    export_runtime_dict,
    load_current_runtime_config,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load raw YAML config."""
    return load_yaml_config(config_path=config_path)


__all__ = [
    "BaseConfig",
    "BacktestConfig",
    "LiveConfig",
    "OptimizationConfig",
    "current_market_data_runtime_settings",
    "export_runtime_dict",
    "load_config",
    "load_current_runtime_config",
    "os",
]
