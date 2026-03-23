"""Public strategy registry exports for CLI and dashboard modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_STRATEGY_NAME",
    "get_default_grid_config",
    "get_default_optuna_config",
    "get_default_strategy_params",
    "get_live_strategy_map",
    "get_live_strategy_names",
    "get_strategy_canonical_param_names",
    "get_strategy_map",
    "get_strategy_metadata",
    "get_strategy_names",
    "get_strategy_param_schema",
    "get_strategy_tier",
    "registry",
    "resolve_grid_config",
    "resolve_optuna_config",
    "resolve_strategy_class",
    "resolve_strategy_params",
]

_REGISTRY_EXPORTS = frozenset(name for name in __all__ if name != "registry")


def __getattr__(name: str) -> Any:
    """Lazily resolve registry exports to avoid package-level import cycles."""
    if name == "registry":
        module = import_module(f"{__name__}.registry")
        globals()[name] = module
        return module
    if name in _REGISTRY_EXPORTS:
        registry_module = import_module(f"{__name__}.registry")
        value = getattr(registry_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy registry exports in interactive/module introspection."""
    return sorted(set(globals()) | set(__all__))
