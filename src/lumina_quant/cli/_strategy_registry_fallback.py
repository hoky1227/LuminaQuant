"""Shared strategy-registry fallback helpers for CLI entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from lumina_quant.strategy import Strategy


class PublicStubStrategy(Strategy):
    """Fallback strategy used when private strategy modules are unavailable."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Strategy modules are unavailable in this distribution.")

    def calculate_signals(self, event):
        _ = event
        return None


class PublicStrategyRegistry:
    """Minimal registry fallback used by public/lightweight installs."""

    DEFAULT_STRATEGY_NAME = "PublicStubStrategy"

    @staticmethod
    def get_strategy_map() -> dict[str, type[Strategy]]:
        return {"PublicStubStrategy": PublicStubStrategy}

    @staticmethod
    def resolve_strategy_class(
        name: str,
        default_name: str | None = None,
    ) -> type[Strategy]:
        _ = (name, default_name)
        return PublicStubStrategy

    @staticmethod
    def get_default_strategy_params(strategy_name: str) -> dict[str, Any]:
        _ = strategy_name
        return {}

    @staticmethod
    def resolve_strategy_params(
        strategy_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = strategy_name
        return dict(overrides or {})

    @staticmethod
    def get_default_optuna_config(strategy_name: str) -> dict[str, Any]:
        _ = strategy_name
        return {"n_trials": 20, "params": {}}

    @staticmethod
    def get_default_grid_config(strategy_name: str) -> dict[str, Any]:
        _ = strategy_name
        return {"params": {}}

    @staticmethod
    def resolve_optuna_config(
        strategy_name: str,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = strategy_name
        cfg = {"n_trials": 20, "params": {}}
        if isinstance(override, dict):
            cfg.update(override)
        return cfg

    @staticmethod
    def resolve_grid_config(
        strategy_name: str,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = strategy_name
        cfg = {"params": {}}
        if isinstance(override, dict):
            cfg.update(override)
        return cfg


def load_strategy_registry(importer: Callable[[], Any]) -> Any:
    """Load private strategy registry with a stable public fallback."""
    try:
        return importer()
    except Exception:
        return PublicStrategyRegistry()


def import_private_strategy_registry() -> Any:
    """Import the private/public strategy registry module when available."""
    return import_module("lumina_quant.strategies").registry
