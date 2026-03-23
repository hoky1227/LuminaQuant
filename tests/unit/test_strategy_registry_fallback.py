from __future__ import annotations

from lumina_quant.cli._strategy_registry_fallback import (
    PublicStrategyRegistry,
    PublicStubStrategy,
    import_private_strategy_registry,
    load_strategy_registry,
)
import lumina_quant.strategies as strategies_package


def test_import_private_strategy_registry_resolves_registry_module():
    registry_module = import_private_strategy_registry()

    assert registry_module.__name__ == "lumina_quant.strategies.registry"
    assert registry_module.DEFAULT_STRATEGY_NAME == strategies_package.DEFAULT_STRATEGY_NAME


def test_load_strategy_registry_uses_public_fallback_when_import_fails():
    def _raise():
        raise RuntimeError("boom")

    registry = load_strategy_registry(_raise)

    assert isinstance(registry, PublicStrategyRegistry)
    assert registry.DEFAULT_STRATEGY_NAME == "PublicStubStrategy"
    assert registry.get_strategy_map() == {"PublicStubStrategy": PublicStubStrategy}


def test_strategy_package_exports_registry_attribute():
    assert strategies_package.registry.__name__ == "lumina_quant.strategies.registry"
