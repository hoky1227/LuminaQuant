from __future__ import annotations

from lumina_quant import strategy_factory


def test_strategy_factory_exports_dispatcher_once() -> None:
    exported = strategy_factory.__all__

    assert exported.count("StrategySignalDispatcher") == 1
    assert strategy_factory.StrategySignalDispatcher.__name__ == "StrategySignalDispatcher"
