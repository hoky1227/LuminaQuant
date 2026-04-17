import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from lumina_quant.core.events import MarketEvent, SignalEvent

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lumina_quant"
    / "strategies"
    / "artifact_portfolio_mode.py"
)
SPEC = importlib.util.spec_from_file_location("artifact_portfolio_mode", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_portfolio_mode_strategy_forwards_component_weighted_signals(monkeypatch) -> None:
    class _ChildStrategy:
        required_timeframes = ("1h",)

        def __init__(self, bars, events, **params):
            _ = bars, params
            self.events = events

        def calculate_signals(self, event):
            self.events.put(
                SignalEvent(
                    strategy_id="child",
                    symbol="BNB/USDT",
                    datetime=event.time,
                    signal_type="LONG",
                )
            )

    monkeypatch.setattr(
        MODULE,
        "resolve_portfolio_mode_definition",
        lambda portfolio_mode: MODULE.PortfolioModeDefinition(
            portfolio_mode=portfolio_mode,
            components=(
                MODULE.PortfolioModeComponent(
                    component_id="comp-a",
                    label="component-a",
                    strategy_class="MovingAverageCrossStrategy",
                    symbols=("BNB/USDT",),
                    params={},
                    weight=0.3,
                    source="test",
                ),
            ),
            cash_weight=0.7,
            source_artifacts={},
        ),
    )
    monkeypatch.setattr(MODULE, "resolve_strategy_class", lambda name, default_name=None: _ChildStrategy)

    events = []
    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0),
        events=SimpleNamespace(put=lambda item: events.append(item)),
        portfolio_mode="hybrid_guarded_mode",
    )
    strategy.calculate_signals(
        MarketEvent(
            time="2026-04-17T00:00:00Z",
            symbol="BNB/USDT",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
    )

    assert len(events) == 1
    signal = events[0]
    assert signal.metadata["component_id"] == "comp-a"
    assert signal.metadata["target_allocation_scale"] == 0.3
    assert signal.client_order_id.startswith("LQPM-") or signal.client_order_id.startswith("comp-a-")

