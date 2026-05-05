from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.hourly_shock_reversion import HourlyShockReversionStrategy
from lumina_quant.timeframe_aggregator import TimeframeAggregator


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


def _event(when: datetime, close: float) -> MarketEvent:
    return MarketEvent(
        time=when,
        symbol="ETH/USDT",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _step(strategy, aggregator, when: datetime, close: float) -> None:
    event = _event(when, close)
    aggregator.update_from_1s_batch("ETH/USDT", [event])
    strategy.calculate_signals_window(event, aggregator)


def test_hourly_shock_reversion_fades_completed_negative_shock_once() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        max_hold_bars=48,
        target_allocation=0.008,
        max_order_value=175.0,
        stop_loss_pct=0.02,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for offset, close in enumerate([100.0, 100.0, 100.0, 100.0, 99.0, 99.0]):
        _step(strategy, aggregator, start + timedelta(hours=offset), close)

    assert len(queue.items) == 1
    signal = queue.items[0]
    assert signal.signal_type == "LONG"
    assert signal.symbol == "ETH/USDT"
    assert signal.strength == 0.008
    assert signal.price == 99.0
    assert signal.stop_loss == 99.0 * 0.98
    assert signal.metadata["reason"] == "negative_shock_reversion_long"
    assert signal.metadata["shock_return"] < -0.006
    assert signal.metadata["max_order_value"] == 175.0

    strategy.calculate_signals_window(
        _event(start + timedelta(hours=5, minutes=1), 99.0),
        aggregator,
    )
    assert len(queue.items) == 1


def test_hourly_shock_reversion_shorts_positive_shock_and_consumes_exit_bar() -> None:
    queue = _Queue()
    strategy = HourlyShockReversionStrategy(
        SimpleNamespace(symbol_list=["ETH/USDT"]),
        queue,
        lookback_bars=4,
        return_threshold=0.006,
        max_hold_bars=2,
        target_allocation=0.008,
        max_order_value=175.0,
        stop_loss_pct=0.02,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    start = datetime(2026, 1, 1, tzinfo=UTC)

    for offset, close in enumerate([100.0, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 101.0]):
        _step(strategy, aggregator, start + timedelta(hours=offset), close)

    assert [item.signal_type for item in queue.items] == ["SHORT", "EXIT"]
    entry, exit_signal = queue.items
    assert entry.price == 101.0
    assert entry.stop_loss == 101.0 * 1.02
    assert entry.metadata["reason"] == "positive_shock_reversion_short"
    assert exit_signal.metadata["reason"] == "max_hold_exit"
