import queue
import unittest
from dataclasses import dataclass

from lumina_quant.core.events import MarketEvent
from lumina_quant.strategies.topcap_tsmom import TopCapTimeSeriesMomentumStrategy


@dataclass
class _MomentumBarStore:
    symbol_list: list[str]

    def __post_init__(self):
        self._latest_close = dict.fromkeys(self.symbol_list)
        self._latest_time = dict.fromkeys(self.symbol_list)

    def set_bar(self, symbol, time_index, close_price):
        self._latest_time[symbol] = time_index
        self._latest_close[symbol] = float(close_price)

    def get_latest_bar_value(self, symbol, value_type):
        _ = value_type
        value = self._latest_close.get(symbol)
        return float(value) if value is not None else 0.0

    def get_latest_bar_datetime(self, symbol):
        return self._latest_time.get(symbol)


def _build_price_rows(length=140):
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
    ]
    prices = {
        "BTC/USDT": 40000.0,
        "ETH/USDT": 2500.0,
        "BNB/USDT": 350.0,
        "SOL/USDT": 100.0,
        "XRP/USDT": 1.0,
        "ADA/USDT": 0.8,
    }

    rows = []
    split_point = int(length // 2)
    for idx in range(length):
        if idx < split_point:
            step = {
                "BTC/USDT": 0.0010,
                "ETH/USDT": 0.0030,
                "BNB/USDT": 0.0025,
                "SOL/USDT": 0.0020,
                "XRP/USDT": -0.0025,
                "ADA/USDT": -0.0020,
            }
        else:
            step = {
                "BTC/USDT": -0.0008,
                "ETH/USDT": -0.0028,
                "BNB/USDT": -0.0020,
                "SOL/USDT": -0.0023,
                "XRP/USDT": 0.0028,
                "ADA/USDT": 0.0022,
            }

        frame = {}
        for symbol in symbols:
            prices[symbol] = prices[symbol] * (1.0 + step[symbol])
            frame[symbol] = prices[symbol]
        rows.append((idx, frame))
    return symbols, rows


def _build_common_factor_rows(length=60):
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
    ]
    prices = {
        "BTC/USDT": 40000.0,
        "ETH/USDT": 2500.0,
        "BNB/USDT": 350.0,
        "SOL/USDT": 100.0,
        "XRP/USDT": 1.0,
        "ADA/USDT": 0.8,
    }
    rows = []
    for idx in range(length):
        frame = {}
        for symbol in symbols:
            prices[symbol] = prices[symbol] * 1.015
            frame[symbol] = prices[symbol]
        rows.append((idx, frame))
    return symbols, rows


def _build_benchmark_crash_rows(length=120):
    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
    ]
    prices = {
        "BTC/USDT": 40000.0,
        "ETH/USDT": 2500.0,
        "BNB/USDT": 350.0,
        "SOL/USDT": 100.0,
        "XRP/USDT": 1.0,
        "ADA/USDT": 0.8,
    }
    rows = []
    crash_start = length // 3
    for idx in range(length):
        if idx < crash_start:
            step = {
                "BTC/USDT": 0.0020,
                "ETH/USDT": 0.0030,
                "BNB/USDT": 0.0025,
                "SOL/USDT": 0.0025,
                "XRP/USDT": -0.0010,
                "ADA/USDT": -0.0010,
            }
        else:
            step = {
                "BTC/USDT": -0.0120,
                "ETH/USDT": 0.0025,
                "BNB/USDT": 0.0022,
                "SOL/USDT": 0.0020,
                "XRP/USDT": -0.0010,
                "ADA/USDT": -0.0010,
            }
        frame = {}
        for symbol in symbols:
            prices[symbol] = prices[symbol] * (1.0 + step[symbol])
            frame[symbol] = prices[symbol]
        rows.append((idx, frame))
    return symbols, rows


def _run_strategy(rows, symbols, split=None, param_overrides=None):
    params = {
        "lookback_bars": 8,
        "rebalance_bars": 2,
        "signal_threshold": 0.01,
        "stop_loss_pct": 0.08,
        "max_longs": 2,
        "max_shorts": 2,
        "min_price": 0.1,
        "btc_regime_ma": 0,
        "btc_symbol": "BTC/USDT",
    }
    if param_overrides:
        params.update(param_overrides)

    def _feed(strategy, bars, events, chunk):
        for time_index, frame in chunk:
            for symbol in symbols:
                price = frame[symbol]
                bars.set_bar(symbol, time_index, price)
                event = MarketEvent(time_index, symbol, price, price, price, price, 1000.0)
                strategy.calculate_signals(event)

        out = []
        while not events.empty():
            signal = events.get()
            out.append((int(signal.datetime), str(signal.symbol), str(signal.signal_type)))
        return out

    if split is None:
        bars = _MomentumBarStore(symbols)
        events = queue.Queue()
        strategy = TopCapTimeSeriesMomentumStrategy(bars, events, **params)
        return _feed(strategy, bars, events, rows)

    chunk_a = rows[:split]
    chunk_b = rows[split:]

    bars_a = _MomentumBarStore(symbols)
    events_a = queue.Queue()
    strategy_a = TopCapTimeSeriesMomentumStrategy(bars_a, events_a, **params)
    signals_a = _feed(strategy_a, bars_a, events_a, chunk_a)
    saved_state = strategy_a.get_state()

    bars_b = _MomentumBarStore(symbols)
    events_b = queue.Queue()
    strategy_b = TopCapTimeSeriesMomentumStrategy(bars_b, events_b, **params)
    strategy_b.set_state(saved_state)
    signals_b = _feed(strategy_b, bars_b, events_b, chunk_b)
    return [*signals_a, *signals_b]


class TestTopCapTimeSeriesMomentumStrategy(unittest.TestCase):
    def test_generates_long_short_and_exit(self):
        symbols, rows = _build_price_rows()
        signals = _run_strategy(rows, symbols)
        self.assertGreater(len(signals), 0)

        signal_types = [signal_type for _, _, signal_type in signals]
        self.assertIn("LONG", signal_types)
        self.assertIn("SHORT", signal_types)
        self.assertIn("EXIT", signal_types)

    def test_state_roundtrip_preserves_signal_sequence(self):
        symbols, rows = _build_price_rows()
        full_signals = _run_strategy(rows, symbols)
        split_signals = _run_strategy(rows, symbols, split=80)
        self.assertEqual(full_signals, split_signals)

    def test_supports_long_only_rotation_without_short_signals(self):
        symbols, rows = _build_price_rows()
        signals = _run_strategy(
            rows,
            symbols,
            param_overrides={
                "max_longs": 2,
                "max_shorts": 0,
            },
        )

        signal_types = [signal_type for _, _, signal_type in signals]
        self.assertIn("LONG", signal_types)
        self.assertIn("EXIT", signal_types)
        self.assertNotIn("SHORT", signal_types)

    def test_residualize_btc_blocks_common_factor_only_signals(self):
        symbols, rows = _build_common_factor_rows()
        raw_signals = _run_strategy(rows, symbols)
        residual_signals = _run_strategy(
            rows,
            symbols,
            param_overrides={
                "residualize_btc": True,
            },
        )

        raw_signal_types = [signal_type for _, _, signal_type in raw_signals]
        residual_signal_types = [signal_type for _, _, signal_type in residual_signals]

        self.assertIn("LONG", raw_signal_types)
        self.assertNotIn("LONG", residual_signal_types)
        self.assertNotIn("SHORT", residual_signal_types)

    def test_benchmark_drawdown_gate_switches_regime_to_risk_off(self):
        symbols, _ = _build_benchmark_crash_rows()
        bars = _MomentumBarStore(symbols)
        strategy = TopCapTimeSeriesMomentumStrategy(
            bars,
            queue.Queue(),
            lookback_bars=32,
            benchmark_drawdown_window=24,
            benchmark_drawdown_limit=0.08,
            btc_regime_ma=0,
        )

        history = strategy._price_history["BTC/USDT"]
        for value in [100.0 + idx for idx in range(24)]:
            history.append(value)
        history.append(103.0)
        history.append(100.0)
        history.append(96.0)
        history.append(92.0)
        history.append(88.0)

        self.assertEqual(strategy._btc_regime(), "RISK_OFF")

    def test_take_profit_attaches_to_entry_signals(self):
        symbols, rows = _build_price_rows(length=40)
        bars = _MomentumBarStore(symbols)
        events = queue.Queue()
        strategy = TopCapTimeSeriesMomentumStrategy(
            bars,
            events,
            lookback_bars=8,
            rebalance_bars=2,
            signal_threshold=0.01,
            take_profit_pct=0.10,
            btc_regime_ma=0,
        )

        for time_index, frame in rows:
            for symbol in symbols:
                price = frame[symbol]
                bars.set_bar(symbol, time_index, price)
                event = MarketEvent(time_index, symbol, price, price, price, price, 1000.0)
                strategy.calculate_signals(event)

        signals = []
        while not events.empty():
            signals.append(events.get())

        long_signal = next(signal for signal in signals if str(signal.signal_type) == "LONG")
        assert long_signal.take_profit is not None
        assert long_signal.take_profit > 0.0


if __name__ == "__main__":
    unittest.main()
