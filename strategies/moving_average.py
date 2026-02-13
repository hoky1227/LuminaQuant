import logging

import numpy as np
import talib
from lumina_quant.compute.indicators import compute_sma
from lumina_quant.config import BaseConfig

from lumina_quant.events import SignalEvent
from lumina_quant.strategy import Strategy

LOGGER = logging.getLogger(__name__)


class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a short/long simple weighted moving average.
    """

    def __init__(self, bars, events, short_window=10, long_window=30):
        self.bars = bars
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        self.symbol_list = self.bars.symbol_list
        self.bought = {s: "OUT" for s in self.symbol_list}
        self.compute_backend = getattr(BaseConfig, "COMPUTE_BACKEND", "cpu")
        if str(self.compute_backend).lower() == "torch":
            self._short_ma_fn = lambda values: compute_sma(
                values, self.short_window, backend="torch"
            )
            self._long_ma_fn = lambda values: compute_sma(values, self.long_window, backend="torch")
        else:
            self._short_ma_fn = lambda values: talib.SMA(values, timeperiod=self.short_window)
            self._long_ma_fn = lambda values: talib.SMA(values, timeperiod=self.long_window)

    def get_state(self):
        return {"bought": dict(self.bought)}

    def set_state(self, state):
        if "bought" in state:
            self.bought = state["bought"]

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars_values(
                    s, "close", N=self.long_window + 1
                )  # Get enough data

                if len(bars) > self.long_window:
                    closes = np.asarray(bars, dtype=np.float64)
                    ma_short = self._short_ma_fn(closes)
                    ma_long = self._long_ma_fn(closes)

                    symbol = s
                    curr_short = ma_short[-1]
                    curr_long = ma_long[-1]
                    if np.isnan(curr_short) or np.isnan(curr_long):
                        continue

                    # Trading logic
                    if curr_short > curr_long and self.bought[s] == "OUT":
                        LOGGER.info(
                            "LONG signal %s | short %.5f > long %.5f",
                            s,
                            curr_short,
                            curr_long,
                        )
                        sig_dir = "LONG"
                        signal = SignalEvent(1, symbol, event.time, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = "LONG"

                    elif curr_short < curr_long and self.bought[s] == "LONG":
                        LOGGER.info(
                            "EXIT signal %s | short %.5f < long %.5f",
                            s,
                            curr_short,
                            curr_long,
                        )
                        sig_dir = "EXIT"
                        signal = SignalEvent(1, symbol, event.time, sig_dir, 1.0)
                        self.events.put(signal)
                        self.bought[s] = "OUT"
