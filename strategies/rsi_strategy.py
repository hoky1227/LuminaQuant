import logging

import numpy as np
import talib
from lumina_quant.compute.indicators import compute_rsi
from lumina_quant.config import BaseConfig
from lumina_quant.events import SignalEvent
from lumina_quant.strategy import Strategy

LOGGER = logging.getLogger(__name__)


class RsiStrategy(Strategy):
    """RSI Strategy:
    - Long when RSI < 30 (Oversold)
    - Exit when RSI > 70 (Overbought) or > 50 (Mean Reversion)
    """

    def __init__(self, bars, events, rsi_period=14, oversold=30, overbought=70):
        self.bars = bars
        self.events = events
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.symbol_list = self.bars.symbol_list
        self.bought = dict.fromkeys(self.symbol_list, "OUT")
        self.compute_backend = getattr(BaseConfig, "COMPUTE_BACKEND", "cpu")
        if str(self.compute_backend).lower() == "torch":
            self._rsi_fn = lambda values: compute_rsi(values, self.rsi_period, backend="torch")
        else:
            self._rsi_fn = lambda values: talib.RSI(values, timeperiod=self.rsi_period)

    def get_state(self):
        return {"bought": dict(self.bought)}

    def set_state(self, state):
        if "bought" in state and isinstance(state["bought"], dict):
            self.bought = state["bought"]

    def calculate_signals(self, event):
        if event.type == "MARKET":
            for s in self.symbol_list:
                # RSI needs at least period + 1 data points
                bars = self.bars.get_latest_bars_values(s, "close", N=self.rsi_period + 5)

                if len(bars) > self.rsi_period:
                    closes = np.asarray(bars, dtype=np.float64)
                    rsi_values = self._rsi_fn(closes)
                    current_rsi = rsi_values[-1]
                    if np.isnan(current_rsi):
                        continue

                    # Trading Logic
                    if current_rsi < self.oversold and self.bought[s] == "OUT":
                        LOGGER.info(
                            "LONG signal %s | RSI %.2f < %s",
                            s,
                            current_rsi,
                            self.oversold,
                        )
                        signal = SignalEvent(1, s, event.time, "LONG", 1.0)
                        self.events.put(signal)
                        self.bought[s] = "LONG"

                    elif current_rsi > self.overbought and self.bought[s] == "LONG":
                        LOGGER.info(
                            "EXIT signal %s | RSI %.2f > %s",
                            s,
                            current_rsi,
                            self.overbought,
                        )
                        signal = SignalEvent(1, s, event.time, "EXIT", 1.0)
                        self.events.put(signal)
                        self.bought[s] = "OUT"
