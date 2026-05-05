"""Hourly single-asset shock mean-reversion strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumina_quant.core.events import SignalEvent
from lumina_quant.indicators.common import safe_float
from lumina_quant.strategy import Strategy
from lumina_quant.symbols import canonical_symbol
from lumina_quant.tuning import HyperParam, resolve_params_from_schema


@dataclass(slots=True)
class _PositionState:
    mode: str = "OUT"
    entry_price: float | None = None
    bars_held: int = 0
    last_completed_bar_key: str = ""


class HourlyShockReversionStrategy(Strategy):
    """Fade a large completed hourly move in one liquid crypto symbol."""

    uses_timeframe_aggregator = True
    required_timeframes = ("1h",)
    required_lookbacks = {"1h": 128}

    @classmethod
    def get_param_schema(cls) -> dict[str, HyperParam]:
        return {
            "target_symbol": HyperParam.string("target_symbol", default="ETH/USDT", tunable=False),
            "timeframe": HyperParam.string("timeframe", default="1h", tunable=False),
            "lookback_bars": HyperParam.integer("lookback_bars", default=4, low=1, high=240),
            "return_threshold": HyperParam.floating(
                "return_threshold", default=0.006, low=0.0, high=1.0
            ),
            "max_hold_bars": HyperParam.integer("max_hold_bars", default=48, low=1, high=240),
            "target_allocation": HyperParam.floating(
                "target_allocation", default=0.008, low=0.0, high=1.0, tunable=False
            ),
            "max_order_value": HyperParam.floating(
                "max_order_value", default=175.0, low=0.0, high=1_000_000.0, tunable=False
            ),
            "stop_loss_pct": HyperParam.floating(
                "stop_loss_pct", default=0.02, low=0.0, high=1.0, tunable=False
            ),
            "take_profit_pct": HyperParam.floating(
                "take_profit_pct", default=0.0, low=0.0, high=1.0, tunable=False
            ),
            "allow_long": HyperParam.boolean("allow_long", default=True, grid=[True, False]),
            "allow_short": HyperParam.boolean("allow_short", default=True, grid=[True, False]),
        }

    def __init__(
        self,
        bars: Any,
        events: Any,
        target_symbol: str = "ETH/USDT",
        timeframe: str = "1h",
        lookback_bars: int = 4,
        return_threshold: float = 0.006,
        max_hold_bars: int = 48,
        target_allocation: float = 0.008,
        max_order_value: float = 175.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.0,
        allow_long: bool = True,
        allow_short: bool = True,
    ) -> None:
        self.bars = bars
        self.events = events
        resolved = resolve_params_from_schema(
            self.get_param_schema(),
            {
                "target_symbol": target_symbol,
                "timeframe": timeframe,
                "lookback_bars": lookback_bars,
                "return_threshold": return_threshold,
                "max_hold_bars": max_hold_bars,
                "target_allocation": target_allocation,
                "max_order_value": max_order_value,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "allow_long": allow_long,
                "allow_short": allow_short,
            },
            keep_unknown=False,
        )
        self.target_symbol = canonical_symbol(str(resolved["target_symbol"]))
        self.timeframe = str(resolved["timeframe"] or "1h")
        self.lookback_bars = max(1, int(resolved["lookback_bars"]))
        self.return_threshold = max(0.0, float(resolved["return_threshold"]))
        self.max_hold_bars = max(1, int(resolved["max_hold_bars"]))
        self.target_allocation = max(0.0, float(resolved["target_allocation"]))
        self.max_order_value = max(0.0, float(resolved["max_order_value"]))
        self.stop_loss_pct = max(0.0, float(resolved["stop_loss_pct"]))
        self.take_profit_pct = max(0.0, float(resolved["take_profit_pct"]))
        self.allow_long = bool(resolved["allow_long"])
        self.allow_short = bool(resolved["allow_short"])
        self._state = _PositionState()

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self._state.mode,
            "entry_price": self._state.entry_price,
            "bars_held": int(self._state.bars_held),
            "last_completed_bar_key": self._state.last_completed_bar_key,
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        mode = str(state.get("mode", "OUT")).upper()
        self._state.mode = mode if mode in {"OUT", "LONG", "SHORT"} else "OUT"
        self._state.entry_price = safe_float(state.get("entry_price"))
        try:
            self._state.bars_held = max(0, int(state.get("bars_held", 0)))
        except Exception:
            self._state.bars_held = 0
        self._state.last_completed_bar_key = str(state.get("last_completed_bar_key", ""))

    @staticmethod
    def _completed_bars(aggregator: Any, symbol: str, timeframe: str, lookback: int) -> list[Any]:
        getter = getattr(aggregator, "get_bars", None)
        if not callable(getter):
            return []
        bars = list(
            getter(symbol=str(symbol), timeframe=str(timeframe), n=max(lookback + 1, 2)) or []
        )
        return bars[:-1] if len(bars) >= 2 else []

    @staticmethod
    def _close(bar: Any) -> float | None:
        if isinstance(bar, (tuple, list)) and len(bar) >= 5:
            return safe_float(bar[4])
        if isinstance(bar, dict):
            return safe_float(bar.get("close"))
        return None

    @staticmethod
    def _time_key(bar: Any) -> str:
        if isinstance(bar, (tuple, list)) and bar:
            return str(bar[0])
        if isinstance(bar, dict):
            return str(bar.get("time") or bar.get("datetime") or "")
        return ""

    def _metadata(self, *, shock_return: float, reason: str) -> dict[str, Any]:
        return {
            "strategy": "HourlyShockReversionStrategy",
            "reason": reason,
            "target_symbol": self.target_symbol,
            "timeframe": self.timeframe,
            "lookback_bars": int(self.lookback_bars),
            "shock_return": float(shock_return),
            "return_threshold": float(self.return_threshold),
            "target_allocation": float(self.target_allocation),
            "max_symbol_exposure_pct": float(self.target_allocation),
            "max_order_value": float(self.max_order_value),
        }

    def _emit(
        self,
        *,
        event_time: Any,
        signal_type: str,
        price: float | None,
        metadata: dict[str, Any],
    ) -> None:
        stop_loss = None
        take_profit = None
        if price is not None and price > 0.0:
            if signal_type == "LONG":
                stop_loss = price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
                take_profit = (
                    price * (1.0 + self.take_profit_pct) if self.take_profit_pct > 0.0 else None
                )
            elif signal_type == "SHORT":
                stop_loss = price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0.0 else None
                take_profit = (
                    price * (1.0 - self.take_profit_pct) if self.take_profit_pct > 0.0 else None
                )
        self.events.put(
            SignalEvent(
                strategy_id="hourly_shock_reversion",
                symbol=self.target_symbol,
                datetime=event_time,
                signal_type=signal_type,
                strength=float(self.target_allocation if signal_type in {"LONG", "SHORT"} else 1.0),
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )
        )

    def _maybe_exit(self, *, event_time: Any, price: float, shock_return: float) -> bool:
        if self._state.mode == "OUT":
            return False
        self._state.bars_held += 1
        stop_hit = False
        take_profit_hit = False
        if self._state.entry_price:
            if self._state.mode == "LONG":
                stop_hit = (
                    self.stop_loss_pct > 0.0
                    and price <= self._state.entry_price * (1.0 - self.stop_loss_pct)
                )
                take_profit_hit = (
                    self.take_profit_pct > 0.0
                    and price >= self._state.entry_price * (1.0 + self.take_profit_pct)
                )
            else:
                stop_hit = (
                    self.stop_loss_pct > 0.0
                    and price >= self._state.entry_price * (1.0 + self.stop_loss_pct)
                )
                take_profit_hit = (
                    self.take_profit_pct > 0.0
                    and price <= self._state.entry_price * (1.0 - self.take_profit_pct)
                )
        if not stop_hit and not take_profit_hit and self._state.bars_held < self.max_hold_bars:
            return True
        reason = "max_hold_exit"
        if stop_hit:
            reason = "stop_loss_exit"
        elif take_profit_hit:
            reason = "take_profit_exit"
        metadata = self._metadata(shock_return=shock_return, reason=reason)
        metadata.pop("target_allocation", None)
        metadata.pop("max_symbol_exposure_pct", None)
        metadata.pop("max_order_value", None)
        self._emit(event_time=event_time, signal_type="EXIT", price=price, metadata=metadata)
        self._state.mode = "OUT"
        self._state.entry_price = None
        self._state.bars_held = 0
        return True

    def calculate_signals(self, event: Any) -> None:
        _ = event
        return

    def calculate_signals_window(self, event: Any, aggregator: Any) -> None:
        if aggregator is None:
            return
        bars = self._completed_bars(
            aggregator, self.target_symbol, self.timeframe, self.lookback_bars + 2
        )
        if len(bars) <= self.lookback_bars:
            return
        latest_bar = bars[-1]
        completed_key = self._time_key(latest_bar)
        if not completed_key or completed_key == self._state.last_completed_bar_key:
            return
        self._state.last_completed_bar_key = completed_key

        latest_close = self._close(latest_bar)
        base_close = self._close(bars[-1 - self.lookback_bars])
        if latest_close is None or base_close is None or latest_close <= 0.0 or base_close <= 0.0:
            return

        event_time = (
            latest_bar[0]
            if isinstance(latest_bar, (tuple, list))
            else getattr(event, "time", None)
        )
        shock_return = float(latest_close / base_close - 1.0)
        if self._maybe_exit(event_time=event_time, price=float(latest_close), shock_return=shock_return):
            return
        if self._state.mode != "OUT":
            return

        if shock_return >= self.return_threshold:
            if not self.allow_short:
                return
            signal_type = "SHORT"
            reason = "positive_shock_reversion_short"
        elif shock_return <= -self.return_threshold:
            if not self.allow_long:
                return
            signal_type = "LONG"
            reason = "negative_shock_reversion_long"
        else:
            return

        self._emit(
            event_time=event_time,
            signal_type=signal_type,
            price=float(latest_close),
            metadata=self._metadata(shock_return=shock_return, reason=reason),
        )
        self._state.mode = signal_type
        self._state.entry_price = float(latest_close)
        self._state.bars_held = 0
