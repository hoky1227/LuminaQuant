"""Pair spread z-score strategy wrapper with deterministic default pair anchors."""

from __future__ import annotations

from collections.abc import Sequence

from lumina_quant.strategies.pair_trading_zscore import PairTradingZScoreStrategy
from lumina_quant.symbols import canonical_symbol

_DEFAULT_PAIR_SET: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("BTC/USDT", "TRX/USDT"),
    ("BNB/USDT", "TRX/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT", "XAG/USDT"),
    ("XPT/USDT", "XPD/USDT"),
)

_BOUNDED_PAIR_RETUNE_DEFAULTS: dict[str, float | int] = {
    "lookback_window": 96,
    "hedge_window": 192,
    "min_correlation": 0.20,
    "cooldown_bars": 8,
    "reentry_z_buffer": 0.25,
    "max_hold_bars": 240,
    "stop_loss_pct": 0.03,
}

_BOUNDED_PAIR_RETUNE_BY_TIMEFRAME: dict[str, dict[str, float | int]] = {
    "15m": {
        "lookback_window": 144,
        "hedge_window": 288,
        "min_correlation": 0.25,
        "cooldown_bars": 10,
        "reentry_z_buffer": 0.35,
        "max_hold_bars": 192,
        "stop_loss_pct": 0.025,
    },
    "4h": {
        "lookback_window": 72,
        "hedge_window": 144,
        "min_correlation": 0.05,
        "cooldown_bars": 4,
        "reentry_z_buffer": 0.15,
        "max_hold_bars": 96,
        "stop_loss_pct": 0.025,
    },
    "1d": {
        "lookback_window": 48,
        "hedge_window": 96,
        "min_correlation": 0.0,
        "cooldown_bars": 1,
        "reentry_z_buffer": 0.10,
        "max_hold_bars": 28,
        "stop_loss_pct": 0.020,
    },
}


def bounded_pair_retune_params(timeframe: str) -> dict[str, float | int]:
    """Return bounded turnover/correlation guardrails for pair-spread candidates."""
    payload = dict(_BOUNDED_PAIR_RETUNE_DEFAULTS)
    payload.update(_BOUNDED_PAIR_RETUNE_BY_TIMEFRAME.get(str(timeframe), {}))
    return payload


class PairSpreadZScoreStrategy(PairTradingZScoreStrategy):
    """Alias wrapper around :class:`PairTradingZScoreStrategy` with stable defaults."""

    @staticmethod
    def _resolve_default_pair(symbol_list: Sequence[str]) -> tuple[str, str]:
        canonical = [canonical_symbol(symbol) for symbol in symbol_list if str(symbol).strip()]
        canonical = [symbol for symbol in canonical if symbol]
        universe = set(canonical)

        for left, right in _DEFAULT_PAIR_SET:
            if left in universe and right in universe:
                return left, right

        if len(canonical) >= 2:
            ordered = sorted(dict.fromkeys(canonical))
            return ordered[0], ordered[1]

        raise ValueError("PairSpreadZScoreStrategy requires at least two symbols.")

    def __init__(self, bars, events, *args, symbol_x=None, symbol_y=None, **kwargs):
        if not symbol_x or not symbol_y:
            left, right = self._resolve_default_pair(getattr(bars, "symbol_list", []))
            symbol_x = symbol_x or left
            symbol_y = symbol_y or right
        super().__init__(bars, events, *args, symbol_x=symbol_x, symbol_y=symbol_y, **kwargs)
