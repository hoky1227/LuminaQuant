"""Candidate-library builder for advanced multi-sleeve quant research."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from lumina_quant.strategies.pair_spread_zscore import bounded_pair_retune_params
from lumina_quant.symbols import (
    CANONICAL_STRATEGY_TIMEFRAMES,
    canonicalize_symbol_list,
    normalize_strategy_timeframes,
)

_DEFAULT_SYMBOL_FALLBACK: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TON/USDT",
    "AVAX/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
)

_PAIR_ANCHORS: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("BTC/USDT", "TRX/USDT"),
    ("BNB/USDT", "TRX/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT", "XAG/USDT"),
    ("XPT/USDT", "XPD/USDT"),
    ("BTC/USDT", "XAU/USDT"),
    ("ETH/USDT", "XAU/USDT"),
    ("BNB/USDT", "XAU/USDT"),
    ("BTC/USDT", "XAG/USDT"),
)

_CRYPTO_LEADERS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"}
_METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}


try:  # pragma: no cover - guarded import for bootstrap contexts
    from lumina_quant.config import BaseConfig

    DEFAULT_BINANCE_TOP10_PLUS_METALS: tuple[str, ...] = tuple(
        canonicalize_symbol_list(list(getattr(BaseConfig, "SYMBOLS", _DEFAULT_SYMBOL_FALLBACK)))
    )
except Exception:  # pragma: no cover - safe fallback during partial imports
    DEFAULT_BINANCE_TOP10_PLUS_METALS = _DEFAULT_SYMBOL_FALLBACK

DEFAULT_TIMEFRAMES: tuple[str, ...] = CANONICAL_STRATEGY_TIMEFRAMES

_COMPOSITE_TREND_OOS_STABILITY_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "stable_ls_core",
            "long_threshold": 0.60,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0038,
            "max_signal_strength": 1.35,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.0,
            "max_hold_bars": 960,
            "crowding_reduce_threshold": 0.50,
            "crowding_block_threshold": 0.78,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_highconv",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0036,
            "max_signal_strength": 1.20,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.1,
            "max_hold_bars": 960,
            "crowding_reduce_threshold": 0.50,
            "crowding_block_threshold": 0.78,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_tefilter",
            "long_threshold": 0.60,
            "short_threshold": 0.45,
            "te_min": 0.25,
            "vr_min": 0.80,
            "exit_score_cross": 0.04,
            "chop_max": 58.0,
            "vol_window": 160,
            "risk_target_vol": 0.0035,
            "max_signal_strength": 1.15,
            "atr_stop_mult": 2.1,
            "trail_atr_mult": 3.2,
            "max_hold_bars": 960,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.75,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_crashguard",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.82,
            "exit_score_cross": 0.03,
            "chop_max": 58.0,
            "vol_window": 144,
            "risk_target_vol": 0.0034,
            "max_signal_strength": 1.10,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.2,
            "max_hold_bars": 768,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.72,
            "benchmark_regime_ma": 96,
            "benchmark_symbol": "BTC/USDT",
            "allow_short": True,
        },
        {
            "variant": "stable_ls_exec_trail",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.02,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0035,
            "max_signal_strength": 1.15,
            "atr_stop_mult": 1.8,
            "trail_atr_mult": 2.4,
            "max_hold_bars": 768,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.76,
            "allow_short": True,
        },
        {
            "variant": "stable_ls_exec_shorthold",
            "long_threshold": 0.75,
            "short_threshold": 0.45,
            "te_min": 0.20,
            "vr_min": 0.80,
            "exit_score_cross": 0.04,
            "chop_max": 60.0,
            "vol_window": 144,
            "risk_target_vol": 0.0035,
            "max_signal_strength": 1.10,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 2.8,
            "max_hold_bars": 640,
            "crowding_reduce_threshold": 0.48,
            "crowding_block_threshold": 0.76,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "stable_lo_core",
            "long_threshold": 0.60,
            "short_threshold": 0.75,
            "te_min": 0.25,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 58.0,
            "vol_window": 168,
            "risk_target_vol": 0.0030,
            "max_signal_strength": 1.10,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.0,
            "max_hold_bars": 720,
            "crowding_reduce_threshold": 0.45,
            "crowding_block_threshold": 0.70,
            "allow_short": False,
        },
        {
            "variant": "stable_lo_highconv",
            "long_threshold": 0.75,
            "short_threshold": 0.75,
            "te_min": 0.25,
            "vr_min": 0.80,
            "exit_score_cross": 0.03,
            "chop_max": 58.0,
            "vol_window": 168,
            "risk_target_vol": 0.0028,
            "max_signal_strength": 1.00,
            "atr_stop_mult": 2.0,
            "trail_atr_mult": 3.1,
            "max_hold_bars": 720,
            "crowding_reduce_threshold": 0.45,
            "crowding_block_threshold": 0.70,
            "allow_short": False,
        },
        {
            "variant": "stable_lo_guarded",
            "long_threshold": 0.75,
            "short_threshold": 0.60,
            "te_min": 0.25,
            "vr_min": 0.95,
            "exit_score_cross": 0.02,
            "chop_max": 56.0,
            "vol_window": 192,
            "risk_target_vol": 0.0028,
            "max_signal_strength": 0.95,
            "atr_stop_mult": 2.1,
            "trail_atr_mult": 3.2,
            "max_hold_bars": 720,
            "crowding_reduce_threshold": 0.40,
            "crowding_block_threshold": 0.65,
            "allow_short": False,
        },
    ),
}

_PAIR_RETUNE_FOCUS_PAIRS_15M: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "TRX/USDT"),
    ("BNB/USDT", "TRX/USDT"),
)

_PAIR_RETUNE_FOCUS_PAIRS_4H: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("ETH/USDT", "SOL/USDT"),
    ("XAU/USDT", "XAG/USDT"),
    ("XPT/USDT", "XPD/USDT"),
    ("BTC/USDT", "XAU/USDT"),
    ("ETH/USDT", "XAU/USDT"),
    ("BNB/USDT", "XAU/USDT"),
    ("BTC/USDT", "XAG/USDT"),
)

_PAIR_RETUNE_FOCUS_PAIRS_1D: tuple[tuple[str, str], ...] = (
    ("BTC/USDT", "ETH/USDT"),
    ("BTC/USDT", "BNB/USDT"),
    ("BTC/USDT", "TRX/USDT"),
    ("XPT/USDT", "XPD/USDT"),
    ("BTC/USDT", "XAU/USDT"),
    ("ETH/USDT", "XAU/USDT"),
    ("BNB/USDT", "XAU/USDT"),
    ("BTC/USDT", "XAG/USDT"),
)

_PAIR_RETUNE_SPECS_BY_TIMEFRAME: dict[str, tuple[tuple[float, float, float], ...]] = {
    "15m": (
        (2.6, 0.70, 4.2),
        (3.0, 0.85, 4.8),
    ),
    "1h": (
        (1.8, 0.45, 3.4),
        (2.2, 0.55, 3.9),
        (2.6, 0.70, 4.2),
    ),
    "4h": (
        (1.6, 0.35, 3.0),
        (1.8, 0.45, 3.4),
        (2.0, 0.50, 3.6),
        (2.2, 0.55, 3.9),
        (2.6, 0.70, 4.2),
    ),
    "1d": (
        (1.4, 0.30, 2.8),
        (1.5, 0.33, 2.9),
        (1.6, 0.35, 3.0),
        (1.8, 0.45, 3.4),
        (2.2, 0.55, 3.9),
    ),
}

_PAIR_RETUNE_PARAM_SETS_BY_TIMEFRAME: dict[str, tuple[dict[str, float | int | str], ...]] = {
    "1h": (
        {
            "variant": "core",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.20,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 240,
            "stop_loss_pct": 0.030,
        },
        {
            "variant": "state_vwap",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.25,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.030,
            "vwap_window": 72,
            "min_volume_window": 24,
            "min_volume_ratio": 0.20,
        },
        {
            "variant": "state_volconv",
            "lookback_window": 120,
            "hedge_window": 240,
            "min_correlation": 0.22,
            "cooldown_bars": 10,
            "reentry_z_buffer": 0.30,
            "max_hold_bars": 192,
            "stop_loss_pct": 0.025,
            "vol_lag_bars": 2,
            "min_vol_convergence": 0.60,
            "beta_stop_scale_min": 0.85,
            "beta_stop_scale_max": 2.0,
        },
        {
            "variant": "state_atr",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.25,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.025,
            "atr_window": 14,
            "atr_max_pct": 0.04,
        },
        {
            "variant": "exec_takeprofit",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.20,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.030,
            "take_profit_pct": 0.10,
        },
        {
            "variant": "exec_tightstop_tp",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.20,
            "cooldown_bars": 8,
            "reentry_z_buffer": 0.25,
            "max_hold_bars": 168,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.08,
        },
    ),
    "4h": (
        {
            "variant": "participation",
            "lookback_window": 72,
            "hedge_window": 144,
            "min_correlation": 0.05,
            "cooldown_bars": 4,
            "reentry_z_buffer": 0.15,
            "max_hold_bars": 96,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "balanced",
            "lookback_window": 96,
            "hedge_window": 192,
            "min_correlation": 0.08,
            "cooldown_bars": 5,
            "reentry_z_buffer": 0.18,
            "max_hold_bars": 120,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "stability",
            "lookback_window": 120,
            "hedge_window": 240,
            "min_correlation": 0.12,
            "cooldown_bars": 6,
            "reentry_z_buffer": 0.22,
            "max_hold_bars": 144,
            "stop_loss_pct": 0.020,
        },
        {
            "variant": "fast_cycle",
            "lookback_window": 84,
            "hedge_window": 168,
            "min_correlation": 0.03,
            "cooldown_bars": 3,
            "reentry_z_buffer": 0.12,
            "max_hold_bars": 72,
            "stop_loss_pct": 0.030,
        },
    ),
    "1d": (
        {
            "variant": "participation",
            "lookback_window": 48,
            "hedge_window": 96,
            "min_correlation": 0.00,
            "cooldown_bars": 1,
            "reentry_z_buffer": 0.10,
            "max_hold_bars": 28,
            "stop_loss_pct": 0.020,
        },
        {
            "variant": "balanced",
            "lookback_window": 64,
            "hedge_window": 128,
            "min_correlation": 0.04,
            "cooldown_bars": 2,
            "reentry_z_buffer": 0.12,
            "max_hold_bars": 36,
            "stop_loss_pct": 0.020,
        },
        {
            "variant": "short_window",
            "lookback_window": 40,
            "hedge_window": 80,
            "min_correlation": 0.00,
            "cooldown_bars": 1,
            "reentry_z_buffer": 0.08,
            "max_hold_bars": 24,
            "stop_loss_pct": 0.020,
        },
    ),
}

_LAG_CONVERGENCE_FOCUS_PAIRS_BY_TIMEFRAME: dict[str, tuple[tuple[str, str], ...]] = {
    "4h": (
        ("XAU/USDT", "XAG/USDT"),
        ("XPT/USDT", "XPD/USDT"),
    ),
    "1d": (
        ("XAU/USDT", "XAG/USDT"),
        ("XPT/USDT", "XPD/USDT"),
    ),
}

_LAG_CONVERGENCE_SPECS_BY_TIMEFRAME: dict[str, tuple[dict[str, float | int | str], ...]] = {
    "4h": (
        {
            "variant": "metals_core",
            "lag_bars": 2,
            "entry_threshold": 0.018,
            "exit_threshold": 0.006,
            "stop_threshold": 0.060,
            "max_hold_bars": 36,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "metals_fast",
            "lag_bars": 1,
            "entry_threshold": 0.014,
            "exit_threshold": 0.004,
            "stop_threshold": 0.050,
            "max_hold_bars": 24,
            "stop_loss_pct": 0.030,
        },
    ),
    "1d": (
        {
            "variant": "metals_core",
            "lag_bars": 1,
            "entry_threshold": 0.012,
            "exit_threshold": 0.004,
            "stop_threshold": 0.040,
            "max_hold_bars": 14,
            "stop_loss_pct": 0.025,
        },
        {
            "variant": "metals_patience",
            "lag_bars": 2,
            "entry_threshold": 0.015,
            "exit_threshold": 0.005,
            "stop_threshold": 0.050,
            "max_hold_bars": 18,
            "stop_loss_pct": 0.030,
        },
    ),
}

_ROLLING_BREAKOUT_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "loose_lo",
            "lookback_bars": 48,
            "breakout_buffer": 0.001,
            "atr_window": 14,
            "atr_stop_multiplier": 2.2,
            "stop_loss_pct": 0.025,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "lookback_bars": 64,
            "breakout_buffer": 0.002,
            "atr_window": 21,
            "atr_stop_multiplier": 2.8,
            "stop_loss_pct": 0.030,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "loose_lo",
            "lookback_bars": 36,
            "breakout_buffer": 0.001,
            "atr_window": 14,
            "atr_stop_multiplier": 2.0,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
        {
            "variant": "guarded_ls",
            "lookback_bars": 48,
            "breakout_buffer": 0.002,
            "atr_window": 18,
            "atr_stop_multiplier": 2.5,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
    ),
}

_REGIME_BREAKOUT_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "30m": (
        {
            "variant": "trend_guarded",
            "lookback_window": 48,
            "slope_window": 21,
            "volatility_fast_window": 12,
            "volatility_slow_window": 48,
            "range_entry_threshold": 0.68,
            "slope_entry_threshold": 0.001,
            "momentum_floor": 0.003,
            "max_volatility_ratio": 1.8,
            "stop_loss_pct": 0.025,
            "allow_short": False,
        },
        {
            "variant": "trend_ls",
            "lookback_window": 64,
            "slope_window": 24,
            "volatility_fast_window": 16,
            "volatility_slow_window": 64,
            "range_entry_threshold": 0.72,
            "slope_entry_threshold": 0.0015,
            "momentum_floor": 0.004,
            "max_volatility_ratio": 1.7,
            "stop_loss_pct": 0.030,
            "allow_short": True,
        },
    ),
    "1h": (
        {
            "variant": "trend_guarded",
            "lookback_window": 36,
            "slope_window": 18,
            "volatility_fast_window": 10,
            "volatility_slow_window": 40,
            "range_entry_threshold": 0.65,
            "slope_entry_threshold": 0.0008,
            "momentum_floor": 0.002,
            "max_volatility_ratio": 1.9,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
        {
            "variant": "trend_ls",
            "lookback_window": 48,
            "slope_window": 21,
            "volatility_fast_window": 12,
            "volatility_slow_window": 48,
            "range_entry_threshold": 0.70,
            "slope_entry_threshold": 0.001,
            "momentum_floor": 0.003,
            "max_volatility_ratio": 1.8,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
    ),
}

_MEAN_REVERSION_STD_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "15m": (
        {
            "variant": "balanced_ls",
            "window": 64,
            "entry_z": 2.0,
            "exit_z": 0.50,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 96,
            "entry_z": 2.4,
            "exit_z": 0.40,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
    ),
    "30m": (
        {
            "variant": "balanced_ls",
            "window": 48,
            "entry_z": 1.8,
            "exit_z": 0.45,
            "stop_loss_pct": 0.025,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 72,
            "entry_z": 2.2,
            "exit_z": 0.35,
            "stop_loss_pct": 0.020,
            "allow_short": False,
        },
    ),
}

_VWAP_REVERSION_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "balanced_ls",
            "window": 48,
            "entry_dev": 0.012,
            "exit_dev": 0.003,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 64,
            "entry_dev": 0.016,
            "exit_dev": 0.004,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
    "15m": (
        {
            "variant": "balanced_ls",
            "window": 36,
            "entry_dev": 0.010,
            "exit_dev": 0.002,
            "stop_loss_pct": 0.020,
            "allow_short": True,
        },
        {
            "variant": "guarded_lo",
            "window": 48,
            "entry_dev": 0.014,
            "exit_dev": 0.003,
            "stop_loss_pct": 0.018,
            "allow_short": False,
        },
    ),
}

_TOPCAP_TSMOM_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "balanced",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "resid_btc",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.010,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
            "residualize_btc": True,
            "residualize_mean": False,
        },
        {
            "variant": "resid_beta_neutral",
            "lookback_bars": 24,
            "rebalance_bars": 4,
            "signal_threshold": 0.008,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
            "residualize_btc": True,
            "residualize_mean": True,
        },
        {
            "variant": "defensive",
            "lookback_bars": 24,
            "rebalance_bars": 6,
            "signal_threshold": 0.020,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 1,
            "min_price": 0.10,
            "btc_regime_ma": 64,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "crashguard",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 48,
            "benchmark_drawdown_limit": 0.08,
        },
        {
            "variant": "exec_tightstop",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.05,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_fastrebalance",
            "lookback_bars": 16,
            "rebalance_bars": 2,
            "signal_threshold": 0.012,
            "stop_loss_pct": 0.07,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_takeprofit",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.10,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_tightstop_tp",
            "lookback_bars": 16,
            "rebalance_bars": 4,
            "signal_threshold": 0.015,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
        {
            "variant": "exec_fastrebalance_tp",
            "lookback_bars": 16,
            "rebalance_bars": 2,
            "signal_threshold": 0.012,
            "stop_loss_pct": 0.07,
            "take_profit_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 48,
            "btc_symbol": "BTC/USDT",
        },
    ),
    "4h": (
        {
            "variant": "balanced",
            "lookback_bars": 10,
            "rebalance_bars": 2,
            "signal_threshold": 0.020,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.10,
            "btc_regime_ma": 18,
            "btc_symbol": "BTC/USDT",
        },
    ),
}

_ALPHA101_SIGNAL_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "1h": (
        {
            "variant": "a005_vwap_tuned",
            "alpha_id": 5,
            "rank_window": 24,
            "history_window": 144,
            "score_window": 64,
            "entry_z": 1.15,
            "exit_z": 0.30,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": True,
            "alpha_param_overrides": {
                "alpha101.5.const.001": 8.0,
                "alpha101.5.const.002": 14.0,
            },
        },
        {
            "variant": "a011_flow_tuned",
            "alpha_id": 11,
            "rank_window": 24,
            "history_window": 144,
            "score_window": 64,
            "entry_z": 1.10,
            "exit_z": 0.30,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": True,
            "alpha_param_overrides": {
                "alpha101.11.const.001": 4.0,
                "alpha101.11.const.002": 4.0,
                "alpha101.11.const.003": 5.0,
            },
        },
        {
            "variant": "a017_turn_tuned",
            "alpha_id": 17,
            "rank_window": 20,
            "history_window": 160,
            "score_window": 64,
            "entry_z": 1.00,
            "exit_z": 0.25,
            "signal_sign": -1.0,
            "stop_loss_pct": 0.03,
            "allow_short": False,
            "alpha_param_overrides": {
                "alpha101.17.const.001": 12.0,
                "alpha101.17.const.002": 7.0,
            },
        },
        {
            "variant": "a101_bodyrange_tuned",
            "alpha_id": 101,
            "rank_window": 20,
            "history_window": 96,
            "score_window": 48,
            "entry_z": 1.00,
            "exit_z": 0.25,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": False,
            "alpha_param_overrides": {
                "alpha101.101.const.001": 0.01,
            },
        },
    ),
    "4h": (
        {
            "variant": "a011_flow_swing",
            "alpha_id": 11,
            "rank_window": 20,
            "history_window": 96,
            "score_window": 32,
            "entry_z": 1.05,
            "exit_z": 0.25,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.035,
            "allow_short": True,
            "alpha_param_overrides": {
                "alpha101.11.const.001": 5.0,
                "alpha101.11.const.002": 5.0,
                "alpha101.11.const.003": 4.0,
            },
        },
        {
            "variant": "a101_bodyrange_swing",
            "alpha_id": 101,
            "rank_window": 16,
            "history_window": 80,
            "score_window": 24,
            "entry_z": 0.90,
            "exit_z": 0.20,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.035,
            "allow_short": False,
            "alpha_param_overrides": {
                "alpha101.101.const.001": 0.02,
            },
        },
    ),
}

_VOLCOMP_RETUNE_SLICE: dict[str, tuple[dict[str, Any], ...]] = {
    "5m": (
        {
            "variant": "guarded_lo_core",
            "vwap_window": 96,
            "z_window": 192,
            "entry_z": 2.2,
            "exit_z": 0.18,
            "compression_percentile": 0.12,
            "compression_vol_ratio": 0.72,
            "atr_stop_pct": 0.012,
            "max_hold_bars": 24,
            "allow_short": False,
        },
        {
            "variant": "guarded_lo_strict",
            "vwap_window": 120,
            "z_window": 240,
            "entry_z": 2.6,
            "exit_z": 0.15,
            "compression_percentile": 0.10,
            "compression_vol_ratio": 0.68,
            "atr_stop_pct": 0.010,
            "max_hold_bars": 18,
            "allow_short": False,
        },
    ),
    "15m": (
        {
            "variant": "guarded_lo_core",
            "vwap_window": 72,
            "z_window": 168,
            "entry_z": 2.0,
            "exit_z": 0.22,
            "compression_percentile": 0.16,
            "compression_vol_ratio": 0.78,
            "atr_stop_pct": 0.016,
            "max_hold_bars": 36,
            "allow_short": False,
        },
        {
            "variant": "guarded_lo_strict",
            "vwap_window": 96,
            "z_window": 192,
            "entry_z": 2.4,
            "exit_z": 0.18,
            "compression_percentile": 0.12,
            "compression_vol_ratio": 0.72,
            "atr_stop_pct": 0.014,
            "max_hold_bars": 28,
            "allow_short": False,
        },
    ),
}


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """Serializable strategy-candidate definition."""

    candidate_id: str
    name: str
    family: str
    strategy_class: str
    timeframe: str
    symbols: tuple[str, ...]
    params: dict[str, Any]
    notes: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        timeframe = str(self.timeframe)
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "family": self.family,
            "strategy_class": self.strategy_class,
            "strategy": self.strategy_class,
            "strategy_timeframe": timeframe,
            # Legacy alias retained for compatibility.
            "timeframe": timeframe,
            "symbols": list(self.symbols),
            "params": dict(self.params),
            "notes": self.notes,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


def _normalize_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(canonicalize_symbol_list(values))


def _candidate_id(*, name: str, timeframe: str, params: dict[str, Any], symbols: tuple[str, ...]) -> str:
    payload = {
        "name": name,
        "timeframe": str(timeframe),
        "params": params,
        "symbols": list(symbols),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _article_pipeline_family_ids(
    *,
    strategy_class: str,
    timeframe: str,
    symbols: Sequence[str],
) -> tuple[str, ...]:
    symbol_set = set(canonicalize_symbol_list(symbols))
    strategy_token = str(strategy_class or "").strip()
    timeframe_token = str(timeframe or "").strip()
    if strategy_token == "CompositeTrendStrategy":
        return ("regime-conditioned-composite-trend",)
    if strategy_token == "VolCompressionVWAPReversionStrategy":
        return ("vol-compression-break-reversion",)
    if strategy_token == "LeadLagSpilloverStrategy":
        return ("lead-lag-regime-spillover",)
    if strategy_token == "LagConvergenceStrategy":
        return ("metals-lag-convergence",) if symbol_set.intersection(_METALS) else ()
    if strategy_token == "TopCapTimeSeriesMomentumStrategy":
        return ("topcap-rotation-relative-momentum",)
    if strategy_token == "Alpha101FormulaStrategy":
        return ("formulaic-alpha101-research",)
    if strategy_token in {"RollingBreakoutStrategy", "RegimeBreakoutCandidateStrategy"}:
        return ("regime-breakout-thrust",)
    if strategy_token == "MeanReversionStdStrategy":
        return ("single-asset-zscore-reversion",)
    if strategy_token == "VwapReversionStrategy":
        return ("intraday-vwap-reversion",)
    if strategy_token == "PairSpreadZScoreStrategy":
        if symbol_set.intersection(_METALS):
            return ("crypto-metal-residual-pairs",)
        if timeframe_token in {"15m", "1h"} and symbol_set and symbol_set.isdisjoint(_METALS):
            return ("sector-dispersion-reversion",)
    return ()


def _with_article_pipeline_provenance(
    *,
    strategy_class: str,
    timeframe: str,
    symbols: Sequence[str],
    tags: Sequence[str] | None,
    metadata: dict[str, Any] | None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    merged_metadata = dict(metadata or {})
    family_ids = list(
        dict.fromkeys(
            [
                str(item).strip()
                for item in list(merged_metadata.get("article_pipeline_family_ids") or [])
                if str(item).strip()
            ]
            + list(
                _article_pipeline_family_ids(
                    strategy_class=strategy_class,
                    timeframe=timeframe,
                    symbols=symbols,
                )
            )
        )
    )
    merged_tags = [
        str(tag).strip()
        for tag in list(tags or [])
        if str(tag).strip()
    ]
    if family_ids:
        merged_tags.extend(["article_pipeline", *[f"article_family:{item}" for item in family_ids]])
        merged_metadata["article_pipeline_family_ids"] = list(family_ids)
        merged_metadata["hypothesis_origin"] = "article_research_pipeline"
    return tuple(dict.fromkeys(merged_tags)), merged_metadata


def _has_perp_support_data() -> bool:
    path = Path("data") / "market_parquet" / "feature_points"
    return path.exists()


def _add_candidate(
    out: list[StrategyCandidate],
    *,
    name: str,
    family: str,
    strategy_class: str,
    timeframe: str,
    symbols: Sequence[str],
    params: dict[str, Any],
    notes: str,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    symbol_tuple = tuple(canonicalize_symbol_list(symbols))
    if not symbol_tuple:
        return
    normalized_tags, enriched_metadata = _with_article_pipeline_provenance(
        strategy_class=strategy_class,
        timeframe=str(timeframe),
        symbols=symbol_tuple,
        tags=tags,
        metadata=metadata,
    )
    metadata_payload = {
        "timeframe": str(timeframe),
        "family": str(family),
        **enriched_metadata,
    }
    out.append(
        StrategyCandidate(
            candidate_id=_candidate_id(
                name=name,
                timeframe=timeframe,
                params=params,
                symbols=symbol_tuple,
            ),
            name=name,
            family=family,
            strategy_class=strategy_class,
            timeframe=str(timeframe),
            symbols=symbol_tuple,
            params=dict(params),
            notes=notes,
            tags=normalized_tags,
            metadata=metadata_payload,
        )
    )


def _pairs_in_universe(symbols: Sequence[str]) -> list[tuple[str, str]]:
    universe = set(symbols)
    out: list[tuple[str, str]] = []
    for left, right in _PAIR_ANCHORS:
        if left in universe and right in universe:
            out.append((left, right))
    return out


def _build_alpha101_formula_params(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "alpha_id": int(spec["alpha_id"]),
        "rank_window": int(spec["rank_window"]),
        "history_window": int(spec["history_window"]),
        "score_window": int(spec["score_window"]),
        "entry_z": float(spec["entry_z"]),
        "exit_z": float(spec["exit_z"]),
        "signal_sign": float(spec["signal_sign"]),
        "stop_loss_pct": float(spec["stop_loss_pct"]),
        "allow_short": bool(spec["allow_short"]),
        "alpha_param_overrides": dict(spec["alpha_param_overrides"]),
    }


def _add_alpha101_formula_candidates(
    out: list[StrategyCandidate],
    *,
    timeframes: Sequence[str],
    symbols: Sequence[str],
) -> None:
    for timeframe in timeframes:
        tf_tag = timeframe.replace("/", "-")
        for spec in _ALPHA101_SIGNAL_SLICE.get(timeframe, ()):
            signal_sign = float(spec["signal_sign"])
            alpha_param_overrides = dict(spec["alpha_param_overrides"])
            direction_tag = "inv" if signal_sign < 0.0 else "dir"
            _add_candidate(
                out,
                name=(
                    f"alpha101_formula_{tf_tag}_a{int(spec['alpha_id']):03d}_{spec['variant']}_{direction_tag}"
                ),
                family="formulaic_alpha",
                strategy_class="Alpha101FormulaStrategy",
                timeframe=timeframe,
                symbols=symbols,
                params=_build_alpha101_formula_params(spec),
                notes=(
                    "Single-asset Alpha101 factor sleeve with explicit constant overrides "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("alpha101", "formulaic", "single_asset", "factor"),
                metadata={
                    "timeframe": timeframe,
                    "alpha_id": int(spec["alpha_id"]),
                    "signal_sign": signal_sign,
                    "allow_short": bool(spec["allow_short"]),
                    "alpha_param_override_keys": sorted(alpha_param_overrides.keys()),
                    "retune_profile": str(spec["variant"]),
                },
            )


def build_binance_futures_candidates(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> list[StrategyCandidate]:
    """Build candidate universe for RG_PVTM and diversifier sleeves."""
    normalized_timeframes = tuple(
        normalize_strategy_timeframes(
            list(timeframes),
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    normalized_symbols = _normalize_unique(symbols)
    if not normalized_timeframes:
        raise ValueError("timeframes must not be empty")
    if len(normalized_symbols) < 2:
        raise ValueError("symbols must include at least two instruments")

    candidates: list[StrategyCandidate] = []
    pairs = _pairs_in_universe(normalized_symbols)

    trend_tfs = [tf for tf in ("30m", "1h") if tf in normalized_timeframes]
    mean_rev_tfs = [tf for tf in ("5m", "15m") if tf in normalized_timeframes]
    std_mean_rev_tfs = [tf for tf in ("15m", "30m") if tf in normalized_timeframes]
    breakout_tfs = [tf for tf in ("30m", "1h") if tf in normalized_timeframes]
    topcap_tfs = [tf for tf in ("1h", "4h") if tf in normalized_timeframes]
    alpha101_tfs = [tf for tf in ("1h", "4h") if tf in normalized_timeframes]
    pair_tfs = [tf for tf in ("15m", "1h", "4h", "1d") if tf in normalized_timeframes]
    lag_convergence_tfs = [tf for tf in ("4h", "1d") if tf in normalized_timeframes]
    carry_tfs = [tf for tf in ("30m", "1h", "4h") if tf in normalized_timeframes]
    micro_tfs = [tf for tf in ("1s",) if tf in normalized_timeframes]

    crypto_symbols = [symbol for symbol in normalized_symbols if symbol not in _METALS]
    laggard_symbols = [symbol for symbol in crypto_symbols if symbol not in _CRYPTO_LEADERS]

    # Primary trend sleeve (RG_PVTM) with explicit 30m/1h OOS-stability retune only.
    for timeframe in trend_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _COMPOSITE_TREND_OOS_STABILITY_SLICE.get(timeframe, ()):
            params = {
                "long_threshold": float(spec["long_threshold"]),
                "short_threshold": float(spec["short_threshold"]),
                "exit_score_cross": float(spec["exit_score_cross"]),
                "te_min": float(spec["te_min"]),
                "vr_min": float(spec["vr_min"]),
                "chop_max": float(spec["chop_max"]),
                "vol_window": int(spec["vol_window"]),
                "risk_target_vol": float(spec["risk_target_vol"]),
                "max_signal_strength": float(spec["max_signal_strength"]),
                "atr_stop_mult": float(spec["atr_stop_mult"]),
                "trail_atr_mult": float(spec["trail_atr_mult"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "crowding_reduce_threshold": float(spec["crowding_reduce_threshold"]),
                "crowding_block_threshold": float(spec["crowding_block_threshold"]),
                "allow_short": bool(spec["allow_short"]),
            }
            if "benchmark_regime_ma" in spec:
                params["benchmark_regime_ma"] = int(spec["benchmark_regime_ma"])
            if "benchmark_symbol" in spec:
                params["benchmark_symbol"] = str(spec["benchmark_symbol"])
            regime_tag = "ls" if bool(spec["allow_short"]) else "lo"
            tags = ["trend", "trend-following", "momentum", "oos-stability"]
            note_suffix = ""
            if int(spec.get("benchmark_regime_ma", 0) or 0) > 0:
                tags.append("crash_aware")
                note_suffix = (
                    f" Crash-aware long gate uses {spec.get('benchmark_symbol', 'BTC/USDT')} "
                    f"vs {int(spec['benchmark_regime_ma'])}-bar MA."
                )
            if "exec_" in str(spec.get("variant") or ""):
                tags.append("execution_risk")
                note_suffix = f"{note_suffix} Execution-risk retune." if note_suffix else " Execution-risk retune."
            _add_candidate(
                candidates,
                name=(
                    "composite_trend_stable_"
                    f"{tf_tag}_{spec['variant']}_{regime_tag}_"
                    f"{float(spec['long_threshold']):.2f}_{float(spec['short_threshold']):.2f}_"
                    f"{float(spec['te_min']):.2f}_{float(spec['vr_min']):.2f}"
                ),
                family="trend",
                strategy_class="CompositeTrendStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Primary RG_PVTM trend sleeve with bounded 30m/1h OOS-stability retune "
                    f"({spec['variant']}, {'long-only' if not bool(spec['allow_short']) else 'long/short'})."
                    f"{note_suffix}"
                ),
                tags=tuple(tags),
                metadata={
                    "timeframe": timeframe,
                    "regime": "ls" if bool(spec["allow_short"]) else "lo",
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                    "benchmark_regime_ma": int(spec.get("benchmark_regime_ma", 0) or 0),
                    "benchmark_symbol": str(spec.get("benchmark_symbol") or ""),
                },
            )

    # Vol-compression VWAP reversion sleeve.
    for timeframe in mean_rev_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _VOLCOMP_RETUNE_SLICE.get(timeframe, ()):
            params = {
                "vwap_window": int(spec["vwap_window"]),
                "z_window": int(spec["z_window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "compression_percentile": float(spec["compression_percentile"]),
                "compression_vol_ratio": float(spec["compression_vol_ratio"]),
                "atr_stop_pct": float(spec["atr_stop_pct"]),
                "max_hold_bars": int(spec["max_hold_bars"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"volcomp_vwap_rev_guarded_{tf_tag}_{spec['variant']}_"
                    f"{float(spec['entry_z']):.2f}_{float(spec['compression_percentile']):.2f}"
                ),
                family="mean_reversion",
                strategy_class="VolCompressionVWAPReversionStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Compression-gated VWAP mean reversion with bounded low-turnover guardrails "
                    f"for {timeframe} follow-up ({spec['variant']})."
                ),
                tags=("mean_reversion", "vol_compression", "vwap", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "entry_guard": "zscore",
                    "allow_short": bool(spec["allow_short"]),
                },
            )

    # Classic VWAP deviation reversion sleeve.
    for timeframe in mean_rev_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _VWAP_REVERSION_SLICE.get(timeframe, ()):
            params = {
                "window": int(spec["window"]),
                "entry_dev": float(spec["entry_dev"]),
                "exit_dev": float(spec["exit_dev"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"vwap_reversion_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['window'])}_{float(spec['entry_dev']):.3f}"
                ),
                family="mean_reversion",
                strategy_class="VwapReversionStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Rolling VWAP deviation mean reversion with bounded entry/exit bands "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "vwap", "single_asset", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                },
            )

    # Classic z-score mean reversion sleeve.
    for timeframe in std_mean_rev_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _MEAN_REVERSION_STD_SLICE.get(timeframe, ()):
            params = {
                "window": int(spec["window"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"mean_reversion_std_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['window'])}_{float(spec['entry_z']):.2f}"
                ),
                family="mean_reversion",
                strategy_class="MeanReversionStdStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Single-asset rolling z-score mean reversion with bounded stop rules "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("mean_reversion", "zscore", "single_asset", "bounded"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                },
            )

    # Lead/lag spillover sleeve (metals excluded).
    if laggard_symbols:
        for timeframe in mean_rev_tfs:
            tf_tag = timeframe.replace("/", "-")
            for entry_score, max_lag in product((0.25, 0.35, 0.50), (2, 3, 4)):
                params = {
                    "entry_score": float(entry_score),
                    "exit_score": 0.08,
                    "max_lag": int(max_lag),
                    "ridge_alpha": 1.0,
                    "max_hold_bars": 24,
                    "stop_loss_pct": 0.02,
                    "allow_short": True,
                }
                _add_candidate(
                    candidates,
                    name=f"leadlag_spillover_{tf_tag}_{entry_score:.2f}_lag{max_lag}",
                    family="intraday_alpha",
                    strategy_class="LeadLagSpilloverStrategy",
                    timeframe=timeframe,
                    symbols=tuple(sorted(set(_CRYPTO_LEADERS).intersection(normalized_symbols)) + laggard_symbols),
                    params=params,
                    notes="Cross-asset lead-lag predictor (crypto only, metals excluded).",
                    tags=("leadlag", "cross-asset", "intraday", "alpha"),
                    metadata={
                        "timeframe": timeframe,
                        "symbol_scope": "crypto_excluding_metals",
                        "lag_bands": [2, 3, 4],
                        },
                    )

    if len(crypto_symbols) >= 4:
        for timeframe in topcap_tfs:
            tf_tag = timeframe.replace("/", "-")
            for spec in _TOPCAP_TSMOM_SLICE.get(timeframe, ()):
                params = {
                    "lookback_bars": int(spec["lookback_bars"]),
                    "rebalance_bars": int(spec["rebalance_bars"]),
                    "signal_threshold": float(spec["signal_threshold"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                    "max_longs": int(spec["max_longs"]),
                    "max_shorts": int(spec["max_shorts"]),
                    "min_price": float(spec["min_price"]),
                    "btc_regime_ma": int(spec["btc_regime_ma"]),
                    "btc_symbol": str(spec["btc_symbol"]),
                }
                if "take_profit_pct" in spec:
                    params["take_profit_pct"] = float(spec["take_profit_pct"])
                if "residualize_btc" in spec:
                    params["residualize_btc"] = bool(spec["residualize_btc"])
                if "residualize_mean" in spec:
                    params["residualize_mean"] = bool(spec["residualize_mean"])
                if "benchmark_drawdown_window" in spec:
                    params["benchmark_drawdown_window"] = int(spec["benchmark_drawdown_window"])
                if "benchmark_drawdown_limit" in spec:
                    params["benchmark_drawdown_limit"] = float(spec["benchmark_drawdown_limit"])
                tags = ["cross_sectional", "relative_momentum", "topcap", "crypto"]
                residual_notes = []
                if bool(spec.get("residualize_btc", False)):
                    tags.append("residual_momentum")
                    residual_notes.append("BTC-common-move residualization")
                if bool(spec.get("residualize_mean", False)):
                    tags.append("factor_neutral")
                    residual_notes.append("cross-sectional mean neutralization")
                if int(spec.get("benchmark_drawdown_window", 0) or 0) > 0 and float(spec.get("benchmark_drawdown_limit", 0.0) or 0.0) > 0.0:
                    tags.append("crash_aware")
                    residual_notes.append(
                        f"benchmark drawdown gate {int(spec['benchmark_drawdown_window'])} bars/{float(spec['benchmark_drawdown_limit']):.1%}"
                    )
                if str(spec.get("variant") or "").startswith("exec_"):
                    tags.append("execution_risk")
                    residual_notes.append("execution-risk retune")
                if float(spec.get("take_profit_pct", 0.0) or 0.0) > 0.0:
                    tags.append("take_profit")
                    residual_notes.append(f"take profit {float(spec['take_profit_pct']):.1%}")
                note_suffix = (
                    " with " + " + ".join(residual_notes) + "."
                    if residual_notes
                    else "."
                )
                _add_candidate(
                    candidates,
                    name=(
                        f"topcap_tsmom_{tf_tag}_{spec['variant']}_"
                        f"{int(spec['lookback_bars'])}_{int(spec['rebalance_bars'])}_{float(spec['signal_threshold']):.3f}"
                    ),
                    family="cross_sectional",
                    strategy_class="TopCapTimeSeriesMomentumStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes=(
                        "Top-cap long/short relative-momentum rotation with BTC regime gating "
                        f"for {timeframe} ({spec['variant']}){note_suffix}"
                    ),
                    tags=tuple(tags),
                    metadata={
                        "timeframe": timeframe,
                        "retune_profile": str(spec["variant"]),
                        "symbol_scope": "crypto",
                        "residualize_btc": bool(spec.get("residualize_btc", False)),
                        "residualize_mean": bool(spec.get("residualize_mean", False)),
                        "benchmark_drawdown_window": int(spec.get("benchmark_drawdown_window", 0) or 0),
                        "benchmark_drawdown_limit": float(spec.get("benchmark_drawdown_limit", 0.0) or 0.0),
                    },
                )

    if crypto_symbols:
        _add_alpha101_formula_candidates(
            candidates,
            timeframes=alpha101_tfs,
            symbols=crypto_symbols,
        )

    # Single-asset breakout sleeves.
    for timeframe in breakout_tfs:
        tf_tag = timeframe.replace("/", "-")
        for spec in _ROLLING_BREAKOUT_SLICE.get(timeframe, ()):
            params = {
                "lookback_bars": int(spec["lookback_bars"]),
                "breakout_buffer": float(spec["breakout_buffer"]),
                "atr_window": int(spec["atr_window"]),
                "atr_stop_multiplier": float(spec["atr_stop_multiplier"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"rolling_breakout_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['lookback_bars'])}_{float(spec['breakout_buffer']):.3f}"
                ),
                family="trend",
                strategy_class="RollingBreakoutStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Single-asset channel breakout with ATR-aware protective stops "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("trend", "breakout", "single_asset", "atr"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                },
            )
        for spec in _REGIME_BREAKOUT_SLICE.get(timeframe, ()):
            params = {
                "lookback_window": int(spec["lookback_window"]),
                "slope_window": int(spec["slope_window"]),
                "volatility_fast_window": int(spec["volatility_fast_window"]),
                "volatility_slow_window": int(spec["volatility_slow_window"]),
                "range_entry_threshold": float(spec["range_entry_threshold"]),
                "slope_entry_threshold": float(spec["slope_entry_threshold"]),
                "momentum_floor": float(spec["momentum_floor"]),
                "max_volatility_ratio": float(spec["max_volatility_ratio"]),
                "stop_loss_pct": float(spec["stop_loss_pct"]),
                "allow_short": bool(spec["allow_short"]),
            }
            _add_candidate(
                candidates,
                name=(
                    f"regime_breakout_{tf_tag}_{spec['variant']}_"
                    f"{int(spec['lookback_window'])}_{float(spec['range_entry_threshold']):.2f}"
                ),
                family="trend",
                strategy_class="RegimeBreakoutCandidateStrategy",
                timeframe=timeframe,
                symbols=normalized_symbols,
                params=params,
                notes=(
                    "Regime-gated breakout candidate with trend and volatility filters "
                    f"for {timeframe} ({spec['variant']})."
                ),
                tags=("trend", "breakout", "regime", "single_asset"),
                metadata={
                    "timeframe": timeframe,
                    "allow_short": bool(spec["allow_short"]),
                    "retune_profile": str(spec["variant"]),
                },
            )

    # Pair spread sleeve.
    for timeframe in pair_tfs:
        tf_tag = timeframe.replace("/", "-")
        tuned_params = bounded_pair_retune_params(timeframe)
        tuned_param_sets = tuple(
            dict(item)
            for item in _PAIR_RETUNE_PARAM_SETS_BY_TIMEFRAME.get(timeframe, (dict(tuned_params),))
        )
        pair_universe = list(pairs)
        if timeframe == "15m":
            pair_universe = [pair for pair in pair_universe if pair in _PAIR_RETUNE_FOCUS_PAIRS_15M]
        elif timeframe == "4h":
            pair_universe = [pair for pair in pair_universe if pair in _PAIR_RETUNE_FOCUS_PAIRS_4H]
        elif timeframe == "1d":
            pair_universe = [pair for pair in pair_universe if pair in _PAIR_RETUNE_FOCUS_PAIRS_1D]
        for symbol_x, symbol_y in pair_universe:
            pair_token = f"{symbol_x.replace('/', '').lower()}_{symbol_y.replace('/', '').lower()}"
            for tuned_spec in tuned_param_sets:
                variant = str(tuned_spec.get("variant") or "core")
                for entry_z, exit_z, stop_z in _PAIR_RETUNE_SPECS_BY_TIMEFRAME.get(timeframe, ()):
                    params = {
                        "lookback_window": int(tuned_spec["lookback_window"]),
                        "hedge_window": int(tuned_spec["hedge_window"]),
                        "entry_z": float(entry_z),
                        "exit_z": float(exit_z),
                        "stop_z": float(stop_z),
                        "max_hold_bars": int(tuned_spec["max_hold_bars"]),
                        "min_correlation": float(tuned_spec["min_correlation"]),
                        "cooldown_bars": int(tuned_spec["cooldown_bars"]),
                        "reentry_z_buffer": float(tuned_spec["reentry_z_buffer"]),
                        "stop_loss_pct": float(tuned_spec["stop_loss_pct"]),
                        "symbol_x": symbol_x,
                        "symbol_y": symbol_y,
                    }
                    for optional_key in (
                        "vwap_window",
                        "min_volume_window",
                        "min_volume_ratio",
                        "vol_lag_bars",
                        "min_vol_convergence",
                        "atr_window",
                        "atr_max_pct",
                        "beta_stop_scale_min",
                        "beta_stop_scale_max",
                        "take_profit_pct",
                    ):
                        if optional_key in tuned_spec:
                            params[optional_key] = tuned_spec[optional_key]
                    tags = ["market_neutral", "pair", "spread", "zscore"]
                    state_notes = []
                    if int(tuned_spec.get("vwap_window", 0) or 0) > 0:
                        tags.append("pair_state")
                        state_notes.append(f"VWAP normalization {int(tuned_spec['vwap_window'])}")
                    if float(tuned_spec.get("min_volume_ratio", 0.0) or 0.0) > 0.0:
                        tags.append("pair_state")
                        state_notes.append(f"volume ratio >= {float(tuned_spec['min_volume_ratio']):.2f}")
                    if float(tuned_spec.get("min_vol_convergence", 0.0) or 0.0) > 0.0:
                        tags.append("pair_state")
                        state_notes.append(
                            f"vol convergence z >= {float(tuned_spec['min_vol_convergence']):.2f}"
                        )
                    if int(tuned_spec.get("atr_window", 0) or 0) > 0:
                        tags.append("pair_state")
                        state_notes.append(
                            f"ATR filter {int(tuned_spec['atr_window'])}/{float(tuned_spec.get('atr_max_pct', 1.0)):.2f}"
                        )
                    if float(tuned_spec.get("take_profit_pct", 0.0) or 0.0) > 0.0:
                        tags.append("execution_risk")
                        tags.append("take_profit")
                        state_notes.append(f"take profit {float(tuned_spec['take_profit_pct']):.1%}")
                    note_suffix = (
                        " " + "; ".join(state_notes) + "."
                        if state_notes
                        else ""
                    )
                    _add_candidate(
                        candidates,
                        name=f"pair_spread_{tf_tag}_{variant}_{pair_token}_{entry_z:.1f}_{exit_z:.2f}",
                        family="market_neutral",
                        strategy_class="PairSpreadZScoreStrategy",
                        timeframe=timeframe,
                        symbols=(symbol_x, symbol_y),
                        params=params,
                        notes=(
                            "Rolling-beta spread z-score with bounded turnover/correlation guardrails"
                            + (" and 15m evidence-focused pair pruning." if timeframe == "15m" else "")
                            + (
                                f" {timeframe} uses {variant} tuning to balance participation, stability, and PBO."
                                if timeframe in {"4h", "1d"}
                                else "."
                            )
                            + note_suffix
                        ),
                        tags=tuple(dict.fromkeys(tags)),
                        metadata={
                            "timeframe": timeframe,
                            "pair": f"{symbol_x}_{symbol_y}",
                            "pair_variant": variant,
                        },
                    )

    for timeframe in lag_convergence_tfs:
        tf_tag = timeframe.replace("/", "-")
        pair_universe = [
            pair
            for pair in _LAG_CONVERGENCE_FOCUS_PAIRS_BY_TIMEFRAME.get(timeframe, ())
            if pair in pairs
        ]
        for symbol_x, symbol_y in pair_universe:
            pair_token = f"{symbol_x.replace('/', '').lower()}_{symbol_y.replace('/', '').lower()}"
            for spec in _LAG_CONVERGENCE_SPECS_BY_TIMEFRAME.get(timeframe, ()):
                params = {
                    "symbol_x": symbol_x,
                    "symbol_y": symbol_y,
                    "lag_bars": int(spec["lag_bars"]),
                    "entry_threshold": float(spec["entry_threshold"]),
                    "exit_threshold": float(spec["exit_threshold"]),
                    "stop_threshold": float(spec["stop_threshold"]),
                    "max_hold_bars": int(spec["max_hold_bars"]),
                    "stop_loss_pct": float(spec["stop_loss_pct"]),
                }
                _add_candidate(
                    candidates,
                    name=(
                        f"lag_convergence_{tf_tag}_{spec['variant']}_{pair_token}_"
                        f"{int(spec['lag_bars'])}_{float(spec['entry_threshold']):.3f}"
                    ),
                    family="intermarket",
                    strategy_class="LagConvergenceStrategy",
                    timeframe=timeframe,
                    symbols=(symbol_x, symbol_y),
                    params=params,
                    notes=(
                        "Lagged relative-momentum convergence for short-history metals pairs "
                        f"on {timeframe} ({spec['variant']})."
                    ),
                    tags=("lag_convergence", "metals", "pair", "relative_momentum"),
                    metadata={
                        "timeframe": timeframe,
                        "pair": f"{symbol_x}_{symbol_y}",
                        "pair_variant": str(spec["variant"]),
                    },
                )

    # Optional carry/crowding sleeve.
    if _has_perp_support_data() and carry_tfs:
        for timeframe in carry_tfs:
            tf_tag = timeframe.replace("/", "-")
            for entry, exit_th in ((0.25, 0.08), (0.35, 0.10), (0.45, 0.15)):
                params = {
                    "entry_threshold": float(entry),
                    "exit_threshold": float(exit_th),
                    "mild_funding": 0.0002,
                    "extreme_funding": 0.0012,
                    "stop_loss_pct": 0.02,
                    "max_hold_bars": 72,
                    "allow_short": True,
                }
                _add_candidate(
                    candidates,
                    name=f"perp_crowding_carry_{tf_tag}_{entry:.2f}_{exit_th:.2f}",
                    family="carry",
                    strategy_class="PerpCrowdingCarryStrategy",
                    timeframe=timeframe,
                    symbols=crypto_symbols,
                    params=params,
                    notes="Funding/OI crowding-aware carry sleeve.",
                    tags=("carry", "perp", "funding", "crowding"),
                    metadata={
                        "timeframe": timeframe,
                        "data_dependent": _has_perp_support_data(),
                        "symbol_scope": "crypto",
                    },
                )

    # Research-only micro sleeve.
    for timeframe in micro_tfs:
        tf_tag = timeframe.replace("/", "-")
        for lookback, range_z, vol_z in ((20, 1.2, 0.8), (30, 1.5, 1.0), (45, 2.0, 1.2)):
            params = {
                "lookback": int(lookback),
                "range_z_threshold": float(range_z),
                "volume_z_threshold": float(vol_z),
                "max_hold_bars": 20,
                "allow_short": True,
            }
            _add_candidate(
                candidates,
                name=f"micro_range_expansion_{tf_tag}_{lookback}_{range_z:.1f}_{vol_z:.1f}",
                family="micro",
                strategy_class="MicroRangeExpansion1sStrategy",
                timeframe=timeframe,
                symbols=crypto_symbols,
                params=params,
                notes="Research-only micro breakout sleeve with strict turnover controls.",
                tags=("micro", "range", "breakout", "research"),
                metadata={
                    "timeframe": timeframe,
                    "research_only": True,
                },
            )

    return candidates


def build_candidate_manifest(
    *,
    timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
    symbols: Sequence[str] = DEFAULT_BINANCE_TOP10_PLUS_METALS,
) -> dict[str, Any]:
    """Build a JSON-ready manifest with aggregate metadata."""
    normalized_symbols = tuple(canonicalize_symbol_list(symbols))
    normalized_timeframes = tuple(
        normalize_strategy_timeframes(
            list(timeframes),
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    candidates = build_binance_futures_candidates(
        timeframes=normalized_timeframes,
        symbols=normalized_symbols,
    )

    family_counts: dict[str, int] = {}
    strategy_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}

    for candidate in candidates:
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
        strategy_counts[candidate.strategy_class] = strategy_counts.get(candidate.strategy_class, 0) + 1
        timeframe_counts[candidate.timeframe] = timeframe_counts.get(candidate.timeframe, 0) + 1

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol_universe": list(normalized_symbols),
        "timeframes": list(normalized_timeframes),
        "candidate_count": len(candidates),
        "family_counts": family_counts,
        "strategy_counts": strategy_counts,
        "timeframe_counts": timeframe_counts,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
