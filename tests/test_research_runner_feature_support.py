from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from lumina_quant.strategy_factory import research_runner


def _minute_datetimes(length: int) -> np.ndarray:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values = [start + timedelta(minutes=idx) for idx in range(length)]
    return pl.Series("datetime", values).to_numpy()


def _bundle(symbol: str, *, length: int = 420) -> research_runner.SeriesBundle:
    close = np.linspace(100.0, 120.0, length, dtype=float)
    return research_runner.SeriesBundle(
        symbol=symbol,
        timeframe="1m",
        datetime=_minute_datetimes(length),
        open=close - 0.2,
        high=close + 0.4,
        low=close - 0.5,
        close=close,
        volume=np.linspace(1_000.0, 1_400.0, length, dtype=float),
    )


def test_align_bundles_augments_feature_series_from_feature_cache():
    bundle = _bundle("BTC/USDT")
    datetimes = pl.Series("datetime", bundle.datetime)
    feature_frame = pl.DataFrame(
        {
            "datetime": datetimes,
            "funding_rate": np.linspace(0.00005, 0.00025, datetimes.len(), dtype=float),
            "funding_mark_price": np.linspace(100.1, 120.1, datetimes.len(), dtype=float),
            "funding_fee_rate": np.linspace(0.00005, 0.00025, datetimes.len(), dtype=float),
            "funding_fee_quote_per_unit": np.linspace(0.005, 0.03, datetimes.len(), dtype=float),
            "mark_price": np.linspace(100.1, 120.1, datetimes.len(), dtype=float),
            "index_price": np.linspace(100.0, 120.0, datetimes.len(), dtype=float),
            "open_interest": np.linspace(1_000_000.0, 1_600_000.0, datetimes.len(), dtype=float),
            "liquidation_long_qty": np.linspace(10.0, 20.0, datetimes.len(), dtype=float),
            "liquidation_short_qty": np.linspace(5.0, 8.0, datetimes.len(), dtype=float),
            "liquidation_long_notional": np.linspace(100_000.0, 140_000.0, datetimes.len(), dtype=float),
            "liquidation_short_notional": np.linspace(30_000.0, 40_000.0, datetimes.len(), dtype=float),
        }
    )

    aligned = research_runner._align_bundles(
        [bundle],
        feature_cache={"BTC/USDT": feature_frame},
    )

    assert aligned is not None
    assert "BTC/USDT:funding_rate" in aligned
    assert "BTC/USDT:open_interest" in aligned
    assert "BTC/USDT:crowding_score" in aligned
    assert np.isclose(float(aligned["BTC/USDT:funding_fee_rate"][-1]), 0.00025)
    assert np.isfinite(float(aligned["BTC/USDT:crowding_score"][-1]))


def test_perp_carry_strategy_signal_uses_actual_support_feature_arrays():
    length = 420
    close = np.linspace(100.0, 118.0, length, dtype=float)
    funding = 0.00005 + np.linspace(0.0, 0.00020, length, dtype=float)
    open_interest = 1_000_000.0 * np.exp(np.linspace(0.0, 0.25, length, dtype=float))
    liq_long = 120_000.0 + np.linspace(0.0, 40_000.0, length, dtype=float)
    liq_short = 20_000.0 + np.linspace(0.0, 5_000.0, length, dtype=float)

    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(1_000.0, 1_500.0, length, dtype=float),
        "BTC/USDT:funding_rate": funding,
        "BTC/USDT:open_interest": open_interest,
        "BTC/USDT:liquidation_long_notional": liq_long,
        "BTC/USDT:liquidation_short_notional": liq_short,
        "BTC/USDT:mark_price": close * 1.001,
        "BTC/USDT:index_price": close,
    }
    candidate = {
        "strategy_class": "PerpCrowdingCarryStrategy",
        "params": {
            "window": 96,
            "mild_funding": 0.0004,
            "extreme_funding": 0.002,
            "entry_threshold": 0.05,
            "exit_threshold": 0.02,
            "stop_loss_pct": 0.05,
            "max_hold_bars": 512,
            "allow_short": True,
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert returns_raw.shape == (length,)
    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta.get("missing_support_data") is not True
    assert np.any(np.abs(exposure) > 0.0)


def test_composite_trend_blocks_entries_when_crowding_score_is_extreme():
    length = 420
    close = np.linspace(100.0, 160.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(2_000.0, 4_000.0, length, dtype=float),
        "BTC/USDT:crowding_score": np.full(length, 1.0, dtype=float),
    }
    candidate = {
        "strategy_class": "CompositeTrendStrategy",
        "params": {
            "long_threshold": 0.10,
            "short_threshold": 0.10,
            "te_min": 0.0,
            "vr_min": 0.0,
            "crowding_reduce_threshold": 0.55,
            "crowding_block_threshold": 0.85,
        },
    }

    _, _, exposure, _ = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert np.allclose(exposure, 0.0)


def test_composite_trend_respects_long_only_retune_and_exposure_damping():
    length = 420
    close = np.linspace(160.0, 100.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close + 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(2_000.0, 4_000.0, length, dtype=float),
    }

    shortable = {
        "strategy_class": "CompositeTrendStrategy",
        "params": {
            "long_threshold": 0.10,
            "short_threshold": 0.10,
            "exit_score_cross": 0.03,
            "te_min": 0.0,
            "vr_min": 0.0,
            "risk_target_vol": 0.0025,
            "max_signal_strength": 0.60,
            "allow_short": True,
        },
    }
    long_only = {
        "strategy_class": "CompositeTrendStrategy",
        "params": {
            **dict(shortable["params"]),
            "allow_short": False,
        },
    }

    _, _, exposure_shortable, _ = research_runner._strategy_signal(
        shortable,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )
    _, _, exposure_long_only, _ = research_runner._strategy_signal(
        long_only,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert np.min(exposure_shortable) < -0.05
    assert np.max(np.abs(exposure_shortable)) <= 0.60 + 1e-9
    assert np.all(exposure_long_only >= -1e-9)
    assert np.max(np.abs(exposure_long_only)) < 0.05


def test_lag_convergence_strategy_signal_trades_xpt_xpd_pair():
    length = 120
    x_close = np.full(length, 100.0, dtype=float)
    y_close = np.full(length, 100.0, dtype=float)
    x_close[10:] = 102.5
    y_close[10:] = 100.0
    x_close[40:] = 100.2
    y_close[40:] = 100.1

    aligned = {
        "datetime": _minute_datetimes(length),
        "XPT/USDT:open": x_close,
        "XPT/USDT:high": x_close + 0.2,
        "XPT/USDT:low": x_close - 0.2,
        "XPT/USDT:close": x_close,
        "XPT/USDT:volume": np.full(length, 100.0, dtype=float),
        "XPD/USDT:open": y_close,
        "XPD/USDT:high": y_close + 0.2,
        "XPD/USDT:low": y_close - 0.2,
        "XPD/USDT:close": y_close,
        "XPD/USDT:volume": np.full(length, 100.0, dtype=float),
    }
    candidate = {
        "strategy_class": "LagConvergenceStrategy",
        "params": {
            "symbol_x": "XPT/USDT",
            "symbol_y": "XPD/USDT",
            "lag_bars": 1,
            "entry_threshold": 0.01,
            "exit_threshold": 0.002,
            "stop_threshold": 0.05,
            "max_hold_bars": 24,
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["XPT/USDT", "XPD/USDT"],
    )

    assert returns_raw.shape == (length,)
    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta == {}
    assert np.allclose(exposure, 0.0)
    assert np.any(turnover > 0.0)


def test_mean_reversion_std_strategy_signal_produces_exposure():
    length = 180
    close = np.full(length, 100.0, dtype=float)
    close[80] = 92.0
    close[81:] = np.linspace(93.0, 101.0, length - 81, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 100.0, dtype=float),
    }
    candidate = {
        "strategy_class": "MeanReversionStdStrategy",
        "params": {
            "window": 32,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "stop_loss_pct": 0.03,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)


def test_rolling_breakout_strategy_signal_produces_exposure():
    length = 160
    close = np.linspace(100.0, 112.0, length, dtype=float)
    high = close + 0.5
    low = close - 0.5
    close[90:] += 3.0
    high[90:] = close[90:] + 0.5
    low[90:] = close[90:] - 0.5
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": high,
        "BTC/USDT:low": low,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }
    candidate = {
        "strategy_class": "RollingBreakoutStrategy",
        "params": {
            "lookback_bars": 24,
            "breakout_buffer": 0.0,
            "atr_window": 8,
            "atr_stop_multiplier": 1.5,
            "stop_loss_pct": 0.02,
            "allow_short": False,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(exposure > 0.0)
    assert np.any(turnover > 0.0)


def test_topcap_tsmom_strategy_signal_produces_cross_sectional_exposure():
    length = 120
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for idx, symbol in enumerate(symbols):
        base = 100.0 + (idx * 10.0)
        close = np.full(length, base, dtype=float)
        if symbol in {"ETH/USDT", "BNB/USDT"}:
            close[20:] = np.linspace(base, base * 1.25, length - 20, dtype=float)
        elif symbol in {"SOL/USDT", "TRX/USDT"}:
            close[20:] = np.linspace(base, base * 0.85, length - 20, dtype=float)
        else:
            close[20:] = np.linspace(base, base * 1.05, length - 20, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (length,)
    assert meta.get("cross_sectional") is True
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)


def test_topcap_tsmom_strategy_signal_supports_zero_short_budget():
    length = 120
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for idx, symbol in enumerate(symbols):
        base = 100.0 + (idx * 10.0)
        close = np.full(length, base, dtype=float)
        if symbol in {"ETH/USDT", "BNB/USDT"}:
            close[20:] = np.linspace(base, base * 1.25, length - 20, dtype=float)
        elif symbol in {"SOL/USDT", "TRX/USDT"}:
            close[20:] = np.linspace(base, base * 0.85, length - 20, dtype=float)
        else:
            close[20:] = np.linspace(base, base * 1.05, length - 20, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 0,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (length,)
    assert meta.get("cross_sectional") is True
    assert np.any(exposure > 0.0)
    assert np.all(exposure >= 0.0)
    assert np.any(turnover > 0.0)


def test_regime_breakout_strategy_signal_still_trades_persistent_breakout():
    length = 160
    close = np.linspace(100.0, 170.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 250.0, dtype=float),
    }
    candidate = {
        "strategy_class": "RegimeBreakoutCandidateStrategy",
        "params": {
            "lookback_window": 36,
            "slope_window": 18,
            "volatility_fast_window": 10,
            "volatility_slow_window": 40,
            "range_entry_threshold": 0.65,
            "slope_entry_threshold": 0.0008,
            "momentum_floor": 0.002,
            "max_volatility_ratio": 2.5,
            "stop_loss_pct": 0.02,
            "allow_short": False,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(exposure > 0.0)
    assert np.any(turnover > 0.0)


def test_regime_breakout_strategy_signal_respects_composite_momentum_gate():
    length = 160
    down = np.linspace(220.0, 100.0, 120, dtype=float)
    rebound = np.linspace(100.0, 108.0, length - 120, dtype=float)
    close = np.concatenate([down, rebound])
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 250.0, dtype=float),
    }
    candidate = {
        "strategy_class": "RegimeBreakoutCandidateStrategy",
        "params": {
            "lookback_window": 36,
            "slope_window": 18,
            "volatility_fast_window": 10,
            "volatility_slow_window": 40,
            "range_entry_threshold": 0.65,
            "slope_entry_threshold": 0.0008,
            "momentum_floor": 0.01,
            "max_volatility_ratio": 2.5,
            "stop_loss_pct": 0.02,
            "allow_short": False,
        },
    }

    _, turnover, exposure, _ = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert np.all(exposure >= 0.0)
    assert not np.any(exposure > 0.0)
    assert not np.any(turnover > 0.0)


def _pair_aligned_prices(length: int = 320) -> dict[str, np.ndarray]:
    x_close = []
    y_close = []
    for idx in range(length):
        y_price = 100.0 + (0.08 * idx)
        spread_noise = 0.08 * np.sin(idx / 5.0)
        if 220 <= idx <= 236:
            spread_noise += 1.6
        if 280 <= idx <= 295:
            spread_noise -= 1.7
        x_price = y_price + spread_noise
        x_close.append(x_price)
        y_close.append(y_price)

    x_close_arr = np.asarray(x_close, dtype=float)
    y_close_arr = np.asarray(y_close, dtype=float)
    return {
        "datetime": _minute_datetimes(length),
        "XAU/USDT:open": x_close_arr,
        "XAU/USDT:high": x_close_arr + 0.2,
        "XAU/USDT:low": x_close_arr - 0.2,
        "XAU/USDT:close": x_close_arr,
        "XAU/USDT:volume": np.full(length, 100.0, dtype=float),
        "XAG/USDT:open": y_close_arr,
        "XAG/USDT:high": y_close_arr + 0.2,
        "XAG/USDT:low": y_close_arr - 0.2,
        "XAG/USDT:close": y_close_arr,
        "XAG/USDT:volume": np.full(length, 100.0, dtype=float),
    }


def test_pair_trading_strategy_signal_uses_event_driven_proxy():
    aligned = _pair_aligned_prices()
    candidate = {
        "strategy_class": "PairTradingZScoreStrategy",
        "params": {
            "lookback_window": 30,
            "hedge_window": 60,
            "entry_z": 1.5,
            "exit_z": 0.25,
            "stop_z": 4.5,
            "min_correlation": -1.0,
            "max_hold_bars": 120,
            "cooldown_bars": 0,
            "reentry_z_buffer": 0.0,
            "symbol_x": "XAU/USDT",
            "symbol_y": "XAG/USDT",
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["XAU/USDT", "XAG/USDT"],
    )

    assert returns_raw.shape == (320,)
    assert turnover.shape == (320,)
    assert exposure.shape == (320,)
    assert meta.get("event_driven_proxy") is True
    assert np.any(turnover > 0.0)
    assert np.any(np.abs(returns_raw) > 0.0)


def test_pair_trading_strategy_signal_respects_reentry_buffer():
    aligned = _pair_aligned_prices()
    candidate = {
        "strategy_class": "PairTradingZScoreStrategy",
        "params": {
            "lookback_window": 30,
            "hedge_window": 60,
            "entry_z": 1.5,
            "exit_z": 0.25,
            "stop_z": 4.5,
            "min_correlation": -1.0,
            "max_hold_bars": 120,
            "cooldown_bars": 0,
            "reentry_z_buffer": 10.0,
            "symbol_x": "XAU/USDT",
            "symbol_y": "XAG/USDT",
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["XAU/USDT", "XAG/USDT"],
    )

    assert returns_raw.shape == (320,)
    assert turnover.shape == (320,)
    assert exposure.shape == (320,)
    assert meta.get("event_driven_proxy") is True
    assert not np.any(turnover > 0.0)
    assert not np.any(np.abs(returns_raw) > 0.0)


def test_alpha101_formula_strategy_signal_uses_event_driven_proxy():
    length = 96
    open_price = np.linspace(100.0, 112.0, length, dtype=float)
    body = np.concatenate(
        [
            np.full(24, 0.05, dtype=float),
            np.full(24, 0.35, dtype=float),
            np.full(24, -0.35, dtype=float),
            np.full(24, 0.0, dtype=float),
        ]
    )
    close = open_price + body
    high = np.maximum(open_price, close) + 0.08
    low = np.minimum(open_price, close) - 0.08
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": open_price,
        "BTC/USDT:high": high,
        "BTC/USDT:low": low,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(1000.0, 1200.0, length, dtype=float),
    }
    candidate = {
        "strategy_class": "Alpha101FormulaStrategy",
        "params": {
            "alpha_id": 101,
            "rank_window": 20,
            "history_window": 24,
            "score_window": 8,
            "entry_z": 0.9,
            "exit_z": 0.15,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": True,
            "alpha_param_overrides": {"alpha101.101.const.001": 0.01},
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert returns_raw.shape == (length,)
    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta.get("event_driven_proxy") is True
    assert meta.get("formulaic_alpha101") is True
    assert meta.get("alpha_param_override_count") == 1
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
