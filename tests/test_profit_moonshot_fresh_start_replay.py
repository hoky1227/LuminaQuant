from __future__ import annotations

from datetime import date
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts" / "research" / "replay_profit_moonshot_fresh_start.py"
SPEC = importlib.util.spec_from_file_location("replay_profit_moonshot_fresh_start", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load replay_profit_moonshot_fresh_start module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from scripts.research.replay_eth_shock_filters import _materialized_paths
from scripts.research.replay_profit_moonshot_fresh_start import FreshSpec
from datetime import datetime, timezone


def test_fresh_start_hourly_loader_matches_raw_first_1s_aggregation_when_data_exists() -> None:
    market_root = Path("data/market_parquet")
    day = date(2026, 5, 5)
    one_second_paths = _materialized_paths(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start=day,
        end=day,
    )
    one_hour_paths = _materialized_paths(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start=day,
        end=day,
    )
    if not one_second_paths or not one_hour_paths:
        pytest.skip("raw-first materialized fixture data is not present")

    loaded = MODULE._load_symbol_hourly(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        start=day,
        end=day,
    )
    aggregated = (
        pl.scan_parquet(one_second_paths)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("open").first().alias("btcusdt_open"),
                pl.col("high").max().alias("btcusdt_high"),
                pl.col("low").min().alias("btcusdt_low"),
                pl.col("close").last().alias("btcusdt_close"),
                pl.col("volume").sum().alias("btcusdt_volume"),
            ]
        )
        .drop_nulls(["btcusdt_open", "btcusdt_close"])
        .collect()
        .sort("datetime")
    )

    assert loaded.height == aggregated.height
    assert loaded.select("datetime", "btcusdt_close").to_dicts() == aggregated.select(
        "datetime", "btcusdt_close"
    ).to_dicts()


def test_candidate_signal_adaptive_trend_and_persistence_rules() -> None:
    arrays = {
        "datetime": [datetime(2026, 5, 1, h, tzinfo=timezone.utc) for h in range(6)],
        "symbols": ("BTC/USDT",),
        "btcusdt_open": [100.0] * 6,
        "btcusdt_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "btcusdt_ret_12h": np.array([np.nan] * 5 + [0.05]),
        "btcusdt_ret_3h": np.array([np.nan, np.nan, np.nan, 0.01, 0.01, 0.01]),
        "btcusdt_resid_z_12h": [np.nan] * 6,
        "btcusdt_resid_z_3h": [np.nan] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "btcusdt_open_interest_ffill": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
        "market_ret_12h": np.array([np.nan] * 5 + [0.05]),
        "btcusdt_flow_imbalance_3h": [0.25] * 6,
        "btcusdt_flow_imbalance_2h": [0.25] * 6,
    }
    arrays["btcusdt_oi_delta_12h"] = np.full(6, 0.02)
    arrays["btcusdt_ret_6h"] = np.array([np.nan] * 5 + [0.03])
    arrays["market_ret_6h"] = np.array([np.nan] * 5 + [0.02])
    spec = FreshSpec(
        name="adaptive_trend_test",
        family="adaptive_trend",
        lookback_bars=12,
        threshold=0.01,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        adaptive_lookback_bars=12,
    )
    symbol, side, reason = MODULE._candidate_signal(spec, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"

    persistence = FreshSpec(
        name="flow_persistence_test",
        family="flow_imbalance_persistence",
        lookback_bars=3,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        flow_lookback_bars=2,
        flow_threshold=0.0,
        flow_persistence_bars=3,
        flow_persistence_threshold=0.2,
        min_abs_return=0.001,
    )
    arrays["btcusdt_ret_3h"] = np.array([np.nan, np.nan, 0.002, 0.003, 0.003, 0.003])
    arrays["btcusdt_resid_z_3h"] = np.array([np.nan] * 6)
    symbol, side, reason = MODULE._candidate_signal(persistence, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"


def test_candidate_signal_cross_sharpe_rank_and_funding_oi() -> None:
    dt = [datetime(2026, 5, 1, h, tzinfo=timezone.utc) for h in range(6)]
    arrays = {
        "datetime": dt,
        "symbols": ("BTC/USDT", "ETH/USDT"),
        "btcusdt_open": [100.0] * 6,
        "btcusdt_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "ethusdt_open": [50.0] * 6,
        "ethusdt_close": [50.0, 49.0, 49.0, 49.0, 49.0, 49.0],
        "btcusdt_ret_6h": np.array([np.nan] * 5 + [0.03]),
        "ethusdt_ret_6h": np.array([np.nan] * 5 + [-0.03]),
        "btcusdt_resid_z_6h": [np.nan] * 6,
        "ethusdt_resid_z_6h": [np.nan] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "ethusdt_funding_ffill": [0.0] * 6,
        "btcusdt_oi_delta_12h": np.array([np.nan] * 5 + [0.5]),
        "ethusdt_oi_delta_12h": np.array([np.nan] * 5 + [-0.5]),
        "btcusdt_open_interest_ffill": [20.0] * 6,
        "ethusdt_open_interest_ffill": [20.0] * 6,
        "btcusdt_flow_imbalance_3h": [0.0] * 6,
        "ethusdt_flow_imbalance_3h": [0.0] * 6,
        "market_ret_24h": np.array([np.nan] * 6),
        "market_ret_6h": np.array([np.nan] * 5 + [0.005]),
    }
    arrays["btcusdt_ret_24h"] = np.array([np.nan] * 5 + [0.03])
    arrays["ethusdt_ret_24h"] = np.array([np.nan] * 5 + [-0.03])
    arrays["btcusdt_ret_3h"] = np.array([np.nan] * 3 + [0.01, 0.01, 0.01])
    arrays["ethusdt_ret_3h"] = np.array([np.nan] * 3 + [0.01, 0.01, 0.01])
    arrays["btcusdt_resid_z_3h"] = np.array([np.nan] * 3 + [0.4, 0.6, 0.8])
    arrays["ethusdt_resid_z_3h"] = np.array([np.nan] * 3 + [0.1, 0.2, 0.3])

    cross = FreshSpec(
        name="cross_sharpe_rank",
        family="cross_sectional_sharpe_rank",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        sharpe_lookback_bars=6,
        sharpe_rank_min=0.0,
        min_abs_return=0.001,
    )
    symbol, side, reason = MODULE._candidate_signal(cross, arrays, 5)
    assert (symbol, side) == ("BTC/USDT", "LONG")
    assert reason == ""

    oi = FreshSpec(
        name="funding_oi",
        family="funding_oi_carry_fade",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        funding_rank_min=0.01,
        oi_rank_min=0.1,
        sharpe_lookback_bars=12,
        min_abs_return=0.001,
    )
    arrays["btcusdt_funding_ffill"] = [0.02] * 6
    arrays["ethusdt_funding_ffill"] = [-0.02] * 6
    symbol, side, reason = MODULE._candidate_signal(oi, arrays, 5)
    assert (symbol, side) in {("BTC/USDT", "SHORT"), ("ETH/USDT", "LONG")}
    assert reason == ""
