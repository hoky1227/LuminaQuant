from __future__ import annotations

from datetime import date
import importlib.util
import sys
from pathlib import Path

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
