from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from lumina_quant.data.raw_first_lineage import (
    normalize_exchange_timestamp_ms,
    raw_aggtrades_to_1s_frame,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def test_normalize_exchange_timestamp_ms_rejects_non_ms_units() -> None:
    assert normalize_exchange_timestamp_ms(1_700_000_000_000, source="rest") == 1_700_000_000_000
    with pytest.raises(ValueError, match="seconds-like"):
        normalize_exchange_timestamp_ms(1_700_000_000, source="rest")
    with pytest.raises(ValueError, match="microseconds-like"):
        normalize_exchange_timestamp_ms(1_700_000_000_000_000, source="rest")



def test_raw_aggtrades_to_1s_frame_drops_incomplete_last_second() -> None:
    frame = raw_aggtrades_to_1s_frame(
        [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.0,
                "quantity": 1.0,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_700_000_000_500,
                "price": 101.0,
                "quantity": 2.0,
                "is_buyer_maker": True,
            },
            {
                "agg_trade_id": 3,
                "timestamp_ms": 1_700_000_001_100,
                "price": 102.0,
                "quantity": 3.0,
                "is_buyer_maker": False,
            },
        ],
        source="pytest",
        range_start_ms=1_700_000_000_000,
        range_end_ms=1_700_000_001_100,
        complete_through_ms=1_700_000_001_100,
    )

    assert frame.height == 1
    assert frame["open"][0] == pytest.approx(100.0)
    assert frame["close"][0] == pytest.approx(101.0)
    assert frame["volume"][0] == pytest.approx(3.0)



def test_committed_loader_rebuilds_higher_timeframe_from_1s_and_truncates_incomplete_tail(
    tmp_path,
) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    partition_date = "2026-03-01"
    partition_root = repo.materialized_partition_root(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
    )
    commit_id = "20260301-1s"
    commit_dir = partition_root / f"commit={commit_id}"
    commit_dir.mkdir(parents=True, exist_ok=True)

    base_dt = datetime(2026, 3, 1, 0, 0, tzinfo=UTC).replace(tzinfo=None)
    frame = pl.DataFrame(
        {
            "datetime": [base_dt + timedelta(seconds=index) for index in range(90)],
            "open": [100.0 + (index * 0.1) for index in range(90)],
            "high": [100.5 + (index * 0.1) for index in range(90)],
            "low": [99.5 + (index * 0.1) for index in range(90)],
            "close": [100.2 + (index * 0.1) for index in range(90)],
            "volume": [1.0 for _ in range(90)],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))
    data_file = commit_dir / "part-0000.parquet"
    frame.write_parquet(data_file)

    checksum = repo.canonical_row_checksum(frame)
    manifest_payload = {
        "manifest_version": 1,
        "commit_id": commit_id,
        "symbol": "BTC/USDT",
        "timeframe": "1s",
        "partition": str(partition_root),
        "window_start_ms": int(frame["datetime"].min().timestamp() * 1000),
        "window_end_ms": int(frame["datetime"].max().timestamp() * 1000),
        "event_time_watermark_ms": int(frame["datetime"].max().timestamp() * 1000),
        "source_checkpoint_start": int(frame["datetime"].min().timestamp() * 1000),
        "source_checkpoint_end": int(frame["datetime"].max().timestamp() * 1000),
        "row_count": int(frame.height),
        "canonical_row_checksum": checksum,
        "data_files": [f"commit={commit_id}/part-0000.parquet"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "producer": "pytest",
        "status": "committed",
    }
    repo.write_materialized_manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
        payload=manifest_payload,
    )

    rebuilt = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1m",
        start_date="2026-03-01T00:00:00Z",
        end_date="2026-03-01T00:01:29Z",
        chunk_days=1,
        warmup_bars=0,
        staleness_threshold_seconds=None,
    )

    assert rebuilt.height == 1
    assert rebuilt["datetime"][0] == base_dt
    assert rebuilt["open"][0] == pytest.approx(100.0)
    assert rebuilt["close"][0] == pytest.approx(106.1)
    assert rebuilt["volume"][0] == pytest.approx(60.0)
