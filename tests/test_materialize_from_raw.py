from __future__ import annotations

from datetime import UTC, datetime

from lumina_quant.storage.parquet import ParquetMarketDataRepository
from lumina_quant.services.materialize_from_raw import materialize_raw_aggtrades


def test_materialize_from_raw_produces_deterministic_committed_manifest(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_700_000_000_500,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
            {
                "agg_trade_id": 3,
                "timestamp_ms": 1_700_000_001_000,
                "price": 101.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
        ],
    )

    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload={"observed_until_ms": 1_700_000_000_999, "last_timestamp_ms": 1_700_000_000_500},
    )

    first = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date=None,
        end_date=None,
    )
    second = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date=None,
        end_date=None,
    )

    assert first
    assert second
    assert first[0].canonical_row_checksum == second[0].canonical_row_checksum

    loaded = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
    )
    assert loaded.height >= 1


def test_materialize_explicit_historical_range_ignores_stale_older_checkpoint(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    start_ms = int(datetime(2026, 3, 19, tzinfo=UTC).timestamp() * 1000)
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 10,
                "timestamp_ms": start_ms,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 11,
                "timestamp_ms": start_ms + 1_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
        ],
    )
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload={
            "observed_until_ms": int(datetime(2026, 3, 18, 23, 59, tzinfo=UTC).timestamp() * 1000),
            "last_timestamp_ms": int(datetime(2026, 3, 18, 23, 59, tzinfo=UTC).timestamp() * 1000),
        },
    )

    commits = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date="2026-03-19T00:00:00+00:00",
        end_date="2026-03-19T00:00:02+00:00",
    )

    assert commits
    assert commits[0].partition == "2026-03-19"
