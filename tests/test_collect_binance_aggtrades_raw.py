from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.data_collector import collect_binance_aggtrades_raw
from lumina_quant.storage.parquet import ParquetMarketDataRepository


class _ExchangeStub:
    def close(self):
        return None


def test_collect_binance_aggtrades_raw_checkpoint_resume(tmp_path, monkeypatch):
    calls: list[int] = []
    state = {"last": 0}

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        start_ms = int(kwargs["start_ms"])
        calls.append(start_ms)
        if start_ms == 0:
            state["last"] = 1_700_000_001_000
            return SimpleNamespace(
                fetched_rows=2,
                upserted_rows=2,
                first_timestamp_ms=1_700_000_000_000,
                last_timestamp_ms=1_700_000_001_000,
                checkpoint_timestamp_ms=1_700_000_001_000,
                checkpoint_trade_id=2,
            )
        state["last"] = 1_700_000_002_000
        return SimpleNamespace(
            fetched_rows=1,
            upserted_rows=1,
            first_timestamp_ms=1_700_000_002_000,
            last_timestamp_ms=1_700_000_002_000,
            checkpoint_timestamp_ms=1_700_000_002_000,
            checkpoint_trade_id=3,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    first = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=0,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload={"last_timestamp_ms": state["last"], "last_trade_id": 2},
    )
    second = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    assert calls == [0, 1_700_000_001_001]
    assert int(first["fetched_rows"]) == 2
    assert int(second["fetched_rows"]) == 1
    assert int(second["last_trade_id"]) == 3


def test_collect_binance_aggtrades_raw_bootstrap_lookback_used_without_checkpoint(
    tmp_path, monkeypatch
):
    observed_since: list[int] = []

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        observed_since.append(int(kwargs["start_ms"]))
        return SimpleNamespace(
            fetched_rows=0,
            upserted_rows=0,
            first_timestamp_ms=None,
            last_timestamp_ms=None,
            checkpoint_timestamp_ms=None,
            checkpoint_trade_id=None,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    until_ms = 1_700_000_010_000
    result = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=until_ms,
        bootstrap_lookback_hours=2,
        limit=1000,
        max_batches=10,
    )

    expected_since = until_ms - (2 * 60 * 60 * 1000)
    assert observed_since == [expected_since]
    assert int(result["start_cursor_ms"]) == expected_since


def test_collect_binance_aggtrades_raw_recovers_from_corrupt_partition(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        repo = ParquetMarketDataRepository(str(tmp_path))
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    "agg_trade_id": 10,
                    "timestamp_ms": 1_700_000_100_000,
                    "price": 100.0,
                    "quantity": 0.1,
                    "is_buyer_maker": False,
                }
            ],
        )
        return SimpleNamespace(
            fetched_rows=1,
            upserted_rows=1,
            first_timestamp_ms=1_700_000_100_000,
            last_timestamp_ms=1_700_000_100_000,
            checkpoint_timestamp_ms=1_700_000_100_000,
            checkpoint_trade_id=10,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    corrupt_dir = (
        tmp_path
        / "market_data_raw_aggtrades"
        / "binance"
        / "BTCUSDT"
        / "date=2023-11-14"
    )
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "part-0000.parquet").write_bytes(b"not-a-parquet-file")

    result = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=1_700_000_100_000,
        until_ms=1_700_000_100_000,
        limit=1000,
        max_batches=10,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert int(result["upserted_rows"]) == 1
    assert raw.height == 1
    assert list(corrupt_dir.glob("part-0000.corrupt-*.parquet"))


def test_append_raw_aggtrades_recovers_from_corrupt_partition(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    part_path = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    )
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_bytes(b"not-a-real-parquet-file")

    written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )

    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")
    corrupt_files = sorted(part_path.parent.glob("part-0000.corrupt-*.parquet"))

    assert written == 1
    assert raw.height == 1
    assert corrupt_files
