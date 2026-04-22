from __future__ import annotations

from lumina_quant import market_data
from lumina_quant.market_data import (
    load_futures_feature_points_from_db,
    upsert_futures_feature_points_rows,
)


def test_feature_point_upsert_handles_sparse_mixed_numeric_rows(tmp_path):
    db_path = tmp_path / "market_parquet"

    upserted = upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="XAU/USDT",
        rows=[
            {"timestamp_ms": 1_700_000_000_000, "open_interest": 1},
            {
                "timestamp_ms": 1_700_000_060_000,
                "funding_rate": 0.00031,
                "funding_fee_rate": 0.00031,
                "funding_fee_quote_per_unit": 0.62,
            },
            {
                "timestamp_ms": 1_700_000_120_000,
                "liquidation_long_qty": 2,
                "liquidation_long_notional": 1.5,
            },
        ],
    )

    assert upserted == 3

    frame = load_futures_feature_points_from_db(
        str(db_path),
        exchange="binance",
        symbol="XAU/USDT",
    )
    assert frame.height == 3
    assert frame.get_column("funding_rate").drop_nulls().to_list() == [0.00031]
    assert frame.get_column("funding_fee_rate").drop_nulls().to_list() == [0.00031]
    assert frame.get_column("funding_fee_quote_per_unit").drop_nulls().to_list() == [0.62]


def test_feature_point_load_respects_date_partition_bounds(tmp_path):
    db_path = tmp_path / "market_parquet"

    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": 1_735_689_600_000, "funding_rate": 0.00010},  # 2025-01-02
            {"timestamp_ms": 1_741_910_400_000, "funding_rate": 0.00020},  # 2025-03-15
            {"timestamp_ms": 1_749_600_000_000, "funding_rate": 0.00030},  # 2025-06-12
        ],
    )

    frame = load_futures_feature_points_from_db(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        start_date="2025-03-01",
        end_date="2025-04-01",
    )

    assert frame.height == 1
    assert frame.get_column("funding_rate").to_list() == [0.00020]


def test_feature_point_load_emits_partition_and_collect_progress(tmp_path, monkeypatch):
    db_path = tmp_path / "market_parquet"

    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": 1_741_910_400_000, "funding_rate": 0.00020},
        ],
    )
    events: list[tuple[str, dict[str, object]]] = []
    counter = iter([10.0, 10.2, 20.0, 20.3])
    monkeypatch.setattr(market_data, "perf_counter", lambda: next(counter))

    frame = load_futures_feature_points_from_db(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        start_date="2025-03-01",
        end_date="2025-04-01",
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert frame.height == 1
    assert [name for name, _ in events] == [
        "resource_feature_partition_scan_completed",
        "resource_feature_collect_started",
        "resource_feature_collect_completed",
    ]
    assert events[0][1]["partition_count"] == 1
    assert events[0][1]["parquet_file_count"] == 1
    assert events[0][1]["elapsed_seconds"] == 0.2
    assert events[2][1]["row_count"] == 1
    assert events[2][1]["elapsed_seconds"] == 0.3
