from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from lumina_quant.storage.parquet import ParquetMarketDataRepository

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "refresh_final_portfolio_validation_data.py"
SPEC = importlib.util.spec_from_file_location("refresh_final_portfolio_validation_data", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load refresh_final_portfolio_validation_data module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_portfolio_symbols_preserves_saved_weight_order(tmp_path: Path) -> None:
    payload = {
        "weights": [
            {"symbols": ["BNB/USDT", "TRX/USDT"]},
            {"symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"]},
        ]
    }
    path = tmp_path / "portfolio.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert MODULE.load_portfolio_symbols(path) == ["BNB/USDT", "TRX/USDT", "BTC/USDT", "ETH/USDT"]


def test_load_feature_symbols_filters_to_required_strategy_classes(tmp_path: Path) -> None:
    payload = {
        "selected_team": [
            {"strategy_class": "CompositeTrendStrategy", "symbols": ["BTC/USDT", "ETH/USDT"]},
            {"strategy_class": "TopCapTimeSeriesMomentumStrategy", "symbols": ["BTC/USDT", "BNB/USDT"]},
            {"strategy_class": "PerpCrowdingCarryStrategy", "symbols": ["SOL/USDT"]},
        ]
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert MODULE.load_feature_symbols(path) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_latest_runtime_tail_uses_runtime_second_not_previous_day() -> None:
    now = MODULE.parse_utc("2026-03-19T09:30:29.591000Z")
    assert MODULE.iso_utc(MODULE.latest_runtime_tail_utc(now)) == "2026-03-19T09:30:29Z"


def test_iso_utc_treats_naive_datetime_as_utc() -> None:
    assert MODULE.iso_utc(datetime(2026, 3, 18, 23, 59, 58)) == "2026-03-18T23:59:58Z"


def _build_archive_zip(rows: list[tuple[int, float, float, int, bool]]) -> bytes:
    payload = "\n".join(
        f"{agg_trade_id},{price},{quantity},0,0,{timestamp_ms},{str(is_buyer_maker).lower()},true"
        for agg_trade_id, price, quantity, timestamp_ms, is_buyer_maker in rows
    )
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BTCUSDT-aggTrades-2025-01-01.csv", payload)
    return blob.getvalue()


def test_refresh_symbol_raw_first_ohlcv_derives_from_stored_raw_aggtrades(
    tmp_path: Path, monkeypatch
) -> None:
    repo = ParquetMarketDataRepository(str(tmp_path))
    cutoff_dt = MODULE.parse_utc("2025-01-01T00:00:02Z")
    floor_dt = MODULE.parse_utc("2025-01-01T00:00:00Z")
    assert cutoff_dt is not None
    assert floor_dt is not None

    archive_zip = _build_archive_zip(
        [
            (1, 100.0, 0.1, 1_735_689_600_000, False),
            (2, 101.0, 0.2, 1_735_689_600_500, True),
            (3, 102.0, 0.3, 1_735_689_601_500, False),
        ]
    )

    monkeypatch.setattr(MODULE, "_download_zip_bytes", lambda *args, **kwargs: archive_zip)
    monkeypatch.setattr(MODULE, "_binance_archive_url", lambda *args, **kwargs: "https://example.test")
    monkeypatch.setattr(
        MODULE,
        "_collect_live_raw_rows",
        lambda **kwargs: [],
    )

    result = MODULE.refresh_symbol_raw_first_ohlcv(
        repo=repo,
        symbol="BTC/USDT",
        db_path=str(tmp_path),
        exchange_id="binance",
        cutoff_dt=cutoff_dt,
        floor_dt=floor_dt,
        guard=None,
    )

    raw = repo.load_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-01T00:00:02Z",
    )
    ohlcv = repo.load_ohlcv(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-01-01T00:00:02Z",
    )

    assert raw.height == 3
    assert ohlcv.height == 2
    assert result.after_raw_agg_trade_utc == "2025-01-01T00:00:01.500000Z"
    assert result.after_ohlcv_max_utc == "2025-01-01T00:00:01Z"
    assert result.live_raw_rows_upserted == 0
    assert result.derived_ohlcv_rows_upserted >= 2


def test_collect_live_raw_rows_reduces_limit_after_rate_limit(monkeypatch) -> None:
    class _Exchange:
        def close(self):
            return None

    calls: list[int] = []
    state = {"attempt": 0}

    monkeypatch.setattr(MODULE, "create_binance_futures_client", lambda **kwargs: _Exchange())
    monkeypatch.setattr(MODULE.time, "sleep", lambda *_args, **_kwargs: None)

    def _fetch(*, exchange, symbol, since_ms, limit, retries, base_wait_sec):
        _ = exchange, symbol, since_ms, retries, base_wait_sec
        calls.append(int(limit))
        state["attempt"] += 1
        if state["attempt"] == 1:
            raise RuntimeError("Too Many Requests")
        return [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ]

    monkeypatch.setattr(MODULE, "fetch_aggtrades_batch", _fetch)

    rows = MODULE._collect_live_raw_rows(
        symbol="BTC/USDT",
        start_ms=1_735_689_600_000,
        end_ms=1_735_689_600_000,
        limit=1000,
        pause_sec=0.0,
    )

    assert calls[:2] == [1000, 500]
    assert len(rows) == 1
