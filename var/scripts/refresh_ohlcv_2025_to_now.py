from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lumina_quant.config import BaseConfig
from lumina_quant.data_sync import (
    create_binance_exchange,
    sync_symbol_ohlcv,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository, normalize_symbol

START_DT = datetime(2025, 1, 1, tzinfo=UTC)
TIMEFRAME = "1s"
REPORT_PATH = Path("var/reports/ohlcv_refresh_20250101_to_now_summary.json")


def _supports_symbol(exchange, symbol: str) -> bool:
    compact = normalize_symbol(symbol)
    try:
        exchange.market(compact)
        return True
    except Exception:
        return compact in getattr(exchange, "markets", {}) or f"{compact}:USDT" in getattr(
            exchange, "markets", {}
        )


def _datetime_payload(value: datetime | None) -> dict[str, object] | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return {
        "timestamp_ms": int(value.timestamp() * 1000),
        "datetime_utc": value.astimezone(UTC).isoformat(),
    }


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coverage_snapshot(
    repo: ParquetMarketDataRepository, *, exchange_id: str, symbol: str
) -> dict[str, object]:
    first_dt, last_dt = repo.get_symbol_time_range(
        exchange=exchange_id,
        symbol=symbol,
    )
    first_dt = _as_utc(first_dt)
    last_dt = _as_utc(last_dt)
    return {
        "first": _datetime_payload(first_dt),
        "last": _datetime_payload(last_dt),
    }


def main() -> int:
    now_dt = datetime.now(UTC)
    db_path = str(BaseConfig.MARKET_DATA_PARQUET_PATH)
    exchange_id = str(BaseConfig.MARKET_DATA_EXCHANGE).strip().lower() or "binance"
    repo = ParquetMarketDataRepository(db_path)
    requested_symbols = list(BaseConfig.SYMBOLS)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    exchange = create_binance_exchange(market_type="future", testnet=False)
    try:
        exchange.load_markets()

        supported_symbols = [symbol for symbol in requested_symbols if _supports_symbol(exchange, symbol)]
        skipped_symbols = [symbol for symbol in requested_symbols if symbol not in supported_symbols]

        summary: dict[str, object] = {
            "generated_at_utc": now_dt.isoformat(),
            "db_path": db_path,
            "exchange_id": exchange_id,
            "timeframe": TIMEFRAME,
            "start_utc": START_DT.isoformat(),
            "end_utc": now_dt.isoformat(),
            "requested_symbols": requested_symbols,
            "supported_symbols": supported_symbols,
            "skipped_symbols": skipped_symbols,
            "symbols": [],
        }
        REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[OHLCV-REFRESH] start={START_DT.isoformat()} end={now_dt.isoformat()} "
            f"supported={len(supported_symbols)} skipped={len(skipped_symbols)}"
        )
        if skipped_symbols:
            print(f"[OHLCV-REFRESH] skipped unsupported symbols: {', '.join(skipped_symbols)}", flush=True)

        for index, symbol in enumerate(supported_symbols, start=1):
            symbol_t0 = time.monotonic()
            normalized_symbol = normalize_symbol(symbol)
            before = _coverage_snapshot(repo, exchange_id=exchange_id, symbol=normalized_symbol)
            before_first_dt, before_last_dt = repo.get_symbol_time_range(
                exchange=exchange_id,
                symbol=normalized_symbol,
            )
            before_first_dt = _as_utc(before_first_dt)
            before_last_dt = _as_utc(before_last_dt)
            windows: list[tuple[datetime, datetime]] = []
            if before_first_dt is None or before_last_dt is None:
                windows.append((START_DT, now_dt))
            else:
                if before_first_dt > START_DT:
                    windows.append((START_DT, before_first_dt - timedelta(seconds=1)))
                tail_start = max(START_DT, before_last_dt + timedelta(seconds=1))
                if tail_start <= now_dt:
                    windows.append((tail_start, now_dt))

            sync_rows = []
            for window_start, window_end in windows:
                if window_start > window_end:
                    continue
                stats = sync_symbol_ohlcv(
                    exchange=exchange,
                    db_path=db_path,
                    exchange_id=exchange_id,
                    symbol=symbol,
                    timeframe=TIMEFRAME,
                    start_ms=int(window_start.timestamp() * 1000),
                    end_ms=int(window_end.timestamp() * 1000),
                    limit=1000,
                    max_batches=250000,
                    retries=2,
                    base_wait_sec=0.25,
                )
                sync_rows.append(asdict(stats))
            after = _coverage_snapshot(repo, exchange_id=exchange_id, symbol=normalized_symbol)
            payload = {
                "symbol": symbol,
                "normalized_symbol": normalized_symbol,
                "before": before,
                "sync": sync_rows,
                "windows": [
                    {
                        "start_utc": item[0].astimezone(UTC).isoformat(),
                        "end_utc": item[1].astimezone(UTC).isoformat(),
                    }
                    for item in windows
                ],
                "compaction": [],
                "after": after,
                "elapsed_seconds": round(time.monotonic() - symbol_t0, 2),
            }
            cast_summary = summary["symbols"]
            assert isinstance(cast_summary, list)
            cast_summary.append(payload)
            REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            sync_item = payload["sync"][0] if payload["sync"] else {}
            print(
                f"[OHLCV-REFRESH][{index}/{len(supported_symbols)}] {symbol} "
                f"fetched={sync_item.get('fetched_rows', 0)} "
                f"upserted={sync_item.get('upserted_rows', 0)} "
                f"last={after['last']['datetime_utc'] if after['last'] else None} "
                f"elapsed={payload['elapsed_seconds']}s"
            , flush=True)
    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    print(f"[OHLCV-REFRESH] summary written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
