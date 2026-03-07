from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from lumina_quant.config import BaseConfig
from lumina_quant.data_sync import (
    _archive_rows_to_1s_ohlcv,
    _binance_archive_url,
    _download_zip_bytes,
    create_binance_exchange,
    sync_symbol_ohlcv,
)
from lumina_quant.market_data import load_ohlcv_from_db, upsert_ohlcv_rows_1s
from lumina_quant.storage.parquet import normalize_symbol

START_DAY = date(2026, 3, 1)
REPORT_PATH = Path("var/reports/archive_first_tail_refresh_20260301_to_now_summary.json")
SYMBOL_PRIORITY = [
    "TRX/USDT",
    "TON/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "ETH/USDT",
    "BTC/USDT",
]


def _load_report() -> dict[str, object]:
    if REPORT_PATH.exists():
        try:
            payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _save_report(payload: dict[str, object]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _supports_symbol(exchange, symbol: str) -> bool:
    return normalize_symbol(symbol) in getattr(exchange, "markets", {})


def _previous_close(symbol: str, start_dt: datetime) -> float | None:
    window_start = start_dt - timedelta(minutes=5)
    try:
        frame = load_ohlcv_from_db(
            str(BaseConfig.MARKET_DATA_PARQUET_PATH),
            exchange=str(BaseConfig.MARKET_DATA_EXCHANGE),
            symbol=symbol,
            timeframe="1s",
            start_date=window_start,
            end_date=start_dt,
        )
    except Exception:
        return None
    if frame.is_empty():
        return None
    try:
        return float(frame["close"][-1])
    except Exception:
        return None


def _load_last_timestamp_today(symbol: str, today_start: datetime, now_dt: datetime) -> str | None:
    try:
        frame = load_ohlcv_from_db(
            str(BaseConfig.MARKET_DATA_PARQUET_PATH),
            exchange=str(BaseConfig.MARKET_DATA_EXCHANGE),
            symbol=symbol,
            timeframe="1s",
            start_date=today_start,
            end_date=now_dt,
        )
    except Exception:
        return None
    if frame.is_empty():
        return None
    try:
        value = frame["datetime"][-1]
    except Exception:
        return None
    if hasattr(value, "tzinfo") and value.tzinfo is not None:
        return value.astimezone(UTC).isoformat()
    return value.replace(tzinfo=UTC).isoformat()


def main() -> int:
    now_dt = datetime.now(UTC)
    today = now_dt.date()
    today_start = datetime(today.year, today.month, today.day, tzinfo=UTC)
    archive_end_day = today - timedelta(days=1)

    requested_symbols = list(BaseConfig.SYMBOLS)
    prior_report = _load_report()
    prior_rows = prior_report.get("symbols")
    prior_rows = prior_rows if isinstance(prior_rows, list) else []
    completed_symbols = {
        str(item.get("symbol"))
        for item in prior_rows
        if isinstance(item, dict) and str(item.get("status", "ok")) == "ok"
    }

    exchange = create_binance_exchange(market_type="future", testnet=False)
    try:
        exchange.load_markets()
        requested_order = [symbol for symbol in SYMBOL_PRIORITY if symbol in requested_symbols] + [
            symbol for symbol in requested_symbols if symbol not in SYMBOL_PRIORITY
        ]
        supported_symbols = [symbol for symbol in requested_order if _supports_symbol(exchange, symbol)]
        skipped_symbols = [symbol for symbol in requested_symbols if symbol not in supported_symbols]

        report = {
            "generated_at_utc": now_dt.isoformat(),
            "db_path": str(BaseConfig.MARKET_DATA_PARQUET_PATH),
            "exchange_id": str(BaseConfig.MARKET_DATA_EXCHANGE),
            "start_day": START_DAY.isoformat(),
            "archive_end_day": archive_end_day.isoformat(),
            "today_start_utc": today_start.isoformat(),
            "end_utc": now_dt.isoformat(),
            "supported_symbols": supported_symbols,
            "skipped_symbols": skipped_symbols,
            "symbols": prior_rows,
        }
        _save_report(report)

        remaining = [symbol for symbol in supported_symbols if symbol not in completed_symbols]
        print(
            f"[ARCHIVE-FIRST] start_day={START_DAY} archive_end_day={archive_end_day} "
            f"today_start={today_start.isoformat()} remaining={len(remaining)}",
            flush=True,
        )
        if skipped_symbols:
            print(f"[ARCHIVE-FIRST] skipped unsupported symbols: {', '.join(skipped_symbols)}", flush=True)

        for index, symbol in enumerate(remaining, start=1):
            normalized_symbol = normalize_symbol(symbol)
            prev_close = _previous_close(symbol, datetime.combine(START_DAY, datetime.min.time(), tzinfo=UTC))
            archive_days_loaded = 0
            archive_days_missing = 0
            archive_rows = 0
            archive_upserted_rows = 0

            cursor_day = START_DAY
            while cursor_day <= archive_end_day:
                url = _binance_archive_url(symbol, cursor_day, "future")
                blob = _download_zip_bytes(url, retries=2, base_wait_sec=0.25)
                if blob is None:
                    archive_days_missing += 1
                    cursor_day += timedelta(days=1)
                    continue

                day_start = datetime(cursor_day.year, cursor_day.month, cursor_day.day, tzinfo=UTC)
                day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
                rows, prev_close = _archive_rows_to_1s_ohlcv(
                    blob,
                    market_type="future",
                    cursor_ms=int(day_start.timestamp() * 1000),
                    until_ms=int(day_end.timestamp() * 1000),
                    previous_close=prev_close,
                )
                if rows:
                    upserted = upsert_ohlcv_rows_1s(
                        str(BaseConfig.MARKET_DATA_PARQUET_PATH),
                        exchange=str(BaseConfig.MARKET_DATA_EXCHANGE),
                        symbol=normalized_symbol,
                        rows=rows,
                    )
                    archive_rows += int(len(rows))
                    archive_upserted_rows += int(upserted)
                    archive_days_loaded += 1
                cursor_day += timedelta(days=1)

            today_stats = sync_symbol_ohlcv(
                exchange=exchange,
                db_path=str(BaseConfig.MARKET_DATA_PARQUET_PATH),
                exchange_id=str(BaseConfig.MARKET_DATA_EXCHANGE),
                symbol=symbol,
                timeframe="1s",
                start_ms=int(today_start.timestamp() * 1000),
                end_ms=int(now_dt.timestamp() * 1000),
                limit=1000,
                max_batches=250000,
                retries=2,
                base_wait_sec=0.25,
            )

            payload = {
                "symbol": symbol,
                "status": "ok",
                "archive_days_loaded": int(archive_days_loaded),
                "archive_days_missing": int(archive_days_missing),
                "archive_rows": int(archive_rows),
                "archive_upserted_rows": int(archive_upserted_rows),
                "today_fetched_rows": int(today_stats.fetched_rows),
                "today_upserted_rows": int(today_stats.upserted_rows),
                "today_last_utc": datetime.fromtimestamp(today_stats.last_timestamp_ms / 1000, tz=UTC).isoformat()
                if today_stats.last_timestamp_ms
                else None,
                "verified_last_utc": _load_last_timestamp_today(symbol, today_start, now_dt),
            }
            prior_rows.append(payload)
            _save_report(report)
            print(
                f"[ARCHIVE-FIRST][{index}/{len(remaining)}] {symbol} "
                f"archive_upserted={payload['archive_upserted_rows']} "
                f"today_upserted={payload['today_upserted_rows']} "
                f"verified_last={payload['verified_last_utc']}",
                flush=True,
            )

    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    print(f"[ARCHIVE-FIRST] summary written to {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
