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
from lumina_quant.market_data import upsert_ohlcv_rows_1s
from lumina_quant.storage.parquet import ParquetMarketDataRepository

SYMBOL_START_DATES: dict[str, date] = {
    "XAU/USDT": date(2025, 12, 11),
    "XAG/USDT": date(2026, 1, 7),
    "XPT/USDT": date(2026, 1, 30),
    "XPD/USDT": date(2026, 1, 30),
}
REPORT_PATH = Path("var/reports/archive_first_tail_refresh_commodities_futures_summary.json")


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


def _existing_start_day(repo: ParquetMarketDataRepository, *, symbol: str, minimum_day: date) -> date:
    _first_dt, last_dt = repo.get_symbol_time_range(exchange=str(BaseConfig.MARKET_DATA_EXCHANGE), symbol=symbol)
    if last_dt is None:
        return minimum_day
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=UTC)
    else:
        last_dt = last_dt.astimezone(UTC)
    return max(minimum_day, last_dt.date())


def main() -> int:
    now_dt = datetime.now(UTC)
    today = now_dt.date()
    today_start = datetime(today.year, today.month, today.day, tzinfo=UTC)
    archive_end_day = today - timedelta(days=1)
    repo = ParquetMarketDataRepository(str(BaseConfig.MARKET_DATA_PARQUET_PATH))

    prior_report = _load_report()
    prior_rows = prior_report.get("symbols")
    prior_rows = prior_rows if isinstance(prior_rows, list) else []
    completed_symbols = {
        str(item.get("symbol"))
        for item in prior_rows
        if isinstance(item, dict) and str(item.get("status", "")) == "ok"
    }

    report = {
        "generated_at_utc": now_dt.isoformat(),
        "db_path": str(BaseConfig.MARKET_DATA_PARQUET_PATH),
        "exchange_id": str(BaseConfig.MARKET_DATA_EXCHANGE),
        "archive_end_day": archive_end_day.isoformat(),
        "today_start_utc": today_start.isoformat(),
        "end_utc": now_dt.isoformat(),
        "symbols": prior_rows,
    }
    _save_report(report)

    exchange = create_binance_exchange(market_type="future", testnet=False)
    try:
        exchange.load_markets()
        remaining = [symbol for symbol in SYMBOL_START_DATES if symbol not in completed_symbols]
        print(
            f"[COMMODITY-FUTURES] archive_end_day={archive_end_day} "
            f"today_start={today_start.isoformat()} remaining={len(remaining)}",
            flush=True,
        )

        for index, symbol in enumerate(remaining, start=1):
            base_start = SYMBOL_START_DATES[symbol]
            archive_start_day = _existing_start_day(repo, symbol=symbol, minimum_day=base_start)
            previous_close = None
            archive_days_loaded = 0
            archive_days_missing = 0
            archive_rows = 0
            archive_upserted_rows = 0

            cursor_day = archive_start_day
            while cursor_day <= archive_end_day:
                url = _binance_archive_url(symbol, cursor_day, "future")
                blob = _download_zip_bytes(url, retries=2, base_wait_sec=0.25)
                if blob is None:
                    archive_days_missing += 1
                    cursor_day += timedelta(days=1)
                    continue

                day_start = datetime(cursor_day.year, cursor_day.month, cursor_day.day, tzinfo=UTC)
                day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
                rows, previous_close = _archive_rows_to_1s_ohlcv(
                    blob,
                    market_type="future",
                    cursor_ms=int(day_start.timestamp() * 1000),
                    until_ms=int(day_end.timestamp() * 1000),
                    previous_close=previous_close,
                )
                if rows:
                    upserted = upsert_ohlcv_rows_1s(
                        str(BaseConfig.MARKET_DATA_PARQUET_PATH),
                        exchange=str(BaseConfig.MARKET_DATA_EXCHANGE),
                        symbol=symbol,
                        rows=rows,
                    )
                    archive_days_loaded += 1
                    archive_rows += int(len(rows))
                    archive_upserted_rows += int(upserted)
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
            _first_dt, verified_last = repo.get_symbol_time_range(
                exchange=str(BaseConfig.MARKET_DATA_EXCHANGE), symbol=symbol
            )
            if verified_last is not None:
                if verified_last.tzinfo is None:
                    verified_last = verified_last.replace(tzinfo=UTC)
                else:
                    verified_last = verified_last.astimezone(UTC)

            payload = {
                "symbol": symbol,
                "status": "ok",
                "start_day": base_start.isoformat(),
                "archive_start_day": archive_start_day.isoformat(),
                "archive_days_loaded": int(archive_days_loaded),
                "archive_days_missing": int(archive_days_missing),
                "archive_rows": int(archive_rows),
                "archive_upserted_rows": int(archive_upserted_rows),
                "today_fetched_rows": int(today_stats.fetched_rows),
                "today_upserted_rows": int(today_stats.upserted_rows),
                "today_last_utc": datetime.fromtimestamp(today_stats.last_timestamp_ms / 1000, tz=UTC).isoformat()
                if today_stats.last_timestamp_ms
                else None,
                "verified_last_utc": verified_last.isoformat() if verified_last is not None else None,
            }
            prior_rows.append(payload)
            _save_report(report)
            print(
                f"[COMMODITY-FUTURES][{index}/{len(remaining)}] {symbol} "
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

    print(f"[COMMODITY-FUTURES] summary written to {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
