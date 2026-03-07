from __future__ import annotations

import io
import json
import time
import urllib.error
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from lumina_quant.config import BaseConfig
from lumina_quant.data_sync import (
    _binance_archive_url,
    _date_from_ms,
    _day_bounds_ms,
    _download_zip_bytes,
    create_binance_exchange,
    sync_symbol_aggtrades_raw,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository, normalize_symbol

START_DAY = date(2025, 1, 1)
SYMBOL_START_DATES: dict[str, date] = {
    "XAU/USDT": date(2025, 12, 11),
    "XAG/USDT": date(2026, 1, 7),
    "XPT/USDT": date(2026, 1, 30),
    "XPD/USDT": date(2026, 1, 30),
}
NOW = datetime.now(UTC)
TODAY = NOW.date()
ARCHIVE_END_DAY = TODAY - timedelta(days=1)
TODAY_START = datetime(TODAY.year, TODAY.month, TODAY.day, tzinfo=UTC)
TODAY_START_MS = int(TODAY_START.timestamp() * 1000)
NOW_MS = int(NOW.timestamp() * 1000)
DB_PATH = str(BaseConfig.MARKET_DATA_PARQUET_PATH)
EXCHANGE_ID = str(BaseConfig.MARKET_DATA_EXCHANGE)
ALL_SYMBOLS = list(BaseConfig.SYMBOLS)
MARKET_TYPE = "future"

report_dir = Path("var/reports")
report_dir.mkdir(parents=True, exist_ok=True)
summary_path = report_dir / "raw_aggtrades_backfill_20250101_to_now_summary.json"


def _supports_symbol(exchange, symbol: str) -> bool:
    compact = normalize_symbol(symbol)
    try:
        exchange.market(compact)
        return True
    except Exception:
        return compact in getattr(exchange, "markets", {}) or f"{compact}:USDT" in getattr(
            exchange, "markets", {}
        )


def _maybe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _symbol_start_day(symbol: str) -> date:
    return SYMBOL_START_DATES.get(str(symbol), START_DAY)


def _resume_archive_day(checkpoint: dict[str, object] | None, *, symbol: str) -> date:
    minimum_day = _symbol_start_day(symbol)
    cp_ts = _maybe_int((checkpoint or {}).get("last_timestamp_ms"))
    if not cp_ts or cp_ts <= 0:
        return minimum_day
    cp_day = _date_from_ms(cp_ts)
    cp_day_end_ms = _day_bounds_ms(cp_day)[1]
    if cp_ts >= cp_day_end_ms:
        return max(minimum_day, cp_day + timedelta(days=1))
    return max(minimum_day, cp_day)


def _read_archive_df(blob: bytes) -> pl.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        if not names:
            return pl.DataFrame()
        with zf.open(names[0], "r") as raw_file:
            df = pl.read_csv(
                raw_file,
                has_header=True,
                schema_overrides={
                    "agg_trade_id": pl.Int64,
                    "price": pl.Float64,
                    "quantity": pl.Float64,
                    "first_trade_id": pl.Int64,
                    "last_trade_id": pl.Int64,
                    "transact_time": pl.Int64,
                    "timestamp": pl.Int64,
                    "is_buyer_maker": pl.Boolean,
                    "best_match": pl.Boolean,
                },
            )
    if "transact_time" in df.columns and "timestamp_ms" not in df.columns:
        df = df.rename({"transact_time": "timestamp_ms"})
    elif "timestamp" in df.columns and "timestamp_ms" not in df.columns:
        df = df.rename({"timestamp": "timestamp_ms"})
    required = {"agg_trade_id", "timestamp_ms", "price", "quantity", "is_buyer_maker"}
    if not required.issubset(df.columns):
        return pl.DataFrame()
    return df.select(["agg_trade_id", "timestamp_ms", "price", "quantity", "is_buyer_maker"]).sort([
        "timestamp_ms",
        "agg_trade_id",
    ])


def main() -> int:
    repo = ParquetMarketDataRepository(DB_PATH)
    exchange = create_binance_exchange(market_type=MARKET_TYPE)
    exchange.load_markets()

    supported_symbols = [symbol for symbol in ALL_SYMBOLS if _supports_symbol(exchange, symbol)]
    skipped_symbols = [symbol for symbol in ALL_SYMBOLS if symbol not in supported_symbols]

    summary: dict[str, object] = {
        "generated_at_utc": NOW.isoformat(),
        "db_path": DB_PATH,
        "exchange_id": EXCHANGE_ID,
        "market_type": MARKET_TYPE,
        "start_day": START_DAY.isoformat(),
        "archive_end_day": ARCHIVE_END_DAY.isoformat(),
        "today_start_utc": TODAY_START.isoformat(),
        "now_utc": NOW.isoformat(),
        "supported_symbols": supported_symbols,
        "skipped_symbols": skipped_symbols,
        "symbols": [],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[BACKFILL2] start={START_DAY} archive_end={ARCHIVE_END_DAY} "
        f"today_start={TODAY_START.isoformat()} supported={len(supported_symbols)} skipped={len(skipped_symbols)}",
        flush=True,
    )
    if skipped_symbols:
        print(f"[BACKFILL2] skipped unsupported symbols: {', '.join(skipped_symbols)}", flush=True)

    try:
        for index, symbol in enumerate(supported_symbols, start=1):
            symbol_t0 = time.monotonic()
            checkpoint_before = repo.read_raw_checkpoint(exchange=EXCHANGE_ID, symbol=symbol)
            archive_day = _resume_archive_day(checkpoint_before, symbol=symbol)
            archive_rows = 0
            archive_upserted_rows = 0
            archive_days_with_data = 0
            archive_days_missing = 0
            archive_days_skipped = 0
            last_archive_ts = _maybe_int(checkpoint_before.get("last_timestamp_ms")) if checkpoint_before else None
            last_archive_trade_id = _maybe_int(checkpoint_before.get("last_trade_id"), 0) if checkpoint_before else 0
            print(
                f"[BACKFILL2][{index}/{len(supported_symbols)}] {symbol} "
                f"checkpoint_before={checkpoint_before.get('last_timestamp_ms') if checkpoint_before else None} "
                f"archive_resume_day={archive_day}",
                flush=True,
            )

            if archive_day <= ARCHIVE_END_DAY:
                cursor_day = archive_day
                progress_hits = 0
                while cursor_day <= ARCHIVE_END_DAY:
                    url = _binance_archive_url(symbol, cursor_day, MARKET_TYPE)
                    blob = _download_zip_bytes(url, retries=2, base_wait_sec=0.5)
                    if blob is None:
                        archive_days_missing += 1
                        cursor_day += timedelta(days=1)
                        continue
                    payload = _read_archive_df(blob)
                    if payload.is_empty():
                        archive_days_skipped += 1
                        cursor_day += timedelta(days=1)
                        continue
                    upserted = int(repo.append_raw_aggtrades(exchange=EXCHANGE_ID, symbol=symbol, rows=payload))
                    archive_rows += int(payload.height)
                    archive_upserted_rows += int(upserted)
                    archive_days_with_data += 1
                    last_archive_ts = int(payload["timestamp_ms"][-1])
                    last_archive_trade_id = int(payload["agg_trade_id"][-1])
                    repo.write_raw_checkpoint(
                        exchange=EXCHANGE_ID,
                        symbol=symbol,
                        payload={
                            "symbol": symbol,
                            "exchange": EXCHANGE_ID,
                            "last_timestamp_ms": int(last_archive_ts),
                            "last_trade_id": int(last_archive_trade_id or 0),
                            "last_agg_trade_id": int(last_archive_trade_id or 0),
                            "updated_at_utc": datetime.now(UTC).isoformat(),
                            "source": "binance_archive_backfill",
                        },
                    )
                    progress_hits += 1
                    if progress_hits == 1 or progress_hits % 30 == 0 or cursor_day == ARCHIVE_END_DAY:
                        print(
                            f"[BACKFILL2][{symbol}] archive day={cursor_day.isoformat()} rows={payload.height} "
                            f"upserted={upserted} days_with_data={archive_days_with_data} total_rows={archive_rows}",
                            flush=True,
                        )
                    cursor_day += timedelta(days=1)
            else:
                print(
                    f"[BACKFILL2][{symbol}] archive already complete through {ARCHIVE_END_DAY}",
                    flush=True,
                )

            checkpoint_mid = repo.read_raw_checkpoint(exchange=EXCHANGE_ID, symbol=symbol)
            tail_start_ms = max(TODAY_START_MS, _maybe_int((checkpoint_mid or {}).get("last_timestamp_ms"), 0) + 1)
            tail_stats = None
            if tail_start_ms <= NOW_MS:
                tail_stats = sync_symbol_aggtrades_raw(
                    exchange=exchange,
                    db_path=DB_PATH,
                    exchange_id=EXCHANGE_ID,
                    symbol=symbol,
                    start_ms=tail_start_ms,
                    end_ms=NOW_MS,
                    limit=1000,
                    max_batches=250000,
                    retries=3,
                    base_wait_sec=0.35,
                    resume_from_checkpoint=True,
                )
            checkpoint_after = repo.read_raw_checkpoint(exchange=EXCHANGE_ID, symbol=symbol)
            symbol_summary = {
                "symbol": symbol,
                "checkpoint_before": checkpoint_before,
                "archive_resume_day": archive_day.isoformat(),
                "archive_days_with_data": archive_days_with_data,
                "archive_days_missing": archive_days_missing,
                "archive_days_skipped": archive_days_skipped,
                "archive_rows": int(archive_rows),
                "archive_upserted_rows": int(archive_upserted_rows),
                "tail_start_ms": int(tail_start_ms),
                "tail_start_utc": datetime.fromtimestamp(tail_start_ms / 1000, tz=UTC).isoformat(),
                "tail_fetched_rows": int(getattr(tail_stats, 'fetched_rows', 0) or 0),
                "tail_upserted_rows": int(getattr(tail_stats, 'upserted_rows', 0) or 0),
                "tail_last_timestamp_ms": getattr(tail_stats, 'last_timestamp_ms', None),
                "checkpoint_after": checkpoint_after,
                "elapsed_seconds": round(time.monotonic() - symbol_t0, 2),
            }
            summary["symbols"].append(symbol_summary)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[BACKFILL2][{symbol}] done archive_rows={archive_rows} "
                f"archive_upserted={archive_upserted_rows} tail_rows={symbol_summary['tail_fetched_rows']} "
                f"checkpoint={checkpoint_after.get('last_timestamp_ms') if checkpoint_after else None} "
                f"elapsed={symbol_summary['elapsed_seconds']}s",
                flush=True,
            )
    finally:
        try:
            exchange.close()
        except Exception:
            pass

    print(f"[BACKFILL2] summary written to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
