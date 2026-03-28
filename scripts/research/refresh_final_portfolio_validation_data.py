from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
import json
import os
import resource
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.data.native_raw_first_backend import RAW_FIRST_BACKEND_ENV
from lumina_quant.data.support_inventory import (
    INVENTORY_NOTES,
    build_strategy_support_inventory,
    write_strategy_support_inventory,
)
from lumina_quant.data.raw_first_lineage import (
    raw_aggtrades_to_1s_frame,
    resolve_raw_aggtrades_backend_name,
)
from lumina_quant.data_sync import (
    _binance_archive_url,
    _day_bounds_ms,
    _download_zip_bytes,
    _iter_days,
    create_binance_futures_client,
    fetch_aggtrades_batch,
    sync_futures_feature_points,
)
from lumina_quant.eval.exact_window_runtime import RSSGuard
from lumina_quant.market_data import upsert_ohlcv_rows_1s
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    PORTFOLIO_CURRENT_OPTIMIZATION,
    portfolio_followup_default_budget_bytes,
    resolve_incumbent_bundle_path,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository, normalize_symbol

FEATURE_REQUIRED_STRATEGIES = {"CompositeTrendStrategy", "PerpCrowdingCarryStrategy"}
DEFAULT_DB_PATH = "data/market_parquet"
DEFAULT_EXCHANGE_ID = "binance"
DEFAULT_OUTPUT_JSON = FOLLOWUP_ROOT / "final_portfolio_validation_data_refresh_latest.json"
DEFAULT_OUTPUT_MD = FOLLOWUP_ROOT / "final_portfolio_validation_data_refresh_latest.md"
DEFAULT_RSS_LOG = FOLLOWUP_ROOT / "final_portfolio_validation_data_refresh_rss_latest.jsonl"
DEFAULT_SUPPORT_INVENTORY_JSON = FOLLOWUP_ROOT / "final_portfolio_validation_support_inventory_latest.json"
DEFAULT_SUPPORT_INVENTORY_CSV = FOLLOWUP_ROOT / "final_portfolio_validation_support_inventory_latest.csv"
DEFAULT_MIN_START_UTC = "2025-01-01T00:00:00Z"
DEFAULT_SOFT_RSS_BYTES = int(7.2 * 1024 * 1024 * 1024)
DEFAULT_PARALLEL_WORKER_MEMORY_BYTES = int(1.5 * 1024 * 1024 * 1024)
DEFAULT_PARALLEL_MEMORY_RESERVE_BYTES = int(2.0 * 1024 * 1024 * 1024)


@dataclass(slots=True)
class OhlcvRefreshResult:
    symbol: str
    before_ohlcv_max_utc: str | None
    after_ohlcv_max_utc: str | None
    before_raw_agg_trade_utc: str | None
    after_raw_agg_trade_utc: str | None
    resume_start_utc: str | None
    cutoff_utc: str
    archive_days_considered: int
    archive_days_downloaded: int
    archive_raw_rows_fetched: int
    archive_raw_rows_upserted: int
    live_raw_rows_fetched: int
    live_raw_rows_upserted: int
    derived_ohlcv_rows_upserted: int


@dataclass(slots=True)
class FeatureRefreshResult:
    symbol: str
    before_feature_day_utc: str | None
    after_feature_day_utc: str | None
    resume_start_utc: str
    cutoff_utc: str
    upserted_rows: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None


@dataclass(slots=True)
class OhlcvRefreshWorkerPayload:
    result: OhlcvRefreshResult
    peak_rss_bytes: int


def parse_utc(value: str | None) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat().replace("+00:00", "Z")


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def latest_runtime_tail_utc(now: datetime | None = None) -> datetime:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    return current.replace(microsecond=0)


def _day_start(day_value: date) -> datetime:
    return datetime(day_value.year, day_value.month, day_value.day, tzinfo=UTC)


def _latest_feature_partition_day(symbol: str, *, db_path: str, exchange_id: str) -> date | None:
    compact = normalize_symbol(symbol).replace("/", "")
    root = Path(db_path) / "feature_points" / f"exchange={str(exchange_id).lower()}" / f"symbol={compact}"
    latest: date | None = None
    for path in root.glob("date=*"):
        if not path.is_dir() or "=" not in path.name:
            continue
        try:
            day_value = date.fromisoformat(path.name.split("=", 1)[1])
        except ValueError:
            continue
        latest = day_value if latest is None else max(latest, day_value)
    return latest


def feature_resume_start(symbol: str, *, db_path: str, exchange_id: str, floor_dt: datetime) -> datetime:
    latest_day = _latest_feature_partition_day(symbol, db_path=db_path, exchange_id=exchange_id)
    if latest_day is None:
        return floor_dt
    return max(floor_dt, _day_start(latest_day))


def load_portfolio_symbols(portfolio_path: Path | str = PORTFOLIO_CURRENT_OPTIMIZATION) -> list[str]:
    payload = json.loads(Path(portfolio_path).read_text(encoding="utf-8"))
    ordered: dict[str, None] = {}
    for row in list(payload.get("weights") or []):
        for symbol in list(row.get("symbols") or []):
            token = normalize_symbol(str(symbol))
            if token:
                ordered[token] = None
    return list(ordered)


def load_feature_symbols(bundle_path: Path | str | None = None) -> list[str]:
    resolved = resolve_incumbent_bundle_path(bundle_path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    ordered: dict[str, None] = {}
    for row in list(payload.get("selected_team") or payload.get("candidates") or []):
        strategy_class = str(row.get("strategy_class") or row.get("strategy") or "")
        if strategy_class not in FEATURE_REQUIRED_STRATEGIES:
            continue
        for symbol in list(row.get("symbols") or []):
            token = normalize_symbol(str(symbol))
            if token:
                ordered[token] = None
    return list(ordered)


def parse_symbol_tokens(value: str | None, *, default: list[str] | None = None) -> list[str]:
    ordered: dict[str, None] = {}
    for symbol in list(default or []):
        token = normalize_symbol(str(symbol))
        if token:
            ordered[token] = None
    raw = str(value or "").strip()
    if not raw:
        return list(ordered)
    for item in raw.split(","):
        token = normalize_symbol(str(item))
        if token:
            ordered[token] = None
    return list(ordered)


def prioritize_symbols(symbols: list[str], *, priority_symbols: list[str] | None = None) -> list[str]:
    ordered: dict[str, None] = {}
    priorities = [normalize_symbol(symbol) for symbol in list(priority_symbols or []) if str(symbol).strip()]
    symbol_set = {normalize_symbol(symbol) for symbol in list(symbols or [])}
    for symbol in priorities:
        if symbol in symbol_set:
            ordered[symbol] = None
    for symbol in list(symbols or []):
        token = normalize_symbol(symbol)
        if token:
            ordered[token] = None
    return list(ordered)


def estimate_parallel_workers(
    *,
    symbol_count: int,
    memory_budget_bytes: int,
    reserve_memory_bytes: int,
    per_worker_memory_bytes: int,
    max_workers: int,
) -> int:
    if symbol_count <= 0:
        return 1
    budget = max(1, int(memory_budget_bytes))
    reserve = max(0, int(reserve_memory_bytes))
    per_worker = max(1, int(per_worker_memory_bytes))
    available = max(per_worker, budget - reserve)
    budget_limited = max(1, available // per_worker)
    cpu_limited = max(1, int(max_workers))
    return max(1, min(int(symbol_count), int(budget_limited), int(cpu_limited)))


def _process_peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(getattr(usage, "ru_maxrss", 0) or 0)
    if sys.platform == "darwin":  # pragma: no cover - macOS compatibility
        return peak
    return peak * 1024


def _last_close(repo: ParquetMarketDataRepository, *, symbol: str, max_dt: datetime | None) -> float | None:
    if max_dt is None:
        return None
    frame = repo.load_ohlcv(
        exchange=DEFAULT_EXCHANGE_ID,
        symbol=symbol,
        timeframe="1s",
        start_date=max_dt,
        end_date=max_dt,
    )
    if frame.is_empty():
        return None
    try:
        return float(frame.get_column("close")[-1])
    except Exception:
        return None


def _raw_checkpoint_utc(
    repo: ParquetMarketDataRepository,
    *,
    db_path: str,
    exchange_id: str,
    symbol: str,
) -> datetime | None:
    latest: datetime | None = None
    checkpoint = repo.read_raw_checkpoint(exchange=exchange_id, symbol=symbol)
    try:
        ts_ms = int(checkpoint.get("last_timestamp_ms", 0) or 0)
    except Exception:
        ts_ms = 0
    if ts_ms > 0:
        latest = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)

    compact = normalize_symbol(symbol).replace("/", "")
    root = (
        Path(db_path)
        / "market_data_raw_aggtrades"
        / str(exchange_id).lower()
        / compact
    )
    for path in sorted(root.glob("date=*/part-0000.parquet"), reverse=True):
        try:
            frame = pl.scan_parquet(str(path)).select(pl.col("timestamp_ms").max().alias("ts")).collect()
            file_ts = int(frame["ts"][0] or 0)
        except BaseException:
            continue
        if file_ts <= 0:
            continue
        dt = datetime.fromtimestamp(file_ts / 1000.0, tz=UTC)
        latest = dt if latest is None else max(latest, dt)
        break
    return latest


def _archive_rows_to_raw_aggtrades(
    zip_blob: bytes,
    *,
    cursor_ms: int,
    until_ms: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(zip_blob)) as zf:
        names = zf.namelist()
        if not names:
            return rows
        with zf.open(names[0], "r") as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8")
            reader = csv.reader(text_file)
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    agg_trade_id = int(row[0])
                    price = float(row[1])
                    quantity = float(row[2])
                    timestamp_ms = int(row[5])
                except Exception:
                    continue
                if price <= 0.0 or quantity < 0.0:
                    continue
                if timestamp_ms < int(cursor_ms) or timestamp_ms > int(until_ms):
                    continue
                rows.append(
                    {
                        "agg_trade_id": agg_trade_id,
                        "timestamp_ms": timestamp_ms,
                        "price": price,
                        "quantity": max(0.0, quantity),
                        "is_buyer_maker": str(row[6]).strip().lower() in {"1", "true", "t", "yes"},
                    }
                )
    rows.sort(key=lambda item: (int(item["timestamp_ms"]), int(item["agg_trade_id"])))
    return rows


def _write_raw_checkpoint_for_rows(
    repo: ParquetMarketDataRepository,
    *,
    exchange_id: str,
    symbol: str,
    rows: list[dict[str, Any]],
    source: str,
    observed_until_ms: int,
) -> None:
    if not rows:
        return
    last = dict(rows[-1])
    repo.write_raw_checkpoint(
        exchange=exchange_id,
        symbol=symbol,
        payload={
            "symbol": symbol,
            "exchange": str(exchange_id).lower(),
            "last_timestamp_ms": int(last["timestamp_ms"]),
            "last_trade_id": int(last["agg_trade_id"]),
            "last_agg_trade_id": int(last["agg_trade_id"]),
            "observed_until_ms": int(observed_until_ms),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "source": str(source),
        },
    )
    repo.append_raw_wal_record(
        exchange=exchange_id,
        symbol=symbol,
        payload={
            "type": "aggtrades_raw_batch",
            "source": str(source),
            "last_timestamp_ms": int(last["timestamp_ms"]),
            "last_trade_id": int(last["agg_trade_id"]),
            "observed_until_ms": int(observed_until_ms),
            "rows": len(rows),
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _collect_live_raw_rows(
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 10,
    base_wait_sec: float = 2.0,
    pause_sec: float = 0.1,
) -> list[dict[str, Any]]:
    exchange = create_binance_futures_client(market_type="future")
    cursor = max(0, int(start_ms))
    until = max(cursor, int(end_ms))
    current_limit = max(1, int(limit))
    last_trade_id = -1
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []

    try:
        batch_count = 0
        while cursor <= until and batch_count < max(1, int(max_batches)):
            batch_count += 1
            try:
                batch = fetch_aggtrades_batch(
                    exchange=exchange,
                    symbol=normalize_symbol(symbol),
                    since_ms=int(cursor),
                    limit=int(current_limit),
                    retries=max(0, int(retries)),
                    base_wait_sec=float(base_wait_sec),
                )
            except Exception as exc:
                message = str(exc)
                if "429" in message or "Too Many Requests" in message or "DDoSProtection" in message:
                    current_limit = max(100, current_limit // 2)
                    time.sleep(max(float(base_wait_sec), 5.0))
                    continue
                raise
            if not batch:
                break

            filtered: list[dict[str, Any]] = []
            max_seen_ts = int(cursor)
            max_seen_id = int(last_trade_id)
            for item in batch:
                ts = int(item["timestamp_ms"])
                trade_id = int(item["agg_trade_id"])
                if ts < int(cursor) or ts > int(until):
                    continue
                if ts == int(cursor) and trade_id <= int(last_trade_id):
                    continue
                key = (ts, trade_id)
                if key in seen:
                    continue
                seen.add(key)
                filtered.append(dict(item))
                if ts > max_seen_ts or (ts == max_seen_ts and trade_id > max_seen_id):
                    max_seen_ts = ts
                    max_seen_id = trade_id

            if not filtered:
                cursor = max(int(cursor) + 1, max_seen_ts + 1)
                last_trade_id = int(max_seen_id)
                continue

            filtered.sort(key=lambda item: (int(item["timestamp_ms"]), int(item["agg_trade_id"])))
            out.extend(filtered)
            last_trade_id = int(filtered[-1]["agg_trade_id"])
            cursor = int(filtered[-1]["timestamp_ms"]) + 1
            if len(filtered) < int(current_limit):
                break
            if float(pause_sec) > 0.0:
                time.sleep(float(pause_sec))
    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            close_fn()

    return out


def refresh_symbol_raw_first_ohlcv(
    *,
    repo: ParquetMarketDataRepository,
    symbol: str,
    db_path: str,
    exchange_id: str,
    cutoff_dt: datetime,
    floor_dt: datetime,
    guard: RSSGuard | None = None,
) -> OhlcvRefreshResult:
    _, before_max = repo.get_symbol_time_range(exchange=exchange_id, symbol=symbol)
    before_max_utc = None
    if before_max is not None:
        before_max_utc = before_max.replace(tzinfo=UTC) if before_max.tzinfo is None else before_max.astimezone(UTC)
    before_raw_dt = _raw_checkpoint_utc(
        repo,
        db_path=db_path,
        exchange_id=exchange_id,
        symbol=symbol,
    )
    resume_dt = max(floor_dt, before_max_utc if before_max_utc is not None else floor_dt)
    cursor_ms = int(resume_dt.timestamp() * 1000) + (1000 if before_max is not None else 0)
    cutoff_ms = int(cutoff_dt.timestamp() * 1000)
    archive_days_considered = 0
    archive_days_downloaded = 0
    archive_raw_rows_fetched = 0
    archive_raw_rows_upserted = 0
    live_raw_rows_fetched = 0
    live_raw_rows_upserted = 0
    derived_ohlcv_rows_upserted = 0
    after_max_utc = before_max_utc
    previous_close = _last_close(repo, symbol=symbol, max_dt=before_max_utc)
    live_start_ms = cursor_ms
    if before_raw_dt is not None:
        live_start_ms = max(live_start_ms, int(before_raw_dt.timestamp() * 1000) + 1)

    if cursor_ms <= cutoff_ms:
        for day_value in _iter_days(cursor_ms, cutoff_ms):
            archive_days_considered += 1
            day_start_ms, day_end_ms = _day_bounds_ms(day_value)
            range_start = max(cursor_ms, day_start_ms)
            range_end = min(cutoff_ms, day_end_ms)
            if range_start > range_end:
                continue
            blob = _download_zip_bytes(
                _binance_archive_url(symbol, day_value, "future"),
                retries=3,
                base_wait_sec=1.0,
            )
            if blob is None:
                continue
            raw_rows = _archive_rows_to_raw_aggtrades(
                blob,
                cursor_ms=range_start,
                until_ms=range_end,
            )
            if not raw_rows:
                continue
            archive_days_downloaded += 1
            archive_raw_rows_fetched += len(raw_rows)
            archive_raw_rows_upserted += repo.append_raw_aggtrades(
                exchange=str(exchange_id).lower(),
                symbol=symbol,
                rows=raw_rows,
            )
            _write_raw_checkpoint_for_rows(
                repo,
                exchange_id=str(exchange_id).lower(),
                symbol=symbol,
                rows=raw_rows,
                source="binance_archive_backfill",
                observed_until_ms=int(range_end),
            )
            derived_frame = raw_aggtrades_to_1s_frame(
                raw_rows,
                source=f"{exchange_id}:{symbol}:archive",
                range_start_ms=int(range_start),
                range_end_ms=int(range_end),
                previous_close=previous_close,
                complete_through_ms=int(range_end),
            )
            if not derived_frame.is_empty():
                derived_ohlcv_rows_upserted += upsert_ohlcv_rows_1s(
                    db_path,
                    exchange=str(exchange_id).lower(),
                    symbol=symbol,
                    rows=derived_frame,
                )
                after_max_utc = _as_utc(derived_frame.get_column("datetime")[-1])
                previous_close = float(derived_frame.get_column("close")[-1])
            live_start_ms = max(live_start_ms, int(raw_rows[-1]["timestamp_ms"]) + 1)
            if guard is not None:
                guard.checkpoint(
                    "after_archive_day",
                    {
                        "symbol": symbol,
                        "day": day_value.isoformat(),
                        "archive_raw_rows_upserted": archive_raw_rows_upserted,
                        "derived_ohlcv_rows_upserted": derived_ohlcv_rows_upserted,
                    },
                )

    if live_start_ms <= cutoff_ms:
        live_rows = _collect_live_raw_rows(
            symbol=symbol,
            start_ms=live_start_ms,
            end_ms=cutoff_ms,
            limit=1000,
            max_batches=100_000,
            retries=10,
            base_wait_sec=2.0,
        )
        live_raw_rows_fetched = len(live_rows)
        if live_rows:
            live_raw_rows_upserted = repo.append_raw_aggtrades(
                exchange=str(exchange_id).lower(),
                symbol=symbol,
                rows=live_rows,
            )
            _write_raw_checkpoint_for_rows(
                repo,
                exchange_id=str(exchange_id).lower(),
                symbol=symbol,
                rows=live_rows,
                source="binance_futures_live_tail",
                observed_until_ms=int(cutoff_ms),
            )
            derived_frame = raw_aggtrades_to_1s_frame(
                live_rows,
                source=f"{exchange_id}:{symbol}:rest_tail",
                range_start_ms=int(live_start_ms),
                range_end_ms=int(cutoff_ms),
                previous_close=previous_close,
                complete_through_ms=int(cutoff_ms),
            )
            if not derived_frame.is_empty():
                derived_ohlcv_rows_upserted += upsert_ohlcv_rows_1s(
                    db_path,
                    exchange=str(exchange_id).lower(),
                    symbol=symbol,
                    rows=derived_frame,
                )
                after_max_utc = _as_utc(derived_frame.get_column("datetime")[-1])
                previous_close = float(derived_frame.get_column("close")[-1])
            if guard is not None:
                guard.checkpoint(
                    "after_live_tail",
                    {
                        "symbol": symbol,
                        "live_raw_rows_upserted": live_raw_rows_upserted,
                        "derived_ohlcv_rows_upserted": derived_ohlcv_rows_upserted,
                    },
                )

    refreshed_repo = ParquetMarketDataRepository(str(db_path))
    after_raw_dt = _raw_checkpoint_utc(
        refreshed_repo,
        db_path=db_path,
        exchange_id=str(exchange_id).lower(),
        symbol=symbol,
    )
    return OhlcvRefreshResult(
        symbol=symbol,
        before_ohlcv_max_utc=iso_utc(before_max_utc),
        after_ohlcv_max_utc=iso_utc(after_max_utc),
        before_raw_agg_trade_utc=iso_utc(before_raw_dt),
        after_raw_agg_trade_utc=iso_utc(after_raw_dt),
        resume_start_utc=iso_utc(resume_dt),
        cutoff_utc=iso_utc(cutoff_dt) or "",
        archive_days_considered=archive_days_considered,
        archive_days_downloaded=archive_days_downloaded,
        archive_raw_rows_fetched=archive_raw_rows_fetched,
        archive_raw_rows_upserted=archive_raw_rows_upserted,
        live_raw_rows_fetched=live_raw_rows_fetched,
        live_raw_rows_upserted=live_raw_rows_upserted,
        derived_ohlcv_rows_upserted=derived_ohlcv_rows_upserted,
    )


def refresh_symbol_raw_first_ohlcv_worker(
    *,
    symbol: str,
    db_path: str,
    exchange_id: str,
    cutoff_utc: str,
    floor_utc: str,
) -> OhlcvRefreshWorkerPayload:
    repo = ParquetMarketDataRepository(str(db_path))
    cutoff_dt = parse_utc(cutoff_utc)
    floor_dt = parse_utc(floor_utc)
    if cutoff_dt is None or floor_dt is None:
        raise ValueError("Worker refresh requires concrete UTC cutoff/floor inputs")
    result = refresh_symbol_raw_first_ohlcv(
        repo=repo,
        symbol=symbol,
        db_path=db_path,
        exchange_id=exchange_id,
        cutoff_dt=cutoff_dt,
        floor_dt=floor_dt,
        guard=None,
    )
    return OhlcvRefreshWorkerPayload(
        result=result,
        peak_rss_bytes=_process_peak_rss_bytes(),
    )


def refresh_ohlcv_symbols(
    *,
    symbols: list[str],
    db_path: str,
    exchange_id: str,
    cutoff_dt: datetime,
    floor_dt: datetime,
    guard: RSSGuard,
    max_workers: int = 1,
    memory_budget_bytes: int,
    reserve_memory_bytes: int,
    per_worker_memory_bytes: int,
) -> tuple[list[OhlcvRefreshResult], dict[str, Any]]:
    ordered_symbols = [normalize_symbol(symbol) for symbol in list(symbols or []) if str(symbol).strip()]
    if not ordered_symbols:
        return [], {"requested_workers": 0, "selected_workers": 0, "mode": "empty"}

    requested_workers = max(1, int(max_workers))
    selected_workers = estimate_parallel_workers(
        symbol_count=len(ordered_symbols),
        memory_budget_bytes=int(memory_budget_bytes),
        reserve_memory_bytes=int(reserve_memory_bytes),
        per_worker_memory_bytes=int(per_worker_memory_bytes),
        max_workers=requested_workers,
    )

    worker_meta: dict[str, Any] = {
        "requested_workers": requested_workers,
        "selected_workers": int(selected_workers),
        "mode": "parallel" if int(selected_workers) > 1 else "sequential",
        "memory_budget_bytes": int(memory_budget_bytes),
        "reserve_memory_bytes": int(reserve_memory_bytes),
        "per_worker_memory_bytes": int(per_worker_memory_bytes),
    }

    ordered_results: dict[str, OhlcvRefreshResult] = {}
    peak_worker_rss_bytes = 0

    if int(selected_workers) <= 1:
        repo = ParquetMarketDataRepository(str(db_path))
        for symbol in ordered_symbols:
            result = refresh_symbol_raw_first_ohlcv(
                repo=repo,
                symbol=symbol,
                db_path=db_path,
                exchange_id=exchange_id,
                cutoff_dt=cutoff_dt,
                floor_dt=floor_dt,
                guard=guard,
            )
            ordered_results[symbol] = result
        worker_meta["peak_worker_rss_bytes"] = 0
        return [ordered_results[symbol] for symbol in ordered_symbols], worker_meta

    cutoff_utc = iso_utc(cutoff_dt) or ""
    floor_utc = iso_utc(floor_dt) or ""
    with ProcessPoolExecutor(max_workers=int(selected_workers)) as executor:
        future_to_symbol = {
            executor.submit(
                refresh_symbol_raw_first_ohlcv_worker,
                symbol=symbol,
                db_path=db_path,
                exchange_id=exchange_id,
                cutoff_utc=cutoff_utc,
                floor_utc=floor_utc,
            ): symbol
            for symbol in ordered_symbols
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            payload = future.result()
            ordered_results[symbol] = payload.result
            peak_worker_rss_bytes = max(int(peak_worker_rss_bytes), int(payload.peak_rss_bytes))
            guard.checkpoint(
                "after_parallel_symbol",
                {
                    "symbol": symbol,
                    "archive_raw_rows_upserted": payload.result.archive_raw_rows_upserted,
                    "live_raw_rows_upserted": payload.result.live_raw_rows_upserted,
                    "derived_ohlcv_rows_upserted": payload.result.derived_ohlcv_rows_upserted,
                    "worker_peak_rss_bytes": int(payload.peak_rss_bytes),
                },
            )

    worker_meta["peak_worker_rss_bytes"] = int(peak_worker_rss_bytes)
    return [ordered_results[symbol] for symbol in ordered_symbols], worker_meta


def refresh_feature_tail(
    *,
    symbol: str,
    db_path: str,
    exchange_id: str,
    cutoff_dt: datetime,
    floor_dt: datetime,
) -> FeatureRefreshResult:
    latest_before = _latest_feature_partition_day(symbol, db_path=db_path, exchange_id=exchange_id)
    resume_dt = feature_resume_start(symbol, db_path=db_path, exchange_id=exchange_id, floor_dt=floor_dt)
    summaries = sync_futures_feature_points(
        db_path=db_path,
        exchange_id=exchange_id,
        symbol_list=[symbol],
        since_ms=int(resume_dt.timestamp() * 1000),
        until_ms=int(cutoff_dt.timestamp() * 1000),
        retries=5,
        base_wait_sec=1.0,
    )
    latest_after = _latest_feature_partition_day(symbol, db_path=db_path, exchange_id=exchange_id)
    summary = summaries[0] if summaries else None
    first_ts = getattr(summary, "first_timestamp_ms", None)
    last_ts = getattr(summary, "last_timestamp_ms", None)
    return FeatureRefreshResult(
        symbol=symbol,
        before_feature_day_utc=latest_before.isoformat() if latest_before else None,
        after_feature_day_utc=latest_after.isoformat() if latest_after else None,
        resume_start_utc=iso_utc(resume_dt) or "",
        cutoff_utc=iso_utc(cutoff_dt) or "",
        upserted_rows=int(getattr(summary, "upserted_rows", 0) or 0),
        first_timestamp_utc=iso_utc(datetime.fromtimestamp(first_ts / 1000.0, tz=UTC) if first_ts else None),
        last_timestamp_utc=iso_utc(datetime.fromtimestamp(last_ts / 1000.0, tz=UTC) if last_ts else None),
    )


def _empty_inventory_payload(*, db_path: str, exchange_id: str) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "exchange_id": str(exchange_id),
        "db_path": str(db_path),
        "symbol_count": 0,
        "symbols": [],
        "notes": dict(INVENTORY_NOTES),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Final Portfolio Validation Data Refresh",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- status: `{payload.get('status')}`",
        f"- collection_cutoff_utc: `{payload.get('collection_cutoff_utc')}`",
        f"- canonical_source: `{payload.get('canonical_source')}`",
        f"- ohlcv_derivation: `{payload.get('ohlcv_derivation')}`",
        f"- ohlcv_symbols: `{', '.join(payload.get('portfolio_symbols') or [])}`",
        f"- feature_symbols: `{', '.join(payload.get('feature_symbols') or [])}`",
        "",
        "## OHLCV refresh",
        "",
        "| Symbol | Before max | After max | Before raw | After raw | Archive days | Raw rows | Live rows | Derived 1s upserted |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in list(payload.get("ohlcv_results") or []):
        lines.append(
            f"| `{row['symbol']}` | `{row.get('before_ohlcv_max_utc')}` | `{row.get('after_ohlcv_max_utc')}` | `{row.get('before_raw_agg_trade_utc')}` | `{row.get('after_raw_agg_trade_utc')}` | {row.get('archive_days_downloaded', 0)} | {int(row.get('archive_raw_rows_upserted', 0) or 0)} | {int(row.get('live_raw_rows_upserted', 0) or 0)} | {int(row.get('derived_ohlcv_rows_upserted', 0) or 0)} |"
        )
    lines.extend(
        [
            "",
            "## Feature refresh",
            "",
            "| Symbol | Before day | After day | Upserted rows | Last timestamp |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in list(payload.get("feature_results") or []):
        lines.append(
            f"| `{row['symbol']}` | `{row.get('before_feature_day_utc')}` | `{row.get('after_feature_day_utc')}` | {row.get('upserted_rows', 0)} | `{row.get('last_timestamp_utc')}` |"
        )
    memory = dict(payload.get("memory") or {})
    lines.extend(
        [
            "",
            "## Memory",
            "",
            f"- peak_rss_mib: `{memory.get('peak_rss_mib')}`",
            f"- soft_limit_mib: `{memory.get('soft_limit_mib')}`",
            f"- hard_limit_mib: `{memory.get('hard_limit_mib')}`",
            f"- soft_trigger_count: `{memory.get('soft_trigger_count')}`",
            f"- hard_trigger_count: `{memory.get('hard_trigger_count')}`",
        ]
    )
    if payload.get("support_inventory"):
        lines.extend(
            [
                "",
                "## Support inventory",
                "",
                f"- json_path: `{payload['support_inventory'].get('json_path')}`",
                f"- csv_path: `{payload['support_inventory'].get('csv_path')}`",
                f"- symbol_count: `{payload['support_inventory'].get('symbol_count')}`",
            ]
        )
    if payload.get("error"):
        lines.extend(["", "## Error", "", f"- `{payload['error']}`"])
    return "\\n".join(lines) + "\\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh OHLCV/support data for strict final-portfolio validation.",
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--exchange-id", default=DEFAULT_EXCHANGE_ID)
    parser.add_argument("--portfolio-path", default=str(PORTFOLIO_CURRENT_OPTIMIZATION))
    parser.add_argument("--symbols", default="")
    parser.add_argument("--priority-symbols", default="")
    parser.add_argument("--bundle-path", default="")
    parser.add_argument("--cutoff-utc", default="")
    parser.add_argument("--min-start-utc", default=DEFAULT_MIN_START_UTC)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--rss-log", default=str(DEFAULT_RSS_LOG))
    parser.add_argument("--support-inventory-json", default=str(DEFAULT_SUPPORT_INVENTORY_JSON))
    parser.add_argument("--support-inventory-csv", default=str(DEFAULT_SUPPORT_INVENTORY_CSV))
    parser.add_argument(
        "--memory-budget-bytes",
        type=int,
        default=portfolio_followup_default_budget_bytes(),
    )
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--parallel-reserve-bytes",
        type=int,
        default=DEFAULT_PARALLEL_MEMORY_RESERVE_BYTES,
    )
    parser.add_argument(
        "--parallel-per-worker-bytes",
        type=int,
        default=DEFAULT_PARALLEL_WORKER_MEMORY_BYTES,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cutoff_dt = parse_utc(args.cutoff_utc) or latest_runtime_tail_utc()
    floor_dt = parse_utc(args.min_start_utc)
    if floor_dt is None:
        raise ValueError("--min-start-utc must resolve to a UTC timestamp")

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    bundle_override = Path(args.bundle_path) if str(args.bundle_path).strip() else None
    portfolio_symbols = (
        parse_symbol_tokens(args.symbols)
        if str(args.symbols or "").strip()
        else load_portfolio_symbols(args.portfolio_path)
    )
    portfolio_symbols = prioritize_symbols(
        portfolio_symbols,
        priority_symbols=parse_symbol_tokens(args.priority_symbols),
    )
    feature_symbols = load_feature_symbols(bundle_override)

    guard = RSSGuard(
        log_path=Path(args.rss_log),
        label="continuity_validation_data_refresh",
        budget_bytes=max(1, int(args.memory_budget_bytes)),
        soft_limit_bytes=max(1, int(args.soft_rss_bytes)),
        hard_limit_bytes=max(1, int(args.memory_budget_bytes)),
    )

    payload: dict[str, Any] = {
        "artifact_kind": "continuity_validation_data_refresh",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "error": None,
        "validation_mode": "continuity_only_extension_refresh",
        "canonical_source": "binance_raw_aggtrades",
        "ohlcv_derivation": "derived_from_raw_aggtrades",
        "aggregation_backend_requested": str(os.getenv(RAW_FIRST_BACKEND_ENV, "auto") or "auto"),
        "aggregation_backend_resolved": resolve_raw_aggtrades_backend_name(),
        "final_signoff_source_of_truth": False,
        "collection_cutoff_utc": iso_utc(cutoff_dt),
        "portfolio_symbols": portfolio_symbols,
        "feature_symbols": feature_symbols,
        "ohlcv_results": [],
        "feature_results": [],
        "support_inventory": {},
        "parallel": {},
    }

    try:
        guard.checkpoint(
            "start",
            {
                "portfolio_symbols": portfolio_symbols,
                "feature_symbols": feature_symbols,
                "collection_cutoff_utc": iso_utc(cutoff_dt),
            },
        )
        requested_workers = (
            max(1, min(len(portfolio_symbols), int(os.cpu_count() or 1)))
            if int(args.max_workers) <= 0
            else int(args.max_workers)
        )
        ohlcv_results, parallel_meta = refresh_ohlcv_symbols(
            symbols=portfolio_symbols,
            db_path=str(args.db_path),
            exchange_id=str(args.exchange_id),
            cutoff_dt=cutoff_dt,
            floor_dt=floor_dt,
            guard=guard,
            max_workers=requested_workers,
            memory_budget_bytes=max(1, int(args.memory_budget_bytes)),
            reserve_memory_bytes=max(0, int(args.parallel_reserve_bytes)),
            per_worker_memory_bytes=max(1, int(args.parallel_per_worker_bytes)),
        )
        payload["ohlcv_results"] = [asdict(result) for result in ohlcv_results]
        payload["parallel"] = dict(parallel_meta)
        for symbol in feature_symbols:
            guard.checkpoint("before_feature_symbol", {"symbol": symbol})
            result = refresh_feature_tail(
                symbol=symbol,
                db_path=str(args.db_path),
                exchange_id=str(args.exchange_id),
                cutoff_dt=cutoff_dt,
                floor_dt=floor_dt,
            )
            payload["feature_results"].append(asdict(result))
            guard.checkpoint(
                "after_feature_symbol",
                {"symbol": symbol, "upserted_rows": result.upserted_rows},
            )
        inventory_payload = (
            build_strategy_support_inventory(
                db_path=str(args.db_path),
                exchange=str(args.exchange_id),
                symbols=feature_symbols,
            )
            if feature_symbols
            else _empty_inventory_payload(db_path=str(args.db_path), exchange_id=str(args.exchange_id))
        )
        inventory_outputs = write_strategy_support_inventory(
            payload=inventory_payload,
            json_path=str(args.support_inventory_json),
            csv_path=str(args.support_inventory_csv),
        )
        payload["support_inventory"] = {
            "json_path": inventory_outputs.get("json_path"),
            "csv_path": inventory_outputs.get("csv_path"),
            "symbol_count": inventory_payload.get("symbol_count"),
        }
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["memory"] = guard.finalize(
            status=str(payload.get("status") or "completed"),
            error=str(payload.get("error") or "") or None,
        )
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        output_md.write_text(build_markdown(payload), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
