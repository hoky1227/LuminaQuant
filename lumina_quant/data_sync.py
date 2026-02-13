"""Binance OHLCV synchronization helpers."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import ccxt
from lumina_quant.market_data import (
    connect_market_data_db,
    ensure_market_ohlcv_schema,
    export_ohlcv_to_csv,
    get_last_ohlcv_timestamp_ms,
    normalize_symbol,
    symbol_csv_filename,
    timeframe_to_milliseconds,
    upsert_ohlcv_rows,
)


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def parse_timestamp_input(value: str | int | float | None) -> int | None:
    """Parse timestamp inputs in ISO8601/seconds/milliseconds into milliseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            return numeric * 1000
        return numeric

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return parse_timestamp_input(int(text))
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def create_binance_exchange(
    *,
    api_key: str = "",
    secret_key: str = "",
    market_type: str = "spot",
    testnet: bool = False,
) -> ccxt.Exchange:
    """Create a configured Binance CCXT client."""
    kwargs: dict[str, Any] = {
        "enableRateLimit": True,
    }
    if api_key:
        kwargs["apiKey"] = api_key
    if secret_key:
        kwargs["secret"] = secret_key
    if str(market_type).lower() == "future":
        kwargs["options"] = {"defaultType": "future"}
    exchange = ccxt.binance(kwargs)
    if testnet:
        exchange.set_sandbox_mode(True)
    return exchange


def _fetch_ohlcv_with_retry(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    limit: int,
    retries: int,
    base_wait_sec: float,
) -> list[list[float]]:
    wait = max(0.1, float(base_wait_sec))
    attempt = 0
    while True:
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception:
            attempt += 1
            if attempt > max(0, int(retries)):
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 10.0)


def _normalize_ohlcv_batch(
    batch: Sequence[Sequence[Any]],
    *,
    cursor_ms: int,
    until_ms: int,
) -> list[tuple[float, float, float, float, float, float]]:
    out: list[tuple[float, float, float, float, float, float]] = []
    last_ts = cursor_ms - 1
    for row in batch:
        if len(row) < 6:
            continue
        ts = int(row[0])
        if ts < cursor_ms or ts <= last_ts:
            continue
        if ts > until_ms:
            break
        o = float(row[1])
        h = float(row[2])
        low_price = float(row[3])
        c = float(row[4])
        v = float(row[5])
        out.append((float(ts), o, h, low_price, c, v))
        last_ts = ts
    return out


@dataclass(slots=True)
class SyncStats:
    """Synchronization result summary for a symbol."""

    symbol: str
    fetched_rows: int
    upserted_rows: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None


def sync_symbol_ohlcv(
    *,
    exchange: ccxt.Exchange,
    db_path: str,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
) -> SyncStats:
    """Synchronize one symbol OHLCV range into SQLite with idempotent upserts."""
    conn = connect_market_data_db(db_path)
    try:
        ensure_market_ohlcv_schema(conn)
        tf_ms = timeframe_to_milliseconds(timeframe)
        stream_symbol = normalize_symbol(symbol)
        cursor = max(0, int(start_ms))
        until = max(cursor, int(end_ms))

        fetched_rows = 0
        upserted_rows = 0
        first_ts = None
        last_ts = None

        batch_count = 0
        while cursor <= until and batch_count < max(1, int(max_batches)):
            batch_count += 1
            raw_rows = _fetch_ohlcv_with_retry(
                exchange,
                stream_symbol,
                timeframe,
                since_ms=cursor,
                limit=max(1, int(limit)),
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            if not raw_rows:
                break

            normalized = _normalize_ohlcv_batch(raw_rows, cursor_ms=cursor, until_ms=until)
            if not normalized:
                cursor += tf_ms
                continue

            fetched_rows += len(normalized)
            if first_ts is None:
                first_ts = int(normalized[0][0])
            last_ts = int(normalized[-1][0])

            upserted_rows += upsert_ohlcv_rows(
                conn,
                exchange=str(exchange_id).lower(),
                symbol=stream_symbol,
                timeframe=timeframe,
                rows=normalized,
                source="binance_sync",
            )

            next_cursor = last_ts + tf_ms
            if next_cursor <= cursor:
                cursor += tf_ms
            else:
                cursor = next_cursor

            rate_limit_sec = float(getattr(exchange, "rateLimit", 0) or 0) / 1000.0
            if rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

        return SyncStats(
            symbol=stream_symbol,
            fetched_rows=fetched_rows,
            upserted_rows=upserted_rows,
            first_timestamp_ms=first_ts,
            last_timestamp_ms=last_ts,
        )
    finally:
        conn.close()


def sync_market_data(
    *,
    exchange: ccxt.Exchange,
    db_path: str,
    exchange_id: str,
    symbol_list: Sequence[str],
    timeframe: str,
    since_ms: int | None = None,
    until_ms: int | None = None,
    force_full: bool = False,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
    export_csv_dir: str | None = None,
) -> list[SyncStats]:
    """Synchronize OHLCV for multiple symbols and optionally export CSV copies."""
    effective_until = int(until_ms) if until_ms is not None else _now_ms()
    default_since = (
        int(since_ms)
        if since_ms is not None
        else int(datetime(2017, 1, 1, tzinfo=UTC).timestamp() * 1000)
    )

    conn = connect_market_data_db(db_path)
    try:
        ensure_market_ohlcv_schema(conn)
        stats: list[SyncStats] = []

        for symbol in symbol_list:
            stream_symbol = normalize_symbol(symbol)
            last_ts = get_last_ohlcv_timestamp_ms(
                conn,
                exchange=str(exchange_id).lower(),
                symbol=stream_symbol,
                timeframe=timeframe,
            )
            start_ms = default_since
            if last_ts is not None and not force_full:
                start_ms = last_ts + timeframe_to_milliseconds(timeframe)

            stat = sync_symbol_ohlcv(
                exchange=exchange,
                db_path=db_path,
                exchange_id=exchange_id,
                symbol=stream_symbol,
                timeframe=timeframe,
                start_ms=start_ms,
                end_ms=effective_until,
                limit=limit,
                max_batches=max_batches,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            stats.append(stat)

            if export_csv_dir:
                csv_path = f"{export_csv_dir}/{symbol_csv_filename(stream_symbol)}"
                export_ohlcv_to_csv(
                    db_path,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    timeframe=timeframe,
                    csv_path=csv_path,
                )
        return stats
    finally:
        conn.close()
