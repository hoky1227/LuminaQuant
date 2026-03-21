"""Canonical raw-first Binance futures market-data lineage helpers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
import re
from typing import Any

import polars as pl

_TIMEFRAME_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
    "M": 2_592_000_000,
}
_TIMEFRAME_PATTERN = re.compile(r"^([1-9][0-9]*)([smhdwM])$")

_RAW_COLUMNS = (
    "agg_trade_id",
    "timestamp_ms",
    "price",
    "quantity",
    "is_buyer_maker",
)
_OHLCV_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")


def normalize_timeframe_token(timeframe: str) -> str:
    raw = str(timeframe or "").strip()
    if not raw:
        raise ValueError("Timeframe cannot be empty")
    value = raw[:-1].strip()
    unit_raw = raw[-1]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")
    unit = "M" if unit_raw == "M" else unit_raw.lower()
    token = f"{int(value)}{unit}"
    if _TIMEFRAME_PATTERN.fullmatch(token) is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return token


def timeframe_to_milliseconds(timeframe: str) -> int:
    token = normalize_timeframe_token(timeframe)
    unit = token[-1]
    value = int(token[:-1])
    unit_ms = _TIMEFRAME_UNIT_MS.get(unit)
    if unit_ms is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return int(value * unit_ms)


def normalize_exchange_timestamp_ms(value: Any, *, source: str) -> int:
    """Normalize official Binance timestamps and fail on non-ms units."""
    try:
        ts = int(value)
    except Exception as exc:  # pragma: no cover - defensive input guard
        raise ValueError(f"{source} timestamp is not an integer: {value!r}") from exc

    magnitude = abs(ts)
    if magnitude < 100_000_000_000:
        raise ValueError(
            f"{source} timestamp must be milliseconds, got seconds-like value {ts}."
        )
    if magnitude >= 100_000_000_000_000:
        raise ValueError(
            f"{source} timestamp must be milliseconds, got microseconds-like value {ts}."
        )
    return ts


def latest_complete_bucket_start_ms(
    *,
    timeframe: str,
    complete_through_ms: int | None,
) -> int | None:
    """Return the latest fully complete bucket start timestamp in ms."""
    if complete_through_ms is None:
        return None
    bucket_ms = int(timeframe_to_milliseconds(timeframe))
    boundary = int(complete_through_ms)
    if boundary < bucket_ms - 1:
        return None
    return (((boundary + 1) // bucket_ms) * bucket_ms) - bucket_ms


def coerce_raw_aggtrades_frame(
    rows: pl.DataFrame | Iterable[dict[str, Any]],
    *,
    source: str,
) -> pl.DataFrame:
    """Coerce raw aggTrades rows into canonical parquet schema."""
    frame = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(list(rows or []))
    if frame.is_empty():
        return pl.DataFrame(
            {column: [] for column in _RAW_COLUMNS},
            schema={
                "agg_trade_id": pl.Int64,
                "timestamp_ms": pl.Int64,
                "price": pl.Float64,
                "quantity": pl.Float64,
                "is_buyer_maker": pl.Boolean,
            },
        )

    missing = [column for column in _RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} rows missing raw aggTrades columns: {missing}")

    normalized_rows: list[dict[str, Any]] = []
    for row in frame.select(list(_RAW_COLUMNS)).to_dicts():
        timestamp_ms = normalize_exchange_timestamp_ms(row.get("timestamp_ms"), source=source)
        price = float(row.get("price") or 0.0)
        quantity = float(row.get("quantity") or 0.0)
        if price <= 0.0 or quantity < 0.0:
            continue
        normalized_rows.append(
            {
                "agg_trade_id": int(row.get("agg_trade_id") or 0),
                "timestamp_ms": int(timestamp_ms),
                "price": float(price),
                "quantity": float(quantity),
                "is_buyer_maker": bool(row.get("is_buyer_maker")),
            }
        )

    if not normalized_rows:
        return pl.DataFrame(
            {column: [] for column in _RAW_COLUMNS},
            schema={
                "agg_trade_id": pl.Int64,
                "timestamp_ms": pl.Int64,
                "price": pl.Float64,
                "quantity": pl.Float64,
                "is_buyer_maker": pl.Boolean,
            },
        )
    return (
        pl.DataFrame(normalized_rows)
        .select(list(_RAW_COLUMNS))
        .sort(["timestamp_ms", "agg_trade_id"])
        .unique(subset=["timestamp_ms", "agg_trade_id"], keep="last")
        .sort(["timestamp_ms", "agg_trade_id"])
    )


def raw_aggtrades_to_1s_frame(
    rows: pl.DataFrame | Iterable[dict[str, Any]],
    *,
    source: str,
    range_start_ms: int | None = None,
    range_end_ms: int | None = None,
    previous_close: float | None = None,
    complete_through_ms: int | None = None,
) -> pl.DataFrame:
    """Derive continuous canonical 1s OHLCV bars from raw aggTrades."""
    raw = coerce_raw_aggtrades_frame(rows, source=source)
    if raw.is_empty():
        return pl.DataFrame(
            {column: [] for column in _OHLCV_COLUMNS},
            schema={
                "datetime": pl.Datetime(time_unit="ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

    trade_rows = raw.to_dicts()
    trade_rows.sort(key=lambda item: (int(item["timestamp_ms"]), int(item["agg_trade_id"])))

    effective_complete_through_ms = (
        int(complete_through_ms)
        if complete_through_ms is not None
        else int(range_end_ms)
        if range_end_ms is not None
        else int(trade_rows[-1]["timestamp_ms"])
    )
    last_complete_second = latest_complete_bucket_start_ms(
        timeframe="1s",
        complete_through_ms=effective_complete_through_ms,
    )
    if last_complete_second is None:
        return pl.DataFrame(
            {column: [] for column in _OHLCV_COLUMNS},
            schema={
                "datetime": pl.Datetime(time_unit="ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

    first_trade_second = (int(trade_rows[0]["timestamp_ms"]) // 1000) * 1000
    start_second = (
        max(int(range_start_ms) // 1000 * 1000, first_trade_second)
        if previous_close is None and range_start_ms is not None
        else int(range_start_ms) // 1000 * 1000
        if range_start_ms is not None
        else first_trade_second
    )
    end_second = int(last_complete_second)
    if range_end_ms is not None:
        end_second = min(int(range_end_ms) // 1000 * 1000, end_second)
    if end_second < start_second:
        return pl.DataFrame(
            {column: [] for column in _OHLCV_COLUMNS},
            schema={
                "datetime": pl.Datetime(time_unit="ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )

    buckets: dict[int, list[float]] = {}
    for row in trade_rows:
        timestamp_ms = int(row["timestamp_ms"])
        if range_start_ms is not None and timestamp_ms < int(range_start_ms):
            continue
        if range_end_ms is not None and timestamp_ms > int(range_end_ms):
            continue
        if timestamp_ms > int(effective_complete_through_ms):
            continue
        bucket_ms = (timestamp_ms // 1000) * 1000
        current = buckets.get(bucket_ms)
        price = float(row["price"])
        quantity = float(row["quantity"])
        if current is None:
            buckets[bucket_ms] = [price, price, price, price, quantity]
        else:
            current[1] = max(current[1], price)
            current[2] = min(current[2], price)
            current[3] = price
            current[4] += quantity

    rows_1s: list[dict[str, Any]] = []
    close_cursor = previous_close
    cursor_ms = int(start_second)
    while cursor_ms <= int(end_second):
        bucket = buckets.get(cursor_ms)
        if bucket is None:
            if close_cursor is None:
                cursor_ms += 1000
                continue
            open_price = close_cursor
            high_price = close_cursor
            low_price = close_cursor
            close_price = close_cursor
            volume = 0.0
        else:
            open_price = float(bucket[0])
            high_price = float(bucket[1])
            low_price = float(bucket[2])
            close_price = float(bucket[3])
            volume = float(bucket[4])
            close_cursor = close_price
        rows_1s.append(
            {
                "datetime": datetime.fromtimestamp(cursor_ms / 1000.0, tz=UTC).replace(tzinfo=None),
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
            }
        )
        cursor_ms += 1000

    if not rows_1s:
        return pl.DataFrame(
            {column: [] for column in _OHLCV_COLUMNS},
            schema={
                "datetime": pl.Datetime(time_unit="ms"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )
    return pl.DataFrame(rows_1s).with_columns(
        [
            pl.col("datetime").cast(pl.Datetime(time_unit="ms")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    )


def resample_1s_frame(
    frame_1s: pl.DataFrame,
    *,
    timeframe: str,
    complete_through_ms: int | None = None,
) -> pl.DataFrame:
    """Resample canonical 1s bars into higher timeframes and drop incomplete tail."""
    token = normalize_timeframe_token(timeframe)
    if frame_1s.is_empty():
        return frame_1s
    if token == "1s":
        output = frame_1s.select(list(_OHLCV_COLUMNS))
    else:
        bucket_ms = int(timeframe_to_milliseconds(token))
        output = (
            frame_1s.with_columns(pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"))
            .with_columns(((pl.col("timestamp_ms") // bucket_ms) * bucket_ms).alias("bucket_ms"))
            .group_by("bucket_ms")
            .agg(
                [
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                ]
            )
            .sort("bucket_ms")
            .with_columns(pl.from_epoch("bucket_ms", time_unit="ms").alias("datetime"))
            .select(list(_OHLCV_COLUMNS))
        )

    latest_complete_bucket_ms = latest_complete_bucket_start_ms(
        timeframe=token,
        complete_through_ms=complete_through_ms,
    )
    if latest_complete_bucket_ms is None:
        return output.clear()
    return output.filter(
        pl.col("datetime").dt.epoch("ms") <= int(latest_complete_bucket_ms)
    ).sort("datetime")


__all__ = [
    "coerce_raw_aggtrades_frame",
    "latest_complete_bucket_start_ms",
    "normalize_exchange_timestamp_ms",
    "raw_aggtrades_to_1s_frame",
    "resample_1s_frame",
]
