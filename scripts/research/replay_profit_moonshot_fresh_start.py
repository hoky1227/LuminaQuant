#!/usr/bin/env python3
"""Fresh-start profit moonshot strategy replay independent of existing incumbents.

This screen intentionally avoids the incumbent ETH shock-reversion and prior
lead-lag sleeves. It starts from refreshed raw-first materialized OHLCV and
Binance support features, then evaluates new cross-sectional/derivatives-context
families under a one-position state machine before any live-equivalent mode can
be considered.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import resource
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.config import BacktestConfig, BaseConfig
from lumina_quant.market_data import load_futures_feature_points_from_db

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.revalidate_live_equivalent_candidates import (  # noqa: E402
    SplitWindow,
    _metrics_from_equity_totals,
    _safe_float,
    _split_windows,
)
from scripts.research.replay_eth_shock_filters import (  # noqa: E402
    _fill_price,
    _materialized_paths,
    _rolling_rms_log_return,
    _rolling_zscore,
    _fmt_float,
    _fmt_pct,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul"
)
DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT")
BASELINE_OOS_RETURN = 0.008284
SHADOW_OOS_MDD = 0.001778
SUCCESS_SHARPE = 1.0
TARGET_ALLOCATION = 0.008
MAX_ORDER_VALUE = 175.0
FEATURE_VALUE_COLUMNS = (
    "funding_rate",
    "open_interest",
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "liquidation_long_notional",
    "liquidation_short_notional",
)


@dataclass(frozen=True, slots=True)
class FreshSpec:
    name: str
    family: str
    lookback_bars: int
    threshold: float
    hold_bars: int
    cooldown_bars: int
    stop_loss_pct: float
    take_profit_pct: float
    allow_long: bool = True
    allow_short: bool = True
    rv_lookback_bars: int = 24
    max_rv: float = 0.0
    broad_min_abs: float = 0.0
    min_abs_return: float = 0.0
    entry_hours: tuple[int, ...] = ()
    funding_abs_cap: float = 0.0
    funding_rank_min: float = 0.0
    compression_lookback_bars: int = 24
    compression_quantile: float = 0.0
    flow_lookback_bars: int = 0
    flow_threshold: float = 0.0
    flow_persistence_bars: int = 0
    flow_persistence_threshold: float = 0.0
    adaptive_lookback_bars: int = 12
    sharpe_lookback_bars: int = 0
    sharpe_rank_min: float = 0.0
    oi_rank_min: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "lookback_bars": self.lookback_bars,
            "threshold": self.threshold,
            "hold_bars": self.hold_bars,
            "cooldown_bars": self.cooldown_bars,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "rv_lookback_bars": self.rv_lookback_bars,
            "max_rv": self.max_rv,
            "broad_min_abs": self.broad_min_abs,
            "min_abs_return": self.min_abs_return,
            "entry_hours": list(self.entry_hours),
            "funding_abs_cap": self.funding_abs_cap,
            "funding_rank_min": self.funding_rank_min,
            "compression_lookback_bars": self.compression_lookback_bars,
        "compression_quantile": self.compression_quantile,
        "flow_lookback_bars": self.flow_lookback_bars,
        "flow_threshold": self.flow_threshold,
        "flow_persistence_bars": self.flow_persistence_bars,
        "flow_persistence_threshold": self.flow_persistence_threshold,
        "adaptive_lookback_bars": self.adaptive_lookback_bars,
        "sharpe_lookback_bars": self.sharpe_lookback_bars,
        "sharpe_rank_min": self.sharpe_rank_min,
        "oi_rank_min": self.oi_rank_min,
        "target_allocation": TARGET_ALLOCATION,
        "max_order_value": MAX_ORDER_VALUE,
    }


def _compact(symbol: str) -> str:
    return str(symbol).replace("/", "").upper()


def _column_symbol(column_prefix: str) -> str:
    token = column_prefix.upper()
    if token.endswith("USDT"):
        return f"{token[:-4]}/USDT"
    return token


def _month_iter(start: date, end: date) -> list[date]:
    if end < start:
        return []
    cursor = date(start.year, start.month, 1)
    out: list[date] = []
    while cursor <= end:
        out.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return out


def _month_end(month_start: date) -> date:
    if month_start.month == 12:
        return date(month_start.year + 1, 1, 1) - timedelta(days=1)
    return date(month_start.year, month_start.month + 1, 1) - timedelta(days=1)


def _raw_month_path(*, market_root: Path, exchange: str, symbol: str, month_start: date) -> Path:
    return (
        market_root
        / "market_ohlcv_1s"
        / str(exchange).lower()
        / _compact(symbol)
        / f"{month_start:%Y-%m}.parquet"
    )


def _aggregate_1s_paths_to_hourly(
    *,
    paths: list[str],
    prefix: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame(
            schema={
                "datetime": pl.Datetime(time_unit="ms"),
                f"{prefix}_open": pl.Float64,
                f"{prefix}_high": pl.Float64,
                f"{prefix}_low": pl.Float64,
                f"{prefix}_close": pl.Float64,
                f"{prefix}_volume": pl.Float64,
            }
        )
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time())
    # Collect one month/day at a time.  A single all-period 1s lazy aggregation
    # is fast but peaks around 6.5 GiB for BTC alone on this data set; chunking
    # keeps the whole fresh-start session comfortably below the 8 GiB ceiling.
    return (
        pl.scan_parquet(paths)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")).alias("datetime"))
        .filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") < end_dt))
        .set_sorted("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("open").first().alias(f"{prefix}_open"),
                pl.col("high").max().alias(f"{prefix}_high"),
                pl.col("low").min().alias(f"{prefix}_low"),
                pl.col("close").last().alias(f"{prefix}_close"),
                pl.col("volume").sum().alias(f"{prefix}_volume"),
            ]
        )
        .drop_nulls([f"{prefix}_open", f"{prefix}_close"])
        .collect(engine="streaming")
        .sort("datetime")
    )


def _incomplete_hourly_days(frame: pl.DataFrame, *, start: date, end: date) -> list[date]:
    expected_hours = 24
    observed: dict[date, int] = {}
    if not frame.is_empty() and "datetime" in frame.columns:
        observed = {
            row["date"]: int(row["len"])
            for row in (
                frame.with_columns(pl.col("datetime").dt.date().alias("date"))
                .group_by("date")
                .len()
                .to_dicts()
            )
        }
    missing: list[date] = []
    cursor = start
    while cursor <= end:
        if int(observed.get(cursor, 0)) < expected_hours:
            missing.append(cursor)
        cursor += timedelta(days=1)
    return missing


def _load_daily_materialized_hourly(
    *,
    market_root: Path,
    exchange: str,
    symbol: str,
    prefix: str,
    days: list[date],
) -> list[pl.DataFrame]:
    frames: list[pl.DataFrame] = []
    for day in days:
        frame = _aggregate_1s_paths_to_hourly(
            paths=_materialized_paths(
                market_root=market_root,
                exchange=exchange,
                symbol=symbol,
                timeframe="1s",
                start=day,
                end=day,
            ),
            prefix=prefix,
            start=day,
            end=day,
        )
        if not frame.is_empty():
            frames.append(frame)
        gc.collect()
    return frames


def _load_symbol_hourly(
    *, market_root: Path, exchange: str, symbol: str, start: date, end: date
) -> pl.DataFrame:
    prefix = _compact(symbol).lower()
    frames: list[pl.DataFrame] = []
    for month_start in _month_iter(start, end):
        window_start = max(start, month_start)
        window_end = min(end, _month_end(month_start))
        monthly_path = _raw_month_path(
            market_root=market_root,
            exchange=exchange,
            symbol=symbol,
            month_start=month_start,
        )
        if monthly_path.exists():
            monthly_frame = _aggregate_1s_paths_to_hourly(
                paths=[str(monthly_path)],
                prefix=prefix,
                start=window_start,
                end=window_end,
            )
            frames.append(monthly_frame)
            missing_days = _incomplete_hourly_days(monthly_frame, start=window_start, end=window_end)
            if missing_days:
                frames.extend(
                    _load_daily_materialized_hourly(
                        market_root=market_root,
                        exchange=exchange,
                        symbol=symbol,
                        prefix=prefix,
                        days=missing_days,
                    )
                )
            gc.collect()
            continue
        frames.append(
            _aggregate_1s_paths_to_hourly(
                paths=_materialized_paths(
                    market_root=market_root,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe="1s",
                    start=window_start,
                    end=window_end,
                ),
                prefix=prefix,
                start=window_start,
                end=window_end,
            )
        )
        gc.collect()
    frames = [frame for frame in frames if not frame.is_empty()]
    if not frames:
        raise RuntimeError(f"no committed raw-first 1s/monthly paths for {symbol} {start}..{end}")
    return (
        pl.concat(frames, how="vertical")
        .sort("datetime")
        .unique(subset=["datetime"], keep="last", maintain_order=True)
    )


def _load_feature_hourly(
    *, market_root: Path, exchange: str, symbol: str, start: date, end: date
) -> tuple[pl.DataFrame, dict[str, Any]]:
    frame = load_futures_feature_points_from_db(
        str(market_root),
        exchange=exchange,
        symbol=symbol,
        start_date=datetime.combine(start, datetime.min.time()),
        end_date=datetime.combine(end + timedelta(days=1), datetime.min.time()),
    )
    compact = _compact(symbol).lower()
    if frame.is_empty():
        return pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")}), {
            "symbol": symbol,
            "rows": 0,
            "funding_rows": 0,
            "open_interest_rows": 0,
        }
    aligned = frame
    for column in ("timestamp_ms", *FEATURE_VALUE_COLUMNS):
        if column not in aligned.columns:
            dtype = pl.Int64 if column == "timestamp_ms" else pl.Float64
            aligned = aligned.with_columns(pl.lit(None, dtype=dtype).alias(column))
    aligned = aligned.select(["timestamp_ms", *FEATURE_VALUE_COLUMNS]).filter(pl.col("timestamp_ms").is_not_null())
    metadata = {
        "symbol": symbol,
        "rows": int(aligned.height),
        "funding_rows": int(aligned.select(pl.col("funding_rate").is_not_null().sum()).item()),
        "open_interest_rows": int(aligned.select(pl.col("open_interest").is_not_null().sum()).item()),
        "taker_flow_rows": int(
            aligned.select(
                (
                    pl.col("taker_buy_quote_volume").is_not_null()
                    | pl.col("taker_sell_quote_volume").is_not_null()
                ).sum()
            ).item()
        ),
    }
    if aligned.is_empty():
        return pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")}), metadata
    hourly = (
        aligned.with_columns(pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms").alias("datetime"))
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("funding_rate").drop_nulls().last().alias(f"{compact}_funding_rate"),
                pl.col("open_interest").drop_nulls().last().alias(f"{compact}_open_interest"),
                pl.col("taker_buy_base_volume").sum().alias(f"{compact}_taker_buy_base_volume"),
                pl.col("taker_sell_base_volume").sum().alias(f"{compact}_taker_sell_base_volume"),
                pl.col("taker_buy_quote_volume").sum().alias(f"{compact}_taker_buy_quote_volume"),
                pl.col("taker_sell_quote_volume").sum().alias(f"{compact}_taker_sell_quote_volume"),
                pl.col("liquidation_long_notional").sum().alias(f"{compact}_liquidation_long_notional"),
                pl.col("liquidation_short_notional").sum().alias(f"{compact}_liquidation_short_notional"),
            ]
        )
        .sort("datetime")
    )
    return hourly, metadata


def _joined_panel(
    *, market_root: Path, exchange: str, symbols: list[str], start: date, end: date
) -> tuple[pl.DataFrame, dict[str, Any]]:
    panels = [_load_symbol_hourly(market_root=market_root, exchange=exchange, symbol=s, start=start, end=end) for s in symbols]
    panel = panels[0]
    for frame in panels[1:]:
        panel = panel.join(frame, on="datetime", how="inner")
    feature_meta: list[dict[str, Any]] = []
    for symbol in symbols:
        features, meta = _load_feature_hourly(market_root=market_root, exchange=exchange, symbol=symbol, start=start, end=end)
        feature_meta.append(meta)
        if not features.is_empty():
            panel = panel.join(features, on="datetime", how="left")
    for symbol in symbols:
        prefix = _compact(symbol).lower()
        for suffix in FEATURE_VALUE_COLUMNS:
            column = f"{prefix}_{suffix}"
            if column not in panel.columns:
                panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    return panel.sort("datetime"), {"feature_points": feature_meta, "rows": int(panel.height)}


def _to_float_array(panel: pl.DataFrame, column: str) -> np.ndarray:
    if column not in panel.columns:
        return np.full(panel.height, np.nan, dtype=float)
    return np.asarray([float(v) if v is not None else np.nan for v in panel[column].to_list()], dtype=float)


def _pct_change(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    base = values[:-lookback]
    latest = values[lookback:]
    valid = np.isfinite(base) & np.isfinite(latest) & (base > 0.0)
    out[lookback:][valid] = latest[valid] / base[valid] - 1.0
    return out


def _rolling_mean(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    clean = np.where(np.isfinite(values), values, 0.0)
    counts = np.where(np.isfinite(values), 1.0, 0.0)
    csum = np.cumsum(np.insert(clean, 0, 0.0))
    ccnt = np.cumsum(np.insert(counts, 0, 0.0))
    for idx in range(lookback - 1, values.size):
        count = ccnt[idx + 1] - ccnt[idx + 1 - lookback]
        if count > 0:
            out[idx] = float((csum[idx + 1] - csum[idx + 1 - lookback]) / count)
    return out


def _rolling_sum(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if lookback <= 0:
        return out
    clean = np.where(np.isfinite(values), values, 0.0)
    counts = np.where(np.isfinite(values), 1.0, 0.0)
    csum = np.cumsum(np.insert(clean, 0, 0.0))
    ccnt = np.cumsum(np.insert(counts, 0, 0.0))
    for idx in range(lookback - 1, values.size):
        count = ccnt[idx + 1] - ccnt[idx + 1 - lookback]
        if count > 0:
            out[idx] = float(csum[idx + 1] - csum[idx + 1 - lookback])
    return out


def _ffill(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    last = np.nan
    for idx, value in enumerate(out):
        if math.isfinite(float(value)):
            last = float(value)
        else:
            out[idx] = last
    return out


def _build_arrays(panel: pl.DataFrame, symbols: list[str]) -> dict[str, Any]:
    datetimes = panel["datetime"].to_list()
    arrays: dict[str, Any] = {
        "datetime": datetimes,
        "timestamp": np.asarray([int(dt.replace(tzinfo=UTC).timestamp()) for dt in datetimes], dtype=np.int64),
        "symbols": tuple(symbols),
    }
    close_stack: list[np.ndarray] = []
    for symbol in symbols:
        prefix = _compact(symbol).lower()
        for column in ("open", "high", "low", "close", "volume", *FEATURE_VALUE_COLUMNS):
            key = f"{prefix}_{column}"
            arrays[key] = _to_float_array(panel, key)
        close = arrays[f"{prefix}_close"]
        close_stack.append(close)
        for lookback in (3, 6, 12, 24, 48, 72, 168):
            arrays[f"{prefix}_ret_{lookback}h"] = _pct_change(close, lookback)
        for lookback in (24, 48, 72):
            arrays[f"{prefix}_rv_{lookback}h"] = _rolling_rms_log_return(close, lookback)
        arrays[f"{prefix}_rv_24h_mean_72h"] = _rolling_mean(arrays[f"{prefix}_rv_24h"], 72)
    arrays[f"{prefix}_funding_ffill"] = _ffill(arrays[f"{prefix}_funding_rate"])
    arrays[f"{prefix}_open_interest_ffill"] = _ffill(arrays[f"{prefix}_open_interest"])
    buy_quote = arrays[f"{prefix}_taker_buy_quote_volume"]
    sell_quote = arrays[f"{prefix}_taker_sell_quote_volume"]
    flow_den = buy_quote + sell_quote
    arrays[f"{prefix}_flow_imbalance_1h"] = np.divide(
        buy_quote - sell_quote,
            flow_den,
            out=np.full_like(flow_den, np.nan, dtype=float),
            where=np.isfinite(flow_den) & (flow_den > 0.0),
        )
        for lookback in (3, 6, 12, 24):
            buy_sum = _rolling_sum(buy_quote, lookback)
            sell_sum = _rolling_sum(sell_quote, lookback)
            den = buy_sum + sell_sum
            flow = np.divide(
                buy_sum - sell_sum,
                den,
                out=np.full_like(den, np.nan, dtype=float),
                where=np.isfinite(den) & (den > 0.0),
            )
        arrays[f"{prefix}_flow_imbalance_{lookback}h"] = flow
        arrays[f"{prefix}_flow_z_{lookback}h"] = _rolling_zscore(flow, max(24, lookback * 6))
    for lookback in (3, 6, 12, 24):
        oi = arrays[f"{prefix}_open_interest_ffill"]
        oi_delta = np.full_like(oi, np.nan, dtype=float)
        if oi.size > lookback:
            oi_delta[lookback:] = np.divide(
                oi[lookback:] - oi[:-lookback],
                lookback,
                out=np.full_like(oi[lookback:], np.nan, dtype=float),
                where=np.isfinite(oi[:-lookback]) & np.isfinite(oi[lookback:]),
            )
        arrays[f"{prefix}_oi_delta_{lookback}h"] = oi_delta
    stacked = np.vstack(close_stack)
    market_close = np.nanmean(stacked, axis=0)
    arrays["market_close"] = market_close
    for lookback in (3, 6, 12, 24, 48, 72, 168):
        market_ret = _pct_change(market_close, lookback)
        arrays[f"market_ret_{lookback}h"] = market_ret
        residuals: list[np.ndarray] = []
        for symbol in symbols:
            prefix = _compact(symbol).lower()
            residual = arrays[f"{prefix}_ret_{lookback}h"] - market_ret
            arrays[f"{prefix}_resid_{lookback}h"] = residual
            arrays[f"{prefix}_resid_z_{lookback}h"] = _rolling_zscore(residual, max(24, lookback * 4))
            residuals.append(residual)
    return arrays


def _hour_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).hour)


def _symbol_prefix(symbol: str) -> str:
    return _compact(symbol).lower()


def _candidate_signal(spec: FreshSpec, arrays: dict[str, Any], idx: int) -> tuple[str, str, str]:
    if spec.entry_hours and _hour_utc(arrays["datetime"][idx]) not in set(spec.entry_hours):
        return "", "", "entry_hour_block"
    symbols = list(arrays["symbols"])
    lookback = int(spec.lookback_bars)
    market_ret = float(arrays[f"market_ret_{lookback}h"][idx]) if f"market_ret_{lookback}h" in arrays else np.nan
    if spec.broad_min_abs > 0.0 and (not math.isfinite(market_ret) or abs(market_ret) < spec.broad_min_abs):
        return "", "", "broad_move_missing"

    candidates: list[tuple[float, str, str]] = []
    for symbol in symbols:
        prefix = _symbol_prefix(symbol)
        close = float(arrays[f"{prefix}_close"][idx])
        if not math.isfinite(close) or close <= 0.0:
            continue
        rv_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
        if spec.max_rv > 0.0:
            rv = float(arrays.get(rv_key, np.full_like(arrays[f"{prefix}_close"], np.nan))[idx])
            if not math.isfinite(rv) or rv > spec.max_rv:
                continue
        ret = float(arrays[f"{prefix}_ret_{lookback}h"][idx])
        resid_z = float(arrays[f"{prefix}_resid_z_{lookback}h"][idx])
        funding = float(arrays[f"{prefix}_funding_ffill"][idx])
        flow = np.nan
        if spec.flow_lookback_bars > 0:
            flow_key = f"{prefix}_flow_imbalance_{spec.flow_lookback_bars}h"
            flow = float(arrays.get(flow_key, np.full_like(arrays[f"{prefix}_close"], np.nan))[idx])
        if spec.funding_abs_cap > 0.0 and (not math.isfinite(funding) or abs(funding) > spec.funding_abs_cap):
            continue
        if spec.family == "residual_reversion":
            if spec.allow_long and math.isfinite(resid_z) and resid_z <= -spec.threshold and ret <= -spec.min_abs_return:
                candidates.append((abs(resid_z), symbol, "LONG"))
            if spec.allow_short and math.isfinite(resid_z) and resid_z >= spec.threshold and ret >= spec.min_abs_return:
                candidates.append((abs(resid_z), symbol, "SHORT"))
        elif spec.family == "cross_momentum":
            if not math.isfinite(ret) or abs(ret) < spec.min_abs_return:
                continue
            if spec.allow_long and market_ret > spec.broad_min_abs and resid_z >= spec.threshold:
                candidates.append((resid_z, symbol, "LONG"))
            if spec.allow_short and market_ret < -spec.broad_min_abs and resid_z <= -spec.threshold:
                candidates.append((abs(resid_z), symbol, "SHORT"))
        elif spec.family == "funding_carry_fade":
            if not math.isfinite(funding):
                continue
            # Fade crowded positive carry only after price has already outrun the basket;
            # long negative carry only after underperformance.  No direct funding capture.
            if spec.allow_short and funding >= spec.funding_rank_min and resid_z >= spec.threshold and ret >= spec.min_abs_return:
                candidates.append((abs(funding) + abs(resid_z) / 100.0, symbol, "SHORT"))
            if spec.allow_long and funding <= -spec.funding_rank_min and resid_z <= -spec.threshold and ret <= -spec.min_abs_return:
                candidates.append((abs(funding) + abs(resid_z) / 100.0, symbol, "LONG"))
        elif spec.family == "flow_momentum":
            if not math.isfinite(ret) or not math.isfinite(flow) or abs(ret) < spec.min_abs_return:
                continue
            if spec.allow_long and ret > 0.0 and flow >= spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "LONG"))
            if spec.allow_short and ret < 0.0 and flow <= -spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "SHORT"))
        elif spec.family == "flow_exhaustion_fade":
            if not math.isfinite(ret) or not math.isfinite(flow) or abs(ret) < spec.min_abs_return:
                continue
            if spec.allow_short and ret > 0.0 and flow >= spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "SHORT"))
            if spec.allow_long and ret < 0.0 and flow <= -spec.flow_threshold:
                candidates.append((abs(flow) + abs(ret), symbol, "LONG"))
        elif spec.family == "residual_reversion_flow_confirmed":
            if not math.isfinite(resid_z) or not math.isfinite(flow):
                continue
            if (
                spec.allow_long
                and resid_z <= -spec.threshold
                and ret <= -spec.min_abs_return
                and flow >= spec.flow_threshold
            ):
                candidates.append((abs(resid_z) + abs(flow), symbol, "LONG"))
            if (
                spec.allow_short
                and resid_z >= spec.threshold
                and ret >= spec.min_abs_return
                and flow <= -spec.flow_threshold
            ):
                candidates.append((abs(resid_z) + abs(flow), symbol, "SHORT"))
        elif spec.family == "compression_breakout":
            comp_key = f"{prefix}_rv_{spec.rv_lookback_bars}h"
            rv = float(arrays[comp_key][idx]) if comp_key in arrays else np.nan
            rv_mean_key = f"{prefix}_rv_24h_mean_72h"
            rv_mean = float(arrays[rv_mean_key][idx]) if rv_mean_key in arrays else np.nan
            if not math.isfinite(rv) or not math.isfinite(rv_mean) or rv_mean <= 0.0:
                continue
            if rv / rv_mean > spec.compression_quantile:
                continue
            if spec.allow_long and ret >= spec.threshold:
                candidates.append((abs(ret), symbol, "LONG"))
            if spec.allow_short and ret <= -spec.threshold:
                candidates.append((abs(ret), symbol, "SHORT"))
    if not candidates:
        return "", "", "signal_missing"
    if spec.family == "adaptive_trend":
        if spec.sharpe_lookback_bars <= 0:
            spec_sharpe_lb = spec.lookback_bars
        else:
            spec_sharpe_lb = spec.sharpe_lookback_bars
        trend = arrays.get(f"market_ret_{spec.adaptive_lookback_bars}h", np.array([], dtype=float))
        if trend.size == 0 or not math.isfinite(float(trend[idx])):
            return "", "", "adaptive_signal_missing"
        if spec.allow_long and float(trend[idx]) >= spec.threshold and ret >= spec.min_abs_return:
            candidates.append((abs(float(trend[idx])), symbol, "LONG"))
        if spec.allow_short and float(trend[idx]) <= -spec.threshold and ret <= -spec.min_abs_return:
            candidates.append((abs(float(trend[idx])), symbol, "SHORT"))
        if not candidates:
            return "", "", "adaptive_signal_missing"
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, symbol, side = candidates[0]
    return symbol, side, ""


def _run_split(
    *, spec: FreshSpec, arrays: dict[str, Any], split: SplitWindow, include_equity: bool = False
) -> dict[str, Any]:
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = int(datetime.combine(split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()) - 1
    indices = np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))
    if indices.size == 0:
        return {"metrics": {}, "round_trips": 0, "fills": 0, "reject_counts": {"split_empty": 1}, "liquidations": 0}

    cash = 10_000.0
    qty = 0.0
    position_side = "OUT"
    position_symbol = ""
    entry_price = 0.0
    bars_held = 0
    cooldown = 0
    equity_history: list[float] = []
    fills = 0
    round_trips = 0
    reject_counts: dict[str, int] = {}

    def record_reject(reason: str) -> None:
        reject_counts[reason] = int(reject_counts.get(reason, 0)) + 1

    def mark_equity_at(idx: int) -> float:
        if position_side == "OUT" or not position_symbol:
            return cash
        prefix = _symbol_prefix(position_symbol)
        close = float(arrays[f"{prefix}_close"][idx])
        if not math.isfinite(close) or close <= 0.0:
            close = entry_price
        if position_side == "LONG":
            return cash + qty * close
        if position_side == "SHORT":
            return cash - qty * close
        return cash

    for idx in indices:
        if position_side != "OUT":
            bars_held += 1
            prefix = _symbol_prefix(position_symbol)
            close = float(arrays[f"{prefix}_close"][idx])
            high = float(arrays[f"{prefix}_high"][idx])
            low = float(arrays[f"{prefix}_low"][idx])
            open_ = float(arrays[f"{prefix}_open"][idx])
            high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0
            exit_reason = ""
            exit_price = close
            if position_side == "LONG":
                stop = entry_price * (1.0 - spec.stop_loss_pct) if spec.stop_loss_pct > 0.0 else -math.inf
                take = entry_price * (1.0 + spec.take_profit_pct) if spec.take_profit_pct > 0.0 else math.inf
                if low <= stop:
                    exit_reason = "stop"
                    exit_price = min(open_, stop) if open_ < stop else stop
                elif high >= take:
                    exit_reason = "take_profit"
                    exit_price = max(open_, take) if open_ > take else take
                elif bars_held >= spec.hold_bars:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "SELL", high_low_vol=high_low_vol)
                    cash += qty * fill - qty * fill * fee_rate
            elif position_side == "SHORT":
                stop = entry_price * (1.0 + spec.stop_loss_pct) if spec.stop_loss_pct > 0.0 else math.inf
                take = entry_price * (1.0 - spec.take_profit_pct) if spec.take_profit_pct > 0.0 else -math.inf
                if high >= stop:
                    exit_reason = "stop"
                    exit_price = max(open_, stop) if open_ > stop else stop
                elif low <= take:
                    exit_reason = "take_profit"
                    exit_price = min(open_, take) if open_ < take else take
                elif bars_held >= spec.hold_bars:
                    exit_reason = "max_hold"
                if exit_reason:
                    fill, fee_rate = _fill_price(exit_price, "BUY", high_low_vol=high_low_vol)
                    cash -= qty * fill + qty * fill * fee_rate
            if exit_reason:
                position_side = "OUT"
                position_symbol = ""
                qty = 0.0
                entry_price = 0.0
                bars_held = 0
                cooldown = max(0, int(spec.cooldown_bars))
                round_trips += 1
                fills += 1

        if position_side == "OUT":
            if cooldown > 0:
                cooldown -= 1
                record_reject("cooldown")
            else:
                symbol, side, reason = _candidate_signal(spec, arrays, int(idx))
                if not symbol or not side:
                    record_reject(reason or "signal_missing")
                else:
                    prefix = _symbol_prefix(symbol)
                    close = float(arrays[f"{prefix}_close"][idx])
                    high = float(arrays[f"{prefix}_high"][idx])
                    low = float(arrays[f"{prefix}_low"][idx])
                    open_ = float(arrays[f"{prefix}_open"][idx])
                    volume = max(0.0, float(arrays[f"{prefix}_volume"][idx]))
                    high_low_vol = max(0.0, (high - low) / open_) if open_ > 0.0 else 0.0
                    notional = min(TARGET_ALLOCATION * mark_equity_at(int(idx)), MAX_ORDER_VALUE)
                    raw_qty = math.floor((notional / close) / 0.001) * 0.001
                    max_fill_qty = volume * 0.10
                    order_qty = min(raw_qty, max_fill_qty)
                    if order_qty * close < 5.0 or order_qty <= 0.0:
                        record_reject("fill_or_min_notional")
                    elif side == "LONG":
                        fill, fee_rate = _fill_price(close, "BUY", high_low_vol=high_low_vol)
                        cash -= order_qty * fill + order_qty * fill * fee_rate
                        qty = order_qty
                        position_symbol = symbol
                        position_side = "LONG"
                        entry_price = fill
                        bars_held = 0
                        fills += 1
                    else:
                        fill, fee_rate = _fill_price(close, "SELL", high_low_vol=high_low_vol)
                        cash += order_qty * fill - order_qty * fill * fee_rate
                        qty = order_qty
                        position_symbol = symbol
                        position_side = "SHORT"
                        entry_price = fill
                        bars_held = 0
                        fills += 1
        equity_history.append(mark_equity_at(int(idx)))

    if position_side != "OUT" and indices.size:
        idx = int(indices[-1])
        prefix = _symbol_prefix(position_symbol)
        close = float(arrays[f"{prefix}_close"][idx])
        if position_side == "LONG":
            fill, fee_rate = _fill_price(close, "SELL", high_low_vol=0.0)
            cash += qty * fill - qty * fill * fee_rate
        else:
            fill, fee_rate = _fill_price(close, "BUY", high_low_vol=0.0)
            cash -= qty * fill + qty * fill * fee_rate
        equity_history[-1] = cash
        round_trips += 1
        fills += 1

    metrics = _metrics_from_equity_totals(equity_history, periods=int(getattr(BacktestConfig, "ANNUAL_PERIODS", 252)))
    payload = {
        "metrics": metrics,
        "round_trips": int(round_trips),
        "fills": int(fills),
        "final_equity": float(equity_history[-1]) if equity_history else 10_000.0,
        "reject_counts": dict(sorted(reject_counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
        "liquidations": 0,
    }
    if include_equity:
        payload["equity_history"] = [float(item) for item in equity_history]
    return payload


def _rv_caps(arrays: dict[str, Any], symbols: list[str]) -> list[float]:
    values: list[float] = []
    for symbol in symbols:
        prefix = _symbol_prefix(symbol)
        rv = arrays[f"{prefix}_rv_24h"]
        finite = rv[np.isfinite(rv)]
        if finite.size:
            values.extend([float(np.quantile(finite, q)) for q in (0.55, 0.70, 0.85)])
    return sorted({round(v, 8) for v in values if v > 0.0})[:9]


def _candidate_specs(arrays: dict[str, Any], symbols: list[str]) -> list[FreshSpec]:
    specs: dict[str, FreshSpec] = {}
    rv_caps = _rv_caps(arrays, symbols)
    session_sets = {
        "all": (),
        "asia_us": (0, 1, 2, 13, 14, 15, 16, 20, 21),
        "post_funding": (2, 3, 4, 10, 11, 12, 18, 19, 20),
    }
    for lookback in (6, 12, 24, 48, 72):
        for threshold in (1.25, 1.5, 1.75, 2.0, 2.5):
            for hold in (6, 12, 24, 48, 72):
                for session_label, hours in session_sets.items():
                    name = f"fresh_resid_rev_lb{lookback}_z{str(threshold).replace('.', '')}_h{hold}_{session_label}"
                    specs[name] = FreshSpec(
                        name=name,
                        family="residual_reversion",
                        lookback_bars=lookback,
                        threshold=threshold,
                        hold_bars=hold,
                        cooldown_bars=max(0, hold // 4),
                        stop_loss_pct=0.018,
                        take_profit_pct=0.035,
                        min_abs_return=0.0025 if lookback <= 12 else 0.004,
                        entry_hours=hours,
                    )
        for threshold in (1.0, 1.25, 1.5, 1.75):
            for hold in (12, 24, 48, 96):
                name = f"fresh_xs_mom_lb{lookback}_z{str(threshold).replace('.', '')}_h{hold}"
                specs[name] = FreshSpec(
                    name=name,
                    family="cross_momentum",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=hold,
                    cooldown_bars=max(0, hold // 3),
                    stop_loss_pct=0.015,
                    take_profit_pct=0.045,
                    broad_min_abs=0.003 if lookback <= 24 else 0.006,
                    min_abs_return=0.004,
                    allow_short=True,
                )
    for cap in rv_caps:
        for lookback in (12, 24, 48):
            for threshold in (1.25, 1.5, 1.75):
                name = f"fresh_resid_rev_rvcap{int(cap*1e6)}_lb{lookback}_z{str(threshold).replace('.', '')}"
                specs[name] = FreshSpec(
                    name=name,
                    family="residual_reversion",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=48,
                    cooldown_bars=12,
                    stop_loss_pct=0.015,
                    take_profit_pct=0.035,
                    rv_lookback_bars=24,
                    max_rv=cap,
                    min_abs_return=0.003,
                )
    for lookback in (12, 24, 48, 72):
        for threshold in (1.0, 1.25, 1.5):
            for funding_min in (0.00005, 0.00010, 0.00015):
                name = f"fresh_funding_fade_lb{lookback}_z{str(threshold).replace('.', '')}_f{int(funding_min*1e6)}ppm"
                specs[name] = FreshSpec(
                    name=name,
                    family="funding_carry_fade",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=24,
                    cooldown_bars=12,
                    stop_loss_pct=0.018,
                    take_profit_pct=0.035,
                    funding_rank_min=funding_min,
                    min_abs_return=0.003,
                    entry_hours=(1, 2, 9, 10, 17, 18),
                )
    for flow_lookback in (3, 6, 12, 24):
        for price_lookback in (3, 6, 12, 24):
            for flow_threshold in (0.03, 0.06, 0.10, 0.15):
                for hold in (6, 12, 24, 48):
                    name = (
                        f"fresh_flow_mom_fl{flow_lookback}_px{price_lookback}_"
                        f"imb{int(flow_threshold * 100)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="flow_momentum",
                        lookback_bars=price_lookback,
                        threshold=0.0,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 4),
                        stop_loss_pct=0.012,
                        take_profit_pct=0.030,
                        min_abs_return=0.0015 if price_lookback <= 6 else 0.003,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                    )
                    name = (
                        f"fresh_flow_exhaust_fl{flow_lookback}_px{price_lookback}_"
                        f"imb{int(flow_threshold * 100)}_h{hold}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="flow_exhaustion_fade",
                        lookback_bars=price_lookback,
                        threshold=0.0,
                        hold_bars=hold,
                        cooldown_bars=max(1, hold // 3),
                        stop_loss_pct=0.014,
                        take_profit_pct=0.028,
                        min_abs_return=0.0015 if price_lookback <= 6 else 0.003,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                    )
    for flow_lookback in (3, 6, 12):
        for price_lookback in (6, 12, 24, 48):
            for threshold in (1.25, 1.5, 1.75):
                for flow_threshold in (0.03, 0.06, 0.10):
                    name = (
                        f"fresh_resid_flow_rev_fl{flow_lookback}_lb{price_lookback}_"
                        f"z{str(threshold).replace('.', '')}_imb{int(flow_threshold * 100)}"
                    )
                    specs[name] = FreshSpec(
                        name=name,
                        family="residual_reversion_flow_confirmed",
                        lookback_bars=price_lookback,
                        threshold=threshold,
                        hold_bars=24,
                        cooldown_bars=8,
                        stop_loss_pct=0.014,
                        take_profit_pct=0.032,
                        min_abs_return=0.0025 if price_lookback <= 12 else 0.004,
                        flow_lookback_bars=flow_lookback,
                        flow_threshold=flow_threshold,
                    )
    for lookback in (6, 12, 24):
        for threshold in (0.006, 0.010, 0.014):
            for comp in (0.55, 0.70, 0.85):
                name = f"fresh_compression_breakout_lb{lookback}_thr{int(threshold*10000)}_c{int(comp*100)}"
                specs[name] = FreshSpec(
                    name=name,
                    family="compression_breakout",
                    lookback_bars=lookback,
                    threshold=threshold,
                    hold_bars=24,
                    cooldown_bars=12,
                    stop_loss_pct=0.012,
                    take_profit_pct=0.030,
                    rv_lookback_bars=24,
                    compression_lookback_bars=72,
                    compression_quantile=comp,
                    allow_short=True,
                )
    return list(specs.values())


def _evaluate_specs(*, specs: list[FreshSpec], arrays: dict[str, Any], splits: list[SplitWindow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for spec in specs:
        split_results = {split.name: _run_split(spec=spec, arrays=arrays, split=split) for split in splits}
        metrics = {name: dict(result.get("metrics") or {}) for name, result in split_results.items()}
        train = metrics.get("train", {})
        val = metrics.get("val", {})
        oos = metrics.get("oos", {})
        gates = {
            "not_incumbent_or_context_wrapper": not spec.name.startswith("profit_moonshot_hourly_shock"),
            "train_positive": _safe_float(train.get("total_return"), 0.0) > 0.0,
            "val_positive": _safe_float(val.get("total_return"), 0.0) > 0.0,
            "oos_return_beats_incumbent": _safe_float(oos.get("total_return"), 0.0) > BASELINE_OOS_RETURN,
            "oos_mdd_beats_shadow": _safe_float(oos.get("max_drawdown"), 1.0) < SHADOW_OOS_MDD,
            "oos_sharpe_gt_1": _safe_float(oos.get("sharpe"), 0.0) > SUCCESS_SHARPE,
            "oos_trades_not_starved": int(split_results["oos"].get("round_trips") or 0) >= 5,
            "liquidations_zero": int(split_results["train"].get("liquidations") or 0) == 0
            and int(split_results["val"].get("liquidations") or 0) == 0
            and int(split_results["oos"].get("liquidations") or 0) == 0,
        }
        replay_survivor = all(
            gates[key]
            for key in (
                "not_incumbent_or_context_wrapper",
                "train_positive",
                "val_positive",
                "oos_return_beats_incumbent",
                "oos_mdd_beats_shadow",
                "oos_trades_not_starved",
                "liquidations_zero",
            )
        )
        success = bool(replay_survivor and gates["oos_sharpe_gt_1"])
        failed = [key for key, ok in gates.items() if not ok]
        row: dict[str, Any] = {
            "name": spec.name,
            "family": spec.family,
            "replay_survivor": bool(replay_survivor),
            "success_candidate": success,
            "failed_gates": ",".join(failed),
            "filters": json.dumps(spec.payload(), sort_keys=True),
            "reject_top": json.dumps(split_results["oos"].get("reject_counts") or {}, sort_keys=True),
        }
        for split_name, result in split_results.items():
            split_metrics = dict(result.get("metrics") or {})
            for key in ("total_return", "max_drawdown", "sharpe", "sortino", "volatility"):
                row[f"{split_name}_{key}"] = _safe_float(split_metrics.get(key), 0.0)
            row[f"{split_name}_round_trips"] = int(result.get("round_trips") or 0)
            row[f"{split_name}_fills"] = int(result.get("fills") or 0)
            row[f"{split_name}_liquidations"] = int(result.get("liquidations") or 0)
        rows.append(row)
        results.append(
            {
                "name": spec.name,
                "family": spec.family,
                "filters": spec.payload(),
                "split_results": split_results,
                "gates": gates,
                "failed_gates": failed,
                "replay_survivor": bool(replay_survivor),
                "success_candidate": success,
            }
        )
    rows.sort(
        key=lambda row: (
            not bool(row.get("success_candidate")),
            not bool(row.get("replay_survivor")),
            -_safe_float(row.get("oos_total_return"), 0.0),
            -_safe_float(row.get("oos_sharpe"), 0.0),
            _safe_float(row.get("oos_max_drawdown"), 1.0),
        )
    )
    by_name = {item["name"]: item for item in results}
    results = [by_name[str(row["name"])] for row in rows]
    return rows, results


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    top = rows[:15]
    survivors = [r for r in rows if bool(r.get("replay_survivor"))]
    success = [r for r in rows if bool(r.get("success_candidate"))]
    lines = [
        "# Profit moonshot fresh-start overhaul replay",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"OOS end date: `{payload['oos_end_date']}`",
        "",
        "## Intent",
        "",
        "- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.",
        "- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, funding-carry fade, taker-flow momentum/exhaustion, compression breakout.",
        "- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.",
        "",
        "## Gate policy",
        "",
        f"- Success requires OOS return > `{BASELINE_OOS_RETURN:+.4%}`, OOS MDD < `{SHADOW_OOS_MDD:.4%}`, OOS Sharpe > `{SUCCESS_SHARPE:.1f}`, liquidations `0`, and positive train/val.",
        "- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.",
        "",
        "## Result",
        "",
        f"- Specs evaluated: `{len(rows)}`",
        f"- Replay survivors: `{len(survivors)}`",
        f"- Success candidates: `{len(success)}`",
        f"- Peak RSS: `{payload['peak_rss_mib']:.3f} MiB`",
        "",
        "## Top candidates/failures",
        "",
        "| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(
            f"| {idx} | `{row['name']}` | `{row['family']}` | {bool(row.get('replay_survivor'))} | {bool(row.get('success_candidate'))} | "
            f"{_fmt_pct(row.get('train_total_return'))} | {_fmt_pct(row.get('val_total_return'))} | {_fmt_pct(row.get('oos_total_return'))} | "
            f"{_fmt_pct(row.get('oos_max_drawdown'))} | {_fmt_float(row.get('oos_sharpe'))} | {int(row.get('oos_round_trips') or 0)} | `{row.get('failed_gates')}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if success:
        lines.append("- At least one fresh-start replay candidate earned a one-at-a-time full live-equivalent backtest slot.")
    elif survivors:
        lines.append("- Replay survivors exist, but none is a final success until live-equivalent raw-first backtest proves Sharpe > 1.0 and all gates.")
    else:
        lines.append("- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.")
    lines.extend(
        [
            "- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    oos_end = datetime.fromisoformat(str(args.oos_end_date)).date() if str(args.oos_end_date or "").strip() else date(2026, 5, 6)
    splits = _split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = _joined_panel(market_root=Path(args.market_root), exchange=str(args.exchange), symbols=symbols, start=start, end=end)
    arrays = _build_arrays(panel, symbols)
    specs = _candidate_specs(arrays, symbols)
    rows, results = _evaluate_specs(specs=specs, arrays=arrays, splits=splits)
    payload = {
        "artifact_kind": "profit_moonshot_fresh_start_overhaul_replay",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "market_root": str(args.market_root),
        "exchange": str(args.exchange),
        "symbols": symbols,
        "oos_end_date": oos_end.isoformat(),
        "split_windows": [split.as_payload() for split in splits],
        "data_metadata": data_metadata,
        "gate_policy": {
            "baseline_oos_return": BASELINE_OOS_RETURN,
            "shadow_oos_mdd": SHADOW_OOS_MDD,
            "success_oos_sharpe": SUCCESS_SHARPE,
            "target_allocation": TARGET_ALLOCATION,
            "max_order_value": MAX_ORDER_VALUE,
            "liquidations_required": 0,
        },
        "spec_count": len(specs),
        "replay_survivor_count": sum(1 for row in rows if bool(row.get("replay_survivor"))),
        "success_candidate_count": sum(1 for row in rows if bool(row.get("success_candidate"))),
        "top_results": results[:25],
        "peak_rss_mib": _rss_mib(),
    }
    return payload, rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, rows = build_payload(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fresh_start_overhaul_replay_latest.json"
    md_path = output_dir / "fresh_start_overhaul_replay_latest.md"
    csv_path = output_dir / "fresh_start_overhaul_replay_candidates.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    md_path.write_text(_markdown(payload, rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
                "spec_count": payload["spec_count"],
                "replay_survivor_count": payload["replay_survivor_count"],
                "success_candidate_count": payload["success_candidate_count"],
                "peak_rss_mib": payload["peak_rss_mib"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
