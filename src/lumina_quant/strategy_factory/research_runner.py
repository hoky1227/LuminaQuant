"""Candidate research runner with train/val/oos evaluation and robust metrics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import queue
import random
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.config import BacktestConfig
from lumina_quant.market_data import load_futures_feature_points_from_db
from lumina_quant.storage.parquet import load_data_dict_from_parquet
from lumina_quant.symbols import (
    CANONICAL_STRATEGY_TIMEFRAMES,
    canonical_symbol,
    canonicalize_symbol_list,
    normalize_strategy_timeframes,
)
from lumina_quant.strategy_factory.runtime_settings import (
    current_research_market_data_settings as _current_research_market_data_settings_impl,
)
from lumina_quant.strategy_factory.research_reporting import ResearchReportBuilder
from lumina_quant.strategy_factory.research_resources import ResearchResourceLoader
from lumina_quant.strategy_factory.research_stage_selection import ResearchStageSelector
from lumina_quant.strategy_factory.strategy_signal_dispatch import StrategySignalDispatcher
from lumina_quant.utils.risk_free import (
    resolve_risk_free_config,
    sharpe_ratio as compute_sharpe_ratio,
    sortino_ratio as compute_sortino_ratio,
)

_PERIODS_PER_YEAR = {
    "1s": 31_536_000,
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}

_MIN_BARS = 360
_METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}
_LEADERS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT")
_FEATURE_POINT_COLUMNS: tuple[str, ...] = (
    "funding_rate",
    "funding_mark_price",
    "funding_fee_rate",
    "funding_fee_quote_per_unit",
    "mark_price",
    "index_price",
    "open_interest",
    "liquidation_long_qty",
    "liquidation_short_qty",
    "liquidation_long_notional",
    "liquidation_short_notional",
)


def _current_research_market_data_settings(
    runtime_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _current_research_market_data_settings_impl(runtime_settings)


DEFAULT_RESEARCH_SCORING_CONFIG: dict[str, Any] = {
    "stage1_prefilter_weights": {
        "sharpe_weight": 2.0,
        "return_weight": 20.0,
        "pbo_penalty": 2.0,
    },
    "candidate_rank_score_weights": {
        "sharpe_weight": 2.8,
        "deflated_sharpe_weight": 1.4,
        "pbo_penalty": 2.0,
        "return_weight": 35.0,
        "turnover_penalty": 2.5,
        "drawdown_penalty": 3.0,
        "turnover_threshold": 2.5,
        "instability_sharpe_penalty": 0.75,
        "instability_return_penalty": 35.0,
        "instability_turnover_penalty": 1.0,
    },
    "hurdle_score_weights": {
        "sharpe_weight": 2.4,
        "return_weight": 35.0,
        "deflated_sharpe_weight": 1.2,
        "pbo_penalty": 2.0,
        "turnover_penalty": 4.0,
        "drawdown_penalty": 5.0,
        "spa_pvalue_penalty": 1.0,
    },
    "reject_thresholds": {
        "in_sample_sharpe_min": -0.1,
        "oos_sharpe_min": 0.35,
        "max_pbo": 0.45,
        "max_turnover": 2.5,
        "max_drawdown": 0.45,
        "min_trade_count": 5.0,
    },
    "cost_stress_thresholds": {
        "x2_sharpe_min": 0.0,
        "x3_sharpe_min": -0.25,
    },
    "keep_ratio_bounds": {
        "min": 0.05,
        "max": 1.0,
    },
    "score_fallbacks": {
        "stage1_error_score": -1_000_000.0,
        "failed_candidate_selection_score": -1_000_000.0,
        "sort_missing_selection_score": -1_000_000.0,
    },
}


def _resolve_feature_points_path() -> Path:
    candidates: list[Path] = []
    defaults = _current_research_market_data_settings()

    for raw in (
        os.getenv("LQ_MARKET_PARQUET_PATH", ""),
        defaults["parquet_root"],
        "data/market_parquet",
    ):
        token = str(raw or "").strip()
        if not token:
            continue
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        candidates.append(path / "feature_points")

    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        candidates.append(parent / "data" / "market_parquet" / "feature_points")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return candidates[0].resolve() if candidates else (Path.cwd() / "data" / "market_parquet" / "feature_points").resolve()


def _resolve_score_config(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_RESEARCH_SCORING_CONFIG)
    if not isinstance(overrides, Mapping):
        return resolved
    for key, default_value in resolved.items():
        override_value = overrides.get(key)
        if isinstance(default_value, dict) and isinstance(override_value, Mapping):
            for sub_key in default_value:
                if sub_key in override_value:
                    default_value[sub_key] = override_value[sub_key]
        elif override_value is not None and not isinstance(default_value, dict):
            resolved[key] = override_value
    return resolved


@dataclass(slots=True)
class SeriesBundle:
    symbol: str
    timeframe: str
    datetime: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


class _AlignedStrategyBarStore:
    def __init__(self, symbols: Sequence[str]):
        self.symbol_list = canonicalize_symbol_list(list(symbols))
        self._rows = {
            symbol: {
                "datetime": None,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
            }
            for symbol in self.symbol_list
        }

    def set_bar(
        self,
        symbol: str,
        event_time: Any,
        *,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
    ) -> None:
        token = canonical_symbol(str(symbol))
        if token not in self._rows:
            return
        self._rows[token] = {
            "datetime": event_time,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
        }

    def get_latest_bar_value(self, symbol: str, value_type: str) -> float:
        token = canonical_symbol(str(symbol))
        row = self._rows.get(token) or {}
        value = row.get(str(value_type), 0.0)
        return float(value) if value is not None else 0.0

    def get_latest_bar_datetime(self, symbol: str) -> Any:
        token = canonical_symbol(str(symbol))
        row = self._rows.get(token) or {}
        return row.get("datetime")


def _simulate_event_driven_strategy_exposures(
    strategy_cls: type[Any],
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
) -> np.ndarray:
    from lumina_quant.core.events import MarketEvent

    canonical_symbols = canonicalize_symbol_list(list(symbols))
    datetimes = np.asarray(aligned.get("datetime"), dtype=object)
    n = len(datetimes)
    exposures = np.zeros((len(canonical_symbols), n), dtype=float)
    bars = _AlignedStrategyBarStore(canonical_symbols)
    events: queue.Queue[Any] = queue.Queue()
    strategy = strategy_cls(bars, events, **dict(params))
    position_state = dict.fromkeys(canonical_symbols, 0.0)
    symbol_index = {symbol: idx for idx, symbol in enumerate(canonical_symbols)}

    for idx in range(n):
        event_time = datetimes[idx]
        for symbol in canonical_symbols:
            bars.set_bar(
                symbol,
                event_time,
                open_price=float(aligned[f"{symbol}:open"][idx]),
                high_price=float(aligned[f"{symbol}:high"][idx]),
                low_price=float(aligned[f"{symbol}:low"][idx]),
                close_price=float(aligned[f"{symbol}:close"][idx]),
                volume=float(aligned[f"{symbol}:volume"][idx]),
            )

        for symbol in canonical_symbols:
            strategy.calculate_signals(
                MarketEvent(
                    time=event_time,
                    symbol=symbol,
                    open=float(aligned[f"{symbol}:open"][idx]),
                    high=float(aligned[f"{symbol}:high"][idx]),
                    low=float(aligned[f"{symbol}:low"][idx]),
                    close=float(aligned[f"{symbol}:close"][idx]),
                    volume=float(aligned[f"{symbol}:volume"][idx]),
                )
            )

        while not events.empty():
            signal = events.get()
            token = canonical_symbol(str(getattr(signal, "symbol", "")))
            if token not in position_state:
                continue
            signal_type = str(getattr(signal, "signal_type", "")).upper()
            if signal_type == "LONG":
                position_state[token] = 1.0
            elif signal_type == "SHORT":
                position_state[token] = -1.0
            elif signal_type == "EXIT":
                position_state[token] = 0.0

        for symbol, value in position_state.items():
            exposures[symbol_index[symbol], idx] = float(value)

    return exposures


def _load_event_driven_strategy_impl(strategy_class: str) -> type[Any]:
    if strategy_class == "Alpha101FormulaStrategy":
        from lumina_quant.strategies.alpha101_formula import Alpha101FormulaStrategy

        return Alpha101FormulaStrategy
    if strategy_class == "PairTradingZScoreStrategy":
        from lumina_quant.strategies.pair_trading_zscore import PairTradingZScoreStrategy

        return PairTradingZScoreStrategy
    if strategy_class == "PairSpreadZScoreStrategy":
        from lumina_quant.strategies.pair_spread_zscore import PairSpreadZScoreStrategy

        return PairSpreadZScoreStrategy
    msg = f"Unsupported event-driven proxy strategy: {strategy_class}"
    raise ValueError(msg)


def _resolve_symbol_pair(
    symbols: Sequence[str],
    params: Mapping[str, Any],
) -> tuple[str, str, int, int]:
    symbol_x = canonical_symbol(str(params.get("symbol_x") or symbols[0]))
    symbol_y = canonical_symbol(str(params.get("symbol_y") or symbols[1]))
    if symbol_x not in symbols:
        symbol_x = symbols[0]
    if symbol_y not in symbols:
        symbol_y = symbols[1]
    return symbol_x, symbol_y, symbols.index(symbol_x), symbols.index(symbol_y)


def _pair_spread_fallback_exposures(
    *,
    aligned: Mapping[str, np.ndarray],
    symbol_x: str,
    symbol_y: str,
    length: int,
    entry_z: float,
    exit_z: float,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.nan_to_num(
        _pair_spread_z(
            aligned[f"{symbol_x}:close"],
            aligned[f"{symbol_y}:close"],
            window=lookback,
        ),
        nan=0.0,
    )

    long_spread = z <= -entry_z
    short_spread = z >= entry_z
    exit_cond = np.abs(z) <= exit_z

    x_pos = np.zeros(length, dtype=float)
    y_pos = np.zeros(length, dtype=float)
    x_pos = np.where(long_spread, -1.0, x_pos)
    y_pos = np.where(long_spread, 1.0, y_pos)
    x_pos = np.where(short_spread, 1.0, x_pos)
    y_pos = np.where(short_spread, -1.0, y_pos)
    x_pos = np.where(exit_cond, 0.0, x_pos)
    y_pos = np.where(exit_cond, 0.0, y_pos)
    return x_pos, y_pos


def _load_feature_cache(
    *,
    symbols: Sequence[str],
    start_date: Any = None,
    end_date: Any = None,
    market_data_settings: Mapping[str, Any] | None = None,
) -> dict[str, pl.DataFrame]:
    defaults = _current_research_market_data_settings(market_data_settings)
    db_path = str(defaults["parquet_root"])
    exchange = str(defaults["exchange"])
    cache: dict[str, pl.DataFrame] = {}

    for symbol in canonicalize_symbol_list(symbols):
        frame = _load_feature_frame(
            db_path=db_path,
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        cache[symbol] = _normalize_feature_frame(frame)

    return cache


def _load_feature_frame(
    *,
    db_path: str,
    exchange: str,
    symbol: str,
    start_date: Any,
    end_date: Any,
) -> pl.DataFrame:
    try:
        return load_futures_feature_points_from_db(
            db_path,
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return pl.DataFrame()


def _normalize_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "timestamp_ms" not in frame.columns:
        return pl.DataFrame()

    cleaned = frame.filter(pl.col("timestamp_ms").is_not_null()).with_columns(
        pl.col("timestamp_ms").cast(pl.Int64)
    )
    if cleaned.is_empty():
        return pl.DataFrame()

    for field in _FEATURE_POINT_COLUMNS:
        if field not in cleaned.columns:
            cleaned = cleaned.with_columns(pl.lit(None, dtype=pl.Float64).alias(field))

    cleaned = cleaned.select(["timestamp_ms", *_FEATURE_POINT_COLUMNS]).sort("timestamp_ms").unique(
        subset=["timestamp_ms"],
        keep="last",
    )
    return cleaned.with_columns(
        [
            pl.col("timestamp_ms").cast(pl.Int64),
            pl.from_epoch("timestamp_ms", time_unit="ms").alias("datetime"),
            *[
                pl.col(field).cast(pl.Float64).fill_null(strategy="forward").alias(field)
                for field in _FEATURE_POINT_COLUMNS
            ],
        ]
    )


def _crowding_support_series(
    *,
    funding_rate: np.ndarray,
    open_interest: np.ndarray,
    mark_price: np.ndarray | None = None,
    index_price: np.ndarray | None = None,
    liquidation_long_notional: np.ndarray | None = None,
    liquidation_short_notional: np.ndarray | None = None,
    window: int = 96,
) -> dict[str, np.ndarray]:
    n = int(funding_rate.size)
    empty = np.full(n, np.nan, dtype=float)
    if n <= 0:
        return {
            "crowding_score": empty,
            "funding_z": empty,
            "oi_delta_z": empty,
            "basis_z": empty,
            "liquidation_imbalance_z": empty,
        }

    funding = np.asarray(funding_rate, dtype=float)
    oi = np.asarray(open_interest, dtype=float)
    mark = None if mark_price is None else np.asarray(mark_price, dtype=float)
    index = None if index_price is None else np.asarray(index_price, dtype=float)
    long_liq = (
        np.zeros(n, dtype=float)
        if liquidation_long_notional is None
        else np.nan_to_num(np.asarray(liquidation_long_notional, dtype=float), nan=0.0)
    )
    short_liq = (
        np.zeros(n, dtype=float)
        if liquidation_short_notional is None
        else np.nan_to_num(np.asarray(liquidation_short_notional, dtype=float), nan=0.0)
    )

    oi_prev = np.roll(oi, 1)
    oi_prev[0] = np.nan
    oi_delta = np.full(n, np.nan, dtype=float)
    oi_mask = (
        np.isfinite(oi)
        & np.isfinite(oi_prev)
        & (oi > 0.0)
        & (oi_prev > 0.0)
    )
    oi_delta[oi_mask] = np.log(oi[oi_mask] / oi_prev[oi_mask])

    basis = np.full(n, np.nan, dtype=float)
    if mark is not None and index is not None:
        basis_mask = np.isfinite(mark) & np.isfinite(index) & (np.abs(index) > 1e-12)
        basis[basis_mask] = (mark[basis_mask] - index[basis_mask]) / index[basis_mask]

    liq_den = np.abs(long_liq) + np.abs(short_liq) + 1e-12
    liq_imbalance = (long_liq - short_liq) / liq_den

    funding_z = _rolling_z(np.nan_to_num(funding, nan=0.0), max(16, int(window)))
    oi_delta_z = _rolling_z(np.nan_to_num(oi_delta, nan=0.0), max(16, int(window // 2)))
    basis_z = _rolling_z(np.nan_to_num(basis, nan=0.0), max(12, int(window // 2)))
    liq_z = _rolling_z(np.nan_to_num(liq_imbalance, nan=0.0), max(12, int(window // 2)))

    valid = np.isfinite(funding) & np.isfinite(oi)
    score = np.tanh(
        (0.45 * np.nan_to_num(funding_z, nan=0.0))
        + (0.35 * np.nan_to_num(oi_delta_z, nan=0.0))
        + (0.15 * np.nan_to_num(basis_z, nan=0.0))
        + (0.05 * np.nan_to_num(liq_z, nan=0.0))
    )
    score = np.where(valid, score, np.nan)
    return {
        "crowding_score": score.astype(float, copy=False),
        "funding_z": np.where(valid, funding_z, np.nan),
        "oi_delta_z": np.where(valid, oi_delta_z, np.nan),
        "basis_z": np.where(valid, basis_z, np.nan),
        "liquidation_imbalance_z": np.where(valid, liq_z, np.nan),
    }


def _hash_unit_interval(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)


def _hash_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _candidate_identity(candidate: dict[str, Any]) -> str:
    payload = {
        "name": str(candidate.get("name", "")),
        "strategy_class": str(candidate.get("strategy_class", "")),
        "strategy_timeframe": str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""),
        "symbols": list(candidate.get("symbols") or []),
        "params": dict(candidate.get("params") or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def adapt_legacy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Adapt legacy candidate fields to v2 contract without data loss."""
    row = dict(candidate)

    if not row.get("strategy_timeframe") and row.get("timeframe"):
        row["strategy_timeframe"] = str(row.get("timeframe"))
    if not row.get("timeframe") and row.get("strategy_timeframe"):
        row["timeframe"] = str(row.get("strategy_timeframe"))

    if not row.get("strategy_class") and row.get("strategy"):
        row["strategy_class"] = str(row.get("strategy"))
    if not row.get("strategy") and row.get("strategy_class"):
        row["strategy"] = str(row.get("strategy_class"))

    row["strategy_timeframe"] = str(row.get("strategy_timeframe") or "1m").strip().lower()
    row["timeframe"] = str(row.get("timeframe") or row["strategy_timeframe"]).strip().lower()

    symbols = canonicalize_symbol_list(list(row.get("symbols") or []))
    row["symbols"] = symbols

    if not row.get("candidate_id"):
        row["candidate_id"] = _candidate_identity(row)

    row["params"] = dict(row.get("params") or {})
    row["name"] = str(row.get("name") or row.get("candidate_id"))
    row["strategy_class"] = str(row.get("strategy_class") or row.get("strategy") or "")
    row["strategy"] = str(row.get("strategy") or row.get("strategy_class") or "")
    row["family"] = str(row.get("family") or _family_from_strategy(row["strategy_class"]))
    return row


def _family_from_strategy(strategy_class: str) -> str:
    token = str(strategy_class).strip().lower()
    if "composite" in token or "trend" in token:
        return "trend"
    if "vwap" in token or "reversion" in token:
        return "mean_reversion"
    if "leadlag" in token:
        return "intraday_alpha"
    if "pair" in token:
        return "market_neutral"
    if "perp" in token or "carry" in token:
        return "carry"
    if "micro" in token:
        return "micro"
    return "other"


def _coerce_utc_datetime(value: Any, *, end_of_day: bool = False) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, np.datetime64):
        epoch_ms = int(value.astype("datetime64[ms]").astype(np.int64))
        dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
    elif isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            numeric *= 1000
        dt = datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
    else:
        text = str(value).strip()
        if not text:
            return None
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            dt = datetime.fromisoformat(text)
            if end_of_day:
                dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _datetime_to_iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _to_numpy_datetime64_ms(value: datetime | None) -> np.datetime64 | None:
    if value is None:
        return None
    return np.datetime64(value.astimezone(UTC).replace(tzinfo=None), "ms")


def _split_window_bounds(split: Mapping[str, Any] | None) -> tuple[datetime | None, datetime | None]:
    if not isinstance(split, Mapping):
        return None, None
    starts = [
        _coerce_utc_datetime(split.get("train_start")),
        _coerce_utc_datetime(split.get("val_start")),
        _coerce_utc_datetime(split.get("oos_start") or split.get("test_start")),
    ]
    ends = [
        _coerce_utc_datetime(split.get("train_end"), end_of_day=True),
        _coerce_utc_datetime(split.get("val_end"), end_of_day=True),
        _coerce_utc_datetime(split.get("oos_end") or split.get("test_end"), end_of_day=True),
    ]
    valid_starts = [item for item in starts if item is not None]
    valid_ends = [item for item in ends if item is not None]
    return (
        min(valid_starts) if valid_starts else None,
        max(valid_ends) if valid_ends else None,
    )


def _resolve_split_config(
    split: Mapping[str, Any] | None,
    *,
    strategy_timeframe: str,
) -> dict[str, Any]:
    resolved = dict(split) if isinstance(split, Mapping) else _build_default_split(strategy_timeframe)
    train_start = _coerce_utc_datetime(resolved.get("train_start"))
    train_end = _coerce_utc_datetime(resolved.get("train_end"), end_of_day=True)
    val_start = _coerce_utc_datetime(resolved.get("val_start"))
    val_end = _coerce_utc_datetime(resolved.get("val_end"), end_of_day=True)
    oos_start = _coerce_utc_datetime(resolved.get("oos_start") or resolved.get("test_start"))
    oos_end = _coerce_utc_datetime(
        resolved.get("oos_end") or resolved.get("test_end"),
        end_of_day=True,
    )
    return {
        **resolved,
        "train_start": _datetime_to_iso_z(train_start),
        "train_end": _datetime_to_iso_z(train_end),
        "val_start": _datetime_to_iso_z(val_start),
        "val_end": _datetime_to_iso_z(val_end),
        "oos_start": _datetime_to_iso_z(oos_start),
        "oos_end": _datetime_to_iso_z(oos_end),
        "strategy_timeframe": str(
            resolved.get("strategy_timeframe") or resolved.get("timeframe") or strategy_timeframe
        ),
        "mode": str(resolved.get("mode") or ("exact_dates" if isinstance(split, Mapping) else "rolling_default")),
    }


def _split_lengths(total: int, *, train_frac: float = 0.60, val_frac: float = 0.20) -> tuple[slice, slice, slice]:
    if total <= 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(total - 1, val_end)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, total)


def _safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    out = float(np.std(values, ddof=1))
    if not math.isfinite(out):
        return 0.0
    return out


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    out = float(np.mean(values))
    if not math.isfinite(out):
        return 0.0
    return out


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    return float(np.max(drawdown)) if drawdown.size else 0.0


def _rolling_sharpe_min(returns: np.ndarray, *, window: int = 64, periods_per_year: int = 365) -> float:
    if returns.size < max(8, window):
        return 0.0
    vals: list[float] = []
    for idx in range(window, returns.size + 1):
        tail = returns[idx - window : idx]
        mu = _safe_mean(tail)
        sigma = _safe_std(tail)
        if sigma <= 1e-12:
            continue
        vals.append((mu / sigma) * math.sqrt(periods_per_year))
    if not vals:
        return 0.0
    return float(min(vals))


def _worst_month(returns: np.ndarray, *, bars_per_month: int) -> float:
    if returns.size == 0:
        return 0.0
    bars = max(4, int(bars_per_month))
    monthly: list[float] = []
    for idx in range(0, returns.size, bars):
        tail = returns[idx : idx + bars]
        if tail.size == 0:
            continue
        monthly.append(float(np.prod(1.0 + tail) - 1.0))
    if not monthly:
        return 0.0
    return float(min(monthly))


def _deflated_sharpe_ratio(returns: np.ndarray, *, num_trials: int = 1) -> float:
    if returns.size < 16:
        return 0.0
    mu = _safe_mean(returns)
    sigma = _safe_std(returns)
    if sigma <= 1e-12:
        return 0.0

    sharpe = mu / sigma
    n = float(max(2, returns.size))
    k = float(max(1, num_trials))
    expected_max = math.sqrt(2.0 * math.log(k)) / math.sqrt(n)

    centered = returns - mu
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = 0.0 if sigma <= 1e-12 else m3 / (sigma**3)
    kurt = 3.0 if sigma <= 1e-12 else m4 / (sigma**4)

    denom_term = 1.0 - (skew * sharpe) + (((kurt - 1.0) / 4.0) * (sharpe**2))
    denom_term = max(1e-8, denom_term)
    denom = math.sqrt(denom_term / max(1.0, n - 1.0))
    z = (sharpe - expected_max) / max(1e-8, denom)

    # Normal CDF.
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, cdf)))


def _approx_pbo(returns: np.ndarray) -> float:
    """Approximate probability of backtest overfitting from fold instability."""
    n = returns.size
    if n < 64:
        return 1.0
    folds = min(8, max(4, n // 32))
    fold_size = n // folds
    if fold_size <= 0:
        return 1.0

    failures = 0
    trials = 0
    for idx in range(folds):
        test_start = idx * fold_size
        test_end = n if idx == folds - 1 else (idx + 1) * fold_size
        test = returns[test_start:test_end]
        train = np.concatenate((returns[:test_start], returns[test_end:]))
        if train.size < 8 or test.size < 8:
            continue
        train_sharpe = 0.0
        test_sharpe = 0.0
        train_std = _safe_std(train)
        test_std = _safe_std(test)
        if train_std > 1e-12:
            train_sharpe = _safe_mean(train) / train_std
        if test_std > 1e-12:
            test_sharpe = _safe_mean(test) / test_std
        trials += 1
        if train_sharpe > 0.0 and test_sharpe <= 0.0:
            failures += 1
    if trials <= 0:
        return 1.0
    return float(failures / trials)


def _spa_like_pvalue(returns: np.ndarray, *, bootstrap_rounds: int = 200) -> float:
    """Simple bootstrap p-value proxy for data-snooping correction."""
    if returns.size < 16:
        return 1.0
    observed = _safe_mean(returns)
    if observed <= 0.0:
        return 1.0

    rng = np.random.default_rng(12345)
    exceed = 0
    centered = returns - _safe_mean(returns)
    n = centered.size
    for _ in range(max(64, int(bootstrap_rounds))):
        idx = rng.integers(0, n, size=n)
        sample = centered[idx]
        if _safe_mean(sample) >= observed:
            exceed += 1
    return float(exceed / max(1, int(bootstrap_rounds)))


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    n = min(x.size, y.size)
    if n < 8:
        return 0.0
    xa = x[-n:]
    ya = y[-n:]
    xs = _safe_std(xa)
    ys = _safe_std(ya)
    if xs <= 1e-12 or ys <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(xa, ya)[0, 1])
    if not math.isfinite(corr):
        return 0.0
    return corr


def _compute_metrics(
    returns: np.ndarray,
    *,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
    num_trials: int,
    metric_config: Any | None = None,
    timestamps: np.ndarray | None = None,
) -> dict[str, float]:
    if returns.size == 0:
        return _compute_metric_summary(
            _empty_compute_metric_payload(),
            resolved_rf=None,
        )

    resolved_rf = resolve_risk_free_config(
        metric_config or BacktestConfig,
        periods_per_year=periods_per_year,
        timestamps=timestamps,
        size=int(returns.size),
    )
    metric_payload = _resolve_compute_metric_payload(
        returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        num_trials=num_trials,
        resolved_rf=resolved_rf,
    )
    return _compute_metric_summary(
        metric_payload,
        resolved_rf=resolved_rf,
    )


@dataclass(frozen=True, slots=True)
class _ComputedMetricPayload:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    turnover: float
    trade_count: float
    win_rate: float
    avg_trade: float
    exposure: float
    volatility: float
    stability: float
    rolling_sharpe_min: float
    worst_month: float
    benchmark_corr: float
    deflated_sharpe: float
    pbo: float
    spa_pvalue: float


def _empty_compute_metric_payload() -> _ComputedMetricPayload:
    return _ComputedMetricPayload(
        total_return=0.0,
        cagr=0.0,
        sharpe=0.0,
        sortino=0.0,
        calmar=0.0,
        max_drawdown=0.0,
        turnover=0.0,
        trade_count=0.0,
        win_rate=0.0,
        avg_trade=0.0,
        exposure=0.0,
        volatility=0.0,
        stability=0.0,
        rolling_sharpe_min=0.0,
        worst_month=0.0,
        benchmark_corr=0.0,
        deflated_sharpe=0.0,
        pbo=1.0,
        spa_pvalue=1.0,
    )


def _resolve_compute_metric_payload(
    returns: np.ndarray,
    *,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
    num_trials: int,
    resolved_rf: Any,
) -> _ComputedMetricPayload:
    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = max(1.0 / float(periods_per_year), returns.size / float(periods_per_year))
    cagr = float(math.exp(math.log1p(max(-0.999999, total_return)) / years) - 1.0)

    sigma = _safe_std(returns)
    sharpe = compute_sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        risk_free_per_period=np.asarray(resolved_rf.periodic_rates, dtype=float),
    )
    sortino = compute_sortino_ratio(
        returns,
        periods_per_year=periods_per_year,
        target_per_period=np.asarray(resolved_rf.periodic_sortino_targets, dtype=float),
    )

    max_drawdown = _max_drawdown(returns)
    calmar = 0.0 if max_drawdown <= 1e-12 else cagr / max_drawdown

    trade_count = float(np.sum(turnover > 1e-9))
    win_rate = float(np.sum(returns > 0.0) / returns.size)
    avg_trade = float(np.sum(returns) / max(1.0, trade_count))

    rolling_sharpe_min = _rolling_sharpe_min(
        returns,
        window=min(128, max(32, returns.size // 8)),
        periods_per_year=periods_per_year,
    )
    worst_month = _worst_month(
        returns,
        bars_per_month=max(4, int(periods_per_year // 12)),
    )
    stability = 0.5 * max(-3.0, min(3.0, rolling_sharpe_min)) + 0.5 * worst_month

    return _ComputedMetricPayload(
        total_return=total_return,
        cagr=cagr,
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=float(max_drawdown),
        turnover=float(_safe_mean(turnover)),
        trade_count=trade_count,
        win_rate=win_rate,
        avg_trade=avg_trade,
        exposure=float(_safe_mean(np.abs(exposure))),
        volatility=float(sigma * math.sqrt(periods_per_year)),
        stability=float(stability),
        rolling_sharpe_min=float(rolling_sharpe_min),
        worst_month=float(worst_month),
        benchmark_corr=float(_correlation(returns, benchmark_returns)),
        deflated_sharpe=float(_deflated_sharpe_ratio(returns, num_trials=num_trials)),
        pbo=float(_approx_pbo(returns)),
        spa_pvalue=float(_spa_like_pvalue(returns)),
    )


def _compute_metric_summary(
    metric_payload: _ComputedMetricPayload,
    *,
    resolved_rf: Any | None,
) -> dict[str, float]:
    risk_free_annual = 0.0 if resolved_rf is None else float(resolved_rf.annual_rate)
    risk_free_per_period = 0.0 if resolved_rf is None else float(resolved_rf.per_period_rate)
    sortino_target_annual = 0.0 if resolved_rf is None else float(resolved_rf.sortino_target_annual)
    sortino_target_per_period = 0.0 if resolved_rf is None else float(resolved_rf.sortino_target_per_period)
    return {
        "return": metric_payload.total_return,
        "total_return": metric_payload.total_return,
        "cagr": metric_payload.cagr,
        "sharpe": metric_payload.sharpe,
        "sortino": metric_payload.sortino,
        "calmar": metric_payload.calmar,
        "mdd": metric_payload.max_drawdown,
        "max_drawdown": metric_payload.max_drawdown,
        "turnover": metric_payload.turnover,
        "trades": metric_payload.trade_count,
        "trade_count": metric_payload.trade_count,
        "win_rate": metric_payload.win_rate,
        "avg_trade": metric_payload.avg_trade,
        "exposure": metric_payload.exposure,
        "volatility": metric_payload.volatility,
        "stability": metric_payload.stability,
        "rolling_sharpe_min": metric_payload.rolling_sharpe_min,
        "worst_month": metric_payload.worst_month,
        "benchmark_corr": metric_payload.benchmark_corr,
        "deflated_sharpe": metric_payload.deflated_sharpe,
        "pbo": metric_payload.pbo,
        "spa_pvalue": metric_payload.spa_pvalue,
        "risk_free_annual": risk_free_annual,
        "risk_free_per_period": risk_free_per_period,
        "sortino_target_annual": sortino_target_annual,
        "sortino_target_per_period": sortino_target_per_period,
    }


def _hurdle_fields(
    train: dict[str, float],
    val: dict[str, float],
    oos: dict[str, float],
    *,
    scoring_config: Mapping[str, Any] | None = None,
    oos_sharpe_min: float | None = None,
    max_pbo: float | None = None,
    max_turnover: float | None = None,
    max_drawdown: float | None = None,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    config = _resolve_hurdle_config(
        scoring_config=scoring_config,
        oos_sharpe_min=oos_sharpe_min,
        max_pbo=max_pbo,
        max_turnover=max_turnover,
        max_drawdown=max_drawdown,
    )
    fields = {
        "train": _pack_hurdle_stage(train, stage="train", config=config),
        "val": _pack_hurdle_stage(val, stage="val", config=config),
        "oos": _pack_hurdle_stage(oos, stage="oos", config=config),
    }
    hard_reject_reasons = _hurdle_hard_reject_reasons(oos, config=config)
    cost_ok = bool(
        float(oos.get("sharpe", 0.0)) >= config.oos_sharpe_min
        and float(oos.get("pbo", 1.0)) <= config.max_pbo
    )
    return fields, bool(cost_ok and not hard_reject_reasons), hard_reject_reasons


@dataclass(frozen=True, slots=True)
class _ResolvedHurdleConfig:
    weights: dict[str, float]
    in_sample_sharpe_min: float
    oos_sharpe_min: float
    max_pbo: float
    max_turnover: float
    max_drawdown: float
    min_trade_count: float


def _resolve_hurdle_config(
    *,
    scoring_config: Mapping[str, Any] | None,
    oos_sharpe_min: float | None,
    max_pbo: float | None,
    max_turnover: float | None,
    max_drawdown: float | None,
) -> _ResolvedHurdleConfig:
    cfg = _resolve_score_config(scoring_config)
    thresholds = dict(cfg["reject_thresholds"])
    return _ResolvedHurdleConfig(
        weights={key: float(value) for key, value in dict(cfg["hurdle_score_weights"]).items()},
        in_sample_sharpe_min=float(thresholds["in_sample_sharpe_min"]),
        oos_sharpe_min=float(thresholds["oos_sharpe_min"] if oos_sharpe_min is None else oos_sharpe_min),
        max_pbo=float(thresholds["max_pbo"] if max_pbo is None else max_pbo),
        max_turnover=float(thresholds["max_turnover"] if max_turnover is None else max_turnover),
        max_drawdown=float(thresholds["max_drawdown"] if max_drawdown is None else max_drawdown),
        min_trade_count=float(thresholds["min_trade_count"]),
    )


def _pack_hurdle_stage(
    metrics: dict[str, float],
    *,
    stage: str,
    config: _ResolvedHurdleConfig,
) -> dict[str, Any]:
    score = (
        (config.weights["sharpe_weight"] * float(metrics.get("sharpe", 0.0)))
        + (config.weights["return_weight"] * float(metrics.get("return", 0.0)))
        + (config.weights["deflated_sharpe_weight"] * float(metrics.get("deflated_sharpe", 0.0)))
        - (config.weights["pbo_penalty"] * float(metrics.get("pbo", 1.0)))
        - (config.weights["turnover_penalty"] * max(0.0, float(metrics.get("turnover", 0.0)) - config.max_turnover))
        - (config.weights["drawdown_penalty"] * max(0.0, float(metrics.get("mdd", 0.0)) - config.max_drawdown))
        - (config.weights["spa_pvalue_penalty"] * float(metrics.get("spa_pvalue", 1.0)))
    )
    sharpe_min = config.in_sample_sharpe_min if stage != "oos" else config.oos_sharpe_min
    passed = bool(
        float(metrics.get("sharpe", 0.0)) >= sharpe_min
        and float(metrics.get("pbo", 1.0)) <= config.max_pbo
        and float(metrics.get("turnover", 0.0)) <= config.max_turnover
        and float(metrics.get("mdd", 0.0)) <= config.max_drawdown
        and float(metrics.get("trade_count", 0.0)) >= config.min_trade_count
    )
    return {
        "pass": passed,
        "score": float(score),
        "excess_return": float(metrics.get("return", 0.0)),
    }


def _hurdle_hard_reject_reasons(
    oos: dict[str, float],
    *,
    config: _ResolvedHurdleConfig,
) -> dict[str, Any]:
    hard_reject_reasons: dict[str, Any] = {}
    if float(oos.get("sharpe", 0.0)) < config.oos_sharpe_min:
        hard_reject_reasons["oos_sharpe"] = float(oos.get("sharpe", 0.0))
    if float(oos.get("pbo", 1.0)) > config.max_pbo:
        hard_reject_reasons["pbo"] = float(oos.get("pbo", 1.0))
    if float(oos.get("turnover", 0.0)) > config.max_turnover:
        hard_reject_reasons["turnover"] = float(oos.get("turnover", 0.0))
    if float(oos.get("mdd", 0.0)) > config.max_drawdown:
        hard_reject_reasons["max_drawdown"] = float(oos.get("mdd", 0.0))
    if float(oos.get("trade_count", 0.0)) < config.min_trade_count:
        hard_reject_reasons["trade_count"] = float(oos.get("trade_count", 0.0))
    return hard_reject_reasons


def _split_masks_from_datetimes(
    datetimes: np.ndarray,
    *,
    split: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    size = int(datetimes.size)
    empty = np.zeros(size, dtype=bool)
    if size <= 0:
        return {"train": empty, "val": empty, "oos": empty}

    if not isinstance(split, Mapping):
        split_train, split_val, split_oos = _split_lengths(size)
        train_mask = np.zeros(size, dtype=bool)
        val_mask = np.zeros(size, dtype=bool)
        oos_mask = np.zeros(size, dtype=bool)
        train_mask[split_train] = True
        val_mask[split_val] = True
        oos_mask[split_oos] = True
        return {"train": train_mask, "val": val_mask, "oos": oos_mask}

    resolved = _resolve_split_config(split, strategy_timeframe=str(split.get("strategy_timeframe") or "1m"))
    stage_bounds = {
        "train": (
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("train_start"))),
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("train_end"), end_of_day=True)),
        ),
        "val": (
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("val_start"))),
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("val_end"), end_of_day=True)),
        ),
        "oos": (
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("oos_start"))),
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("oos_end"), end_of_day=True)),
        ),
    }

    ts = np.asarray(datetimes, dtype="datetime64[ms]")
    covered = np.zeros(size, dtype=bool)
    out: dict[str, np.ndarray] = {}
    for stage in ("train", "val", "oos"):
        start, end = stage_bounds[stage]
        mask = np.ones(size, dtype=bool)
        if start is not None:
            mask &= ts >= start
        if end is not None:
            mask &= ts <= end
        if start is not None and end is not None and end < start:
            mask &= False
        mask &= ~covered
        covered |= mask
        out[stage] = mask
    return out


def _align_series_to_timestamps(
    target_timestamps: np.ndarray,
    *,
    source_timestamps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_timestamps, dtype="datetime64[ms]")
    source = np.asarray(source_timestamps, dtype="datetime64[ms]")
    arr = np.asarray(values, dtype=float)
    if target.size == 0 or source.size == 0 or arr.size == 0:
        return np.zeros(target.size, dtype=float)
    idx = np.searchsorted(source, target)
    valid = (idx >= 0) & (idx < source.size)
    out = np.zeros(target.size, dtype=float)
    if not np.any(valid):
        return out
    matched = valid.copy()
    matched[valid] = source[idx[valid]] == target[valid]
    if np.any(matched):
        out[matched] = arr[idx[matched]]
    return out


def _series_to_stream(
    values: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
    offset: int = 0,
) -> list[dict[str, float | int]]:
    if timestamps is None:
        return [{"t": float(offset + idx), "v": float(value)} for idx, value in enumerate(values)]

    ts = np.asarray(timestamps, dtype="datetime64[ms]")
    out: list[dict[str, float | int]] = []
    for idx, value in enumerate(values):
        epoch_ms = int(ts[idx].astype("datetime64[ms]").astype(np.int64))
        out.append({"t": epoch_ms, "v": float(value)})
    return out


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _rolling_z(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if values.size < win:
        return out
    for idx in range(win, values.size + 1):
        tail = values[idx - win : idx]
        hist = tail[:-1]
        latest = tail[-1]
        std = _safe_std(hist)
        if std <= 1e-12:
            out[idx - 1] = 0.0
        else:
            out[idx - 1] = (float(latest) - _safe_mean(hist)) / std
    return out


def _trend_efficiency_series(closes: np.ndarray, window: int = 55) -> np.ndarray:
    out = np.full(closes.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if closes.size < win + 1:
        return out
    for idx in range(win, closes.size):
        span = closes[idx - win : idx + 1]
        net = abs(float(span[-1] - span[0]))
        path = float(np.sum(np.abs(np.diff(span))))
        out[idx] = 0.0 if path <= 1e-12 else min(1.0, max(0.0, net / path))
    return out


def _vol_ratio_series(closes: np.ndarray, fast: int = 8, slow: int = 55) -> np.ndarray:
    out = np.full(closes.shape, np.nan, dtype=float)
    rets = np.diff(np.log(np.clip(closes, 1e-12, np.inf)), prepend=np.log(max(1e-12, closes[0])))
    f = max(4, int(fast))
    s = max(f + 2, int(slow))
    if rets.size < s:
        return out
    for idx in range(s, rets.size + 1):
        fast_std = _safe_std(rets[idx - f : idx])
        slow_std = _safe_std(rets[idx - s : idx])
        out[idx - 1] = 0.0 if slow_std <= 1e-12 else fast_std / slow_std
    return out


def _rolling_volatility_series(closes: np.ndarray, window: int) -> np.ndarray:
    out = np.full(closes.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if closes.size < win:
        return out
    log_close = np.log(np.clip(closes, 1e-12, np.inf))
    rets = np.diff(log_close, prepend=log_close[0])
    for idx in range(win, rets.size + 1):
        out[idx - 1] = _safe_std(rets[idx - win : idx])
    return out


def _composite_trend_signal_strength(
    *,
    sigma: float,
    crowding_score: float | None,
    crowding_reduce_threshold: float,
    risk_target_vol: float,
    max_signal_strength: float,
) -> float:
    sigma_floor = max(1e-6, float(sigma))
    strength = float(risk_target_vol) / sigma_floor
    strength = min(float(max_signal_strength), max(0.10, strength))
    if crowding_score is not None and abs(float(crowding_score)) >= float(crowding_reduce_threshold):
        strength *= 0.5
    return float(max(0.05, strength))


def _composite_trend_position_series(
    *,
    close: np.ndarray,
    score: np.ndarray,
    gate: np.ndarray,
    long_gate: np.ndarray | None = None,
    short_gate: np.ndarray | None = None,
    crowding: np.ndarray | None,
    config: _CompositeTrendStrategyConfig,
) -> np.ndarray:
    close_arr = np.asarray(close, dtype=float)
    score_arr = np.asarray(score, dtype=float)
    gate_arr = np.asarray(gate, dtype=bool)
    long_gate_arr = gate_arr if long_gate is None else np.asarray(long_gate, dtype=bool)
    short_gate_arr = gate_arr if short_gate is None else np.asarray(short_gate, dtype=bool)
    crowd_arr = (
        np.full(close_arr.shape, np.nan, dtype=float)
        if crowding is None
        else np.asarray(crowding, dtype=float)
    )
    sigma_arr = _rolling_volatility_series(close_arr, config.vol_window)
    position = np.zeros(close_arr.shape, dtype=float)
    mode = 0
    bars_held = 0

    for idx in range(close_arr.size):
        step = _composite_trend_step_input(
            idx=idx,
            close_arr=close_arr,
            score_arr=score_arr,
            gate_arr=gate_arr,
            long_gate_arr=long_gate_arr,
            short_gate_arr=short_gate_arr,
            crowd_arr=crowd_arr,
            sigma_arr=sigma_arr,
            config=config,
        )
        mode, bars_held, position[idx] = _composite_trend_step(
            mode=mode,
            bars_held=bars_held,
            step=step,
            config=config,
        )

    return position


def _composite_trend_step_input(
    *,
    idx: int,
    close_arr: np.ndarray,
    score_arr: np.ndarray,
    gate_arr: np.ndarray,
    long_gate_arr: np.ndarray,
    short_gate_arr: np.ndarray,
    crowd_arr: np.ndarray,
    sigma_arr: np.ndarray,
    config: _CompositeTrendStrategyConfig,
) -> _CompositeTrendStepInput:
    close_i = float(close_arr[idx])
    score_i = float(score_arr[idx]) if np.isfinite(score_arr[idx]) else float("nan")
    gate_i = bool(gate_arr[idx]) if idx < gate_arr.size else False
    long_gate_i = bool(long_gate_arr[idx]) if idx < long_gate_arr.size else gate_i
    short_gate_i = bool(short_gate_arr[idx]) if idx < short_gate_arr.size else gate_i
    crowd_i = crowd_arr[idx] if idx < crowd_arr.size and np.isfinite(crowd_arr[idx]) else None
    strength = _composite_trend_signal_strength(
        sigma=float(sigma_arr[idx]) if np.isfinite(sigma_arr[idx]) else 0.0,
        crowding_score=crowd_i,
        crowding_reduce_threshold=config.crowding_reduce_threshold,
        risk_target_vol=config.risk_target_vol,
        max_signal_strength=config.max_signal_strength,
    )
    blocked = crowd_i is not None and abs(float(crowd_i)) >= config.crowding_block_threshold
    return _CompositeTrendStepInput(
        close_i=close_i,
        score_i=score_i,
        long_gate_i=long_gate_i,
        short_gate_i=short_gate_i,
        strength=strength,
        blocked=blocked,
    )


def _composite_trend_step(
    *,
    mode: int,
    bars_held: int,
    step: _CompositeTrendStepInput,
    config: _CompositeTrendStrategyConfig,
) -> tuple[int, int, float]:
    if not np.isfinite(step.close_i):
        return mode, bars_held, float(mode) * step.strength

    if mode != 0:
        next_bars_held = bars_held + 1
        if _composite_trend_should_exit(
            mode=mode,
            score_i=step.score_i,
            long_gate_i=step.long_gate_i,
            short_gate_i=step.short_gate_i,
            bars_held=next_bars_held,
            config=config,
        ):
            return 0, 0, 0.0
        return mode, next_bars_held, float(mode) * step.strength

    next_mode = _composite_trend_entry_mode(
        score_i=step.score_i,
        long_gate_i=step.long_gate_i,
        short_gate_i=step.short_gate_i,
        blocked=step.blocked,
        config=config,
    )
    if next_mode == 0:
        return 0, 0, 0.0
    return next_mode, 0, float(next_mode) * step.strength


def _composite_trend_should_exit(
    *,
    mode: int,
    score_i: float,
    long_gate_i: bool,
    short_gate_i: bool,
    bars_held: int,
    config: _CompositeTrendStrategyConfig,
) -> bool:
    if mode == 1:
        return (
            (not long_gate_i)
            or (not np.isfinite(score_i))
            or (score_i <= config.exit_score_cross)
            or (bars_held >= config.max_hold_bars)
        )
    return (
        (not short_gate_i)
        or (not np.isfinite(score_i))
        or (score_i >= -config.exit_score_cross)
        or (bars_held >= config.max_hold_bars)
    )


def _composite_trend_entry_mode(
    *,
    score_i: float,
    long_gate_i: bool,
    short_gate_i: bool,
    blocked: bool,
    config: _CompositeTrendStrategyConfig,
) -> int:
    if (not long_gate_i and not short_gate_i) or not np.isfinite(score_i) or blocked:
        return 0
    if long_gate_i and score_i >= config.long_threshold:
        return 1
    if config.allow_short and short_gate_i and score_i <= -config.short_threshold:
        return -1
    return 0


def _vwap_dev_z(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int = 60, z_window: int = 120) -> np.ndarray:
    n = close.size
    out = np.full(close.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if n < win:
        return out
    dev = np.full(close.shape, np.nan, dtype=float)
    typical = (high + low + close) / 3.0
    vol = np.clip(volume, 0.0, np.inf)
    for idx in range(win, n + 1):
        p = typical[idx - win : idx]
        v = vol[idx - win : idx]
        den = float(np.sum(v))
        vw = float(np.sum(p * v) / den) if den > 1e-12 else float(np.mean(p))
        if vw <= 0.0:
            dev[idx - 1] = 0.0
        else:
            dev[idx - 1] = (close[idx - 1] / vw) - 1.0
    out = _rolling_z(np.nan_to_num(dev, nan=0.0), window=max(16, int(z_window)))
    return out


def _rolling_vwap_deviation(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    out = np.full(close.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if close.size < win:
        return out
    vol = np.clip(np.asarray(volume, dtype=float), 0.0, np.inf)
    close_arr = np.asarray(close, dtype=float)
    for idx in range(win, close_arr.size + 1):
        px = close_arr[idx - win : idx]
        vv = vol[idx - win : idx]
        den = float(np.sum(vv))
        vw = float(np.sum(px * vv) / den) if den > 1e-12 else _safe_mean(px)
        if vw <= 1e-12:
            out[idx - 1] = 0.0
        else:
            out[idx - 1] = (float(close_arr[idx - 1]) / vw) - 1.0
    return out


def _rolling_channel(high: np.ndarray, low: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    high_out = np.full(high.shape, np.nan, dtype=float)
    low_out = np.full(low.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if high.size < win + 1:
        return high_out, low_out
    hi = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    for idx in range(win, hi.size):
        high_out[idx] = float(np.max(hi[idx - win : idx]))
        low_out[idx] = float(np.min(lo[idx - win : idx]))
    return high_out, low_out


def _rolling_slope_series(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    win = max(2, int(window))
    if values.size < win:
        return out

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr).astype(np.int64)
    prefix_valid = np.concatenate(([0], np.cumsum(finite)))
    clean = np.nan_to_num(arr, nan=0.0)
    prefix_y = np.concatenate(([0.0], np.cumsum(clean)))
    indices = np.arange(arr.size, dtype=float)
    prefix_xy_abs = np.concatenate(([0.0], np.cumsum(clean * indices)))

    sum_x = float(win * (win - 1) / 2.0)
    sum_x2 = float(((win - 1) * win * ((2 * win) - 1)) / 6.0)
    denom = float((win * sum_x2) - (sum_x * sum_x))
    if abs(denom) <= 1e-12:
        return out

    for end in range(win - 1, arr.size):
        start = end - win + 1
        if int(prefix_valid[end + 1] - prefix_valid[start]) != win:
            continue
        sum_y = float(prefix_y[end + 1] - prefix_y[start])
        sum_xy = float(prefix_xy_abs[end + 1] - prefix_xy_abs[start]) - (float(start) * sum_y)
        numer = float((win * sum_xy) - (sum_x * sum_y))
        out[end] = numer / denom
    return out


def _composite_momentum_series(
    values: np.ndarray,
    *,
    windows: Sequence[int] = (8, 21, 55),
    weights: Sequence[float] = (0.5, 0.3, 0.2),
) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return out

    score = np.zeros(arr.shape, dtype=float)
    total_weight = np.zeros(arr.shape, dtype=float)
    for win, weight in zip(windows, weights, strict=True):
        window_i = max(1, int(win))
        if arr.size <= window_i:
            continue
        latest = arr[window_i:]
        base = arr[:-window_i]
        valid = np.isfinite(latest) & np.isfinite(base) & (latest > 0.0) & (base > 0.0)
        if not np.any(valid):
            continue
        contribution = np.zeros(latest.shape, dtype=float)
        contribution[valid] = np.log(latest[valid] / base[valid])
        score[window_i:] += float(weight) * contribution
        total_weight[window_i:] += abs(float(weight)) * valid.astype(float)

    valid_out = total_weight > 1e-12
    out[valid_out] = score[valid_out] / total_weight[valid_out]
    return out


@dataclass(frozen=True, slots=True)
class _VwapReversionConfig:
    window: int
    entry_dev: float
    exit_dev: float
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _VwapReversionStepInput:
    close_i: float
    deviation_i: float


def _resolve_vwap_reversion_config(params: Mapping[str, Any]) -> _VwapReversionConfig:
    return _VwapReversionConfig(
        window=max(8, int(params.get("window", 64))),
        entry_dev=float(params.get("entry_dev", 0.02)),
        exit_dev=max(0.0, float(params.get("exit_dev", 0.005))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _vwap_reversion_position_series(
    *,
    close: np.ndarray,
    volume: np.ndarray,
    config: _VwapReversionConfig,
) -> np.ndarray:
    deviation = np.nan_to_num(
        _rolling_vwap_deviation(close, volume, config.window),
        nan=0.0,
    )
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _vwap_reversion_step_input(
            idx=idx,
            close=close,
            deviation=deviation,
        )
        mode, entry_price, position[idx] = _vwap_reversion_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            config=config,
        )

    return position


def _vwap_reversion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    deviation: np.ndarray,
) -> _VwapReversionStepInput:
    return _VwapReversionStepInput(
        close_i=float(close[idx]),
        deviation_i=float(deviation[idx]),
    )


def _vwap_reversion_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _VwapReversionStepInput,
    config: _VwapReversionConfig,
) -> tuple[int, float | None, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, 0.0

    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        revert_hit = step.deviation_i >= -config.exit_dev
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        revert_hit = step.deviation_i <= config.exit_dev
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0:
        if step.deviation_i <= -config.entry_dev:
            return 1, step.close_i, 1.0
        if config.allow_short and step.deviation_i >= config.entry_dev:
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_vwap_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_vwap_reversion_config(params)
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)
        exposures[s_idx] = _vwap_reversion_position_series(
            close=close,
            volume=volume,
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _RollingBreakoutConfig:
    lookback_bars: int
    breakout_buffer: float
    atr_window: int
    atr_stop_multiplier: float
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _RollingBreakoutStepInput:
    close_i: float
    stop_pct: float
    upper: float | None
    lower: float | None


def _resolve_rolling_breakout_config(params: Mapping[str, Any]) -> _RollingBreakoutConfig:
    return _RollingBreakoutConfig(
        lookback_bars=max(8, int(params.get("lookback_bars", 48))),
        breakout_buffer=float(params.get("breakout_buffer", 0.0)),
        atr_window=max(4, int(params.get("atr_window", 14))),
        atr_stop_multiplier=float(params.get("atr_stop_multiplier", 2.5)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", False)),
    )


def _rolling_breakout_atr_pct(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    atr_pct = np.full(close.shape, np.nan, dtype=float)
    if close.size == 0:
        return atr_pct
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    for idx in range(window, close.size + 1):
        atr_pct[idx - 1] = _safe_mean(
            tr[idx - window : idx] / np.clip(close[idx - window : idx], 1e-12, np.inf)
        )
    return atr_pct


def _rolling_breakout_position_series(
    *,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    atr_pct: np.ndarray,
    config: _RollingBreakoutConfig,
) -> np.ndarray:
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _rolling_breakout_step_input(
            idx=idx,
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            atr_pct=atr_pct,
            config=config,
        )
        mode, entry_price, position[idx] = _rolling_breakout_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            config=config,
        )

    return position


def _rolling_breakout_step_input(
    *,
    idx: int,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    atr_pct: np.ndarray,
    config: _RollingBreakoutConfig,
) -> _RollingBreakoutStepInput:
    stop_pct = max(
        config.stop_loss_pct,
        config.atr_stop_multiplier * float(atr_pct[idx])
        if np.isfinite(atr_pct[idx])
        else config.stop_loss_pct,
    )
    return _RollingBreakoutStepInput(
        close_i=float(close[idx]),
        stop_pct=stop_pct,
        upper=float(channel_high[idx]) if np.isfinite(channel_high[idx]) else None,
        lower=float(channel_low[idx]) if np.isfinite(channel_low[idx]) else None,
    )


def _rolling_breakout_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _RollingBreakoutStepInput,
    config: _RollingBreakoutConfig,
) -> tuple[int, float | None, float]:
    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - step.stop_pct)
        exit_hit = step.lower is not None and step.close_i < step.lower
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + step.stop_pct)
        exit_hit = step.upper is not None and step.close_i > step.upper
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0 and step.upper is not None and step.lower is not None:
        if step.close_i >= step.upper * (1.0 + config.breakout_buffer):
            return 1, step.close_i, 1.0
        if config.allow_short and step.close_i <= step.lower * (1.0 - config.breakout_buffer):
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_rolling_breakout_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_rolling_breakout_config(params)
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        channel_high, channel_low = _rolling_channel(high, low, config.lookback_bars)
        atr_pct = _rolling_breakout_atr_pct(
            close=close,
            high=high,
            low=low,
            window=config.atr_window,
        )
        exposures[s_idx] = _rolling_breakout_position_series(
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            atr_pct=atr_pct,
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _RegimeBreakoutCandidateConfig:
    lookback_window: int
    slope_window: int
    volatility_fast_window: int
    volatility_slow_window: int
    range_entry_threshold: float
    slope_entry_threshold: float
    momentum_floor: float
    max_volatility_ratio: float
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _RegimeBreakoutStepInput:
    close_i: float
    upper: float | None
    lower: float | None
    slope: float | None
    momentum: float | None
    vol_ok: bool
    range_pos: float | None


def _resolve_regime_breakout_candidate_config(
    params: Mapping[str, Any],
) -> _RegimeBreakoutCandidateConfig:
    volatility_fast_window = max(4, int(params.get("volatility_fast_window", 12)))
    return _RegimeBreakoutCandidateConfig(
        lookback_window=max(8, int(params.get("lookback_window", 48))),
        slope_window=max(4, int(params.get("slope_window", 21))),
        volatility_fast_window=volatility_fast_window,
        volatility_slow_window=max(
            volatility_fast_window + 2,
            int(params.get("volatility_slow_window", 48)),
        ),
        range_entry_threshold=float(params.get("range_entry_threshold", 0.70)),
        slope_entry_threshold=float(params.get("slope_entry_threshold", 0.0)),
        momentum_floor=float(params.get("momentum_floor", 0.0)),
        max_volatility_ratio=float(params.get("max_volatility_ratio", 1.8)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _regime_breakout_candidate_position_series(
    *,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    vol_ratio: np.ndarray,
    slope_series: np.ndarray,
    momentum_series: np.ndarray,
    config: _RegimeBreakoutCandidateConfig,
) -> np.ndarray:
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _regime_breakout_step_input(
            idx=idx,
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            vol_ratio=vol_ratio,
            slope_series=slope_series,
            momentum_series=momentum_series,
            config=config,
        )
        mode, entry_price, position[idx] = _regime_breakout_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            config=config,
        )

    return position


def _regime_breakout_step_input(
    *,
    idx: int,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    vol_ratio: np.ndarray,
    slope_series: np.ndarray,
    momentum_series: np.ndarray,
    config: _RegimeBreakoutCandidateConfig,
) -> _RegimeBreakoutStepInput:
    close_i = float(close[idx])
    upper = float(channel_high[idx]) if np.isfinite(channel_high[idx]) else None
    lower = float(channel_low[idx]) if np.isfinite(channel_low[idx]) else None
    slope = float(slope_series[idx]) if np.isfinite(slope_series[idx]) else None
    momentum = float(momentum_series[idx]) if np.isfinite(momentum_series[idx]) else None
    range_pos = None
    if upper is not None and lower is not None and upper > lower:
        range_pos = (close_i - lower) / max(upper - lower, 1e-12)
    return _RegimeBreakoutStepInput(
        close_i=close_i,
        upper=upper,
        lower=lower,
        slope=slope,
        momentum=momentum,
        vol_ok=float(vol_ratio[idx]) <= config.max_volatility_ratio,
        range_pos=range_pos,
    )


def _regime_breakout_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _RegimeBreakoutStepInput,
    config: _RegimeBreakoutCandidateConfig,
) -> tuple[int, float | None, float]:
    if (
        step.upper is None
        or step.lower is None
        or step.upper <= step.lower
        or step.slope is None
        or step.momentum is None
        or step.range_pos is None
    ):
        return mode, entry_price, 0.0

    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        exit_hit = step.slope < 0.0 or step.range_pos < 0.50
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        exit_hit = step.slope > 0.0 or step.range_pos > 0.50
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0 and step.vol_ok:
        if (
            step.range_pos >= config.range_entry_threshold
            and step.slope >= config.slope_entry_threshold
            and step.momentum >= config.momentum_floor
        ):
            return 1, step.close_i, 1.0
        if (
            config.allow_short
            and step.range_pos <= (1.0 - config.range_entry_threshold)
            and step.slope <= -config.slope_entry_threshold
            and step.momentum <= -config.momentum_floor
        ):
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_regime_breakout_candidate_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_regime_breakout_candidate_config(params)
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        channel_high, channel_low = _rolling_channel(high, low, config.lookback_window)
        vol_ratio = np.nan_to_num(
            _vol_ratio_series(
                close,
                config.volatility_fast_window,
                config.volatility_slow_window,
            ),
            nan=np.inf,
        )
        slope_series = _rolling_slope_series(close, config.slope_window)
        momentum_series = _composite_momentum_series(close)
        exposures[s_idx] = _regime_breakout_candidate_position_series(
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            vol_ratio=vol_ratio,
            slope_series=slope_series,
            momentum_series=momentum_series,
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _BasisSnapbackReversionConfig:
    window: int
    entry_z: float
    exit_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _BasisSnapbackStepInput:
    close_i: float
    z_i: float


def _resolve_basis_snapback_reversion_config(
    params: Mapping[str, Any],
) -> _BasisSnapbackReversionConfig:
    return _BasisSnapbackReversionConfig(
        window=int(params.get("window", 96)),
        entry_z=float(params.get("entry_z", 1.8)),
        exit_z=max(0.0, float(params.get("exit_z", 0.4))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _basis_snapback_reversion_position_series(
    *,
    close: np.ndarray,
    basis_z: np.ndarray,
    config: _BasisSnapbackReversionConfig,
) -> np.ndarray:
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0

    for idx in range(close.size):
        step = _basis_snapback_step_input(
            idx=idx,
            close=close,
            basis_z=basis_z,
        )
        mode, entry_price, bars_held, position[idx] = _basis_snapback_step(
            mode=mode,
            entry_price=entry_price,
            bars_held=bars_held,
            step=step,
            config=config,
        )

    return position


def _basis_snapback_step_input(
    *,
    idx: int,
    close: np.ndarray,
    basis_z: np.ndarray,
) -> _BasisSnapbackStepInput:
    return _BasisSnapbackStepInput(
        close_i=float(close[idx]),
        z_i=float(basis_z[idx]) if np.isfinite(basis_z[idx]) else float("nan"),
    )


def _basis_snapback_step(
    *,
    mode: int,
    entry_price: float | None,
    bars_held: int,
    step: _BasisSnapbackStepInput,
    config: _BasisSnapbackReversionConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, bars_held, float(mode)

    if mode == 1:
        next_bars_held = bars_held + 1
        should_exit = (
            (np.isfinite(step.z_i) and step.z_i >= -config.exit_z)
            or (next_bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_bars_held, 1.0

    if mode == -1:
        next_bars_held = bars_held + 1
        should_exit = (
            (np.isfinite(step.z_i) and step.z_i <= config.exit_z)
            or (next_bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_bars_held, -1.0

    if not np.isfinite(step.z_i):
        return 0, entry_price, bars_held, 0.0
    if step.z_i <= -config.entry_z:
        return 1, step.close_i, 0, 1.0
    if config.allow_short and step.z_i >= config.entry_z:
        return -1, step.close_i, 0, -1.0
    return 0, entry_price, bars_held, 0.0


def _apply_basis_snapback_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_basis_snapback_reversion_config(params)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        close = aligned[f"{symbol}:close"]
        mark = aligned.get(f"{symbol}:mark_price")
        index = aligned.get(f"{symbol}:index_price")
        if mark is None or index is None:
            missing_symbols.append(symbol)
            continue
        support = _crowding_support_series(
            funding_rate=np.zeros(len(close), dtype=float),
            open_interest=np.ones(len(close), dtype=float),
            mark_price=np.asarray(mark, dtype=float),
            index_price=np.asarray(index, dtype=float),
            liquidation_long_notional=np.zeros(len(close), dtype=float),
            liquidation_short_notional=np.zeros(len(close), dtype=float),
            window=config.window,
        )
        exposures[s_idx] = _basis_snapback_reversion_position_series(
            close=np.asarray(close, dtype=float),
            basis_z=np.asarray(support["basis_z"], dtype=float),
            config=config,
        )

    if missing_symbols:
        meta["missing_support_data"] = True
        meta["missing_support_symbols"] = missing_symbols


@dataclass(frozen=True, slots=True)
class _VolOfVolExhaustionFadeConfig:
    vol_window: int
    vol_z_window: int
    return_z_window: int
    vol_entry_z: float
    return_entry_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _VolOfVolExhaustionStepInput:
    close_i: float
    vol_z_i: float
    return_z_i: float


def _resolve_vol_of_vol_exhaustion_fade_config(
    params: Mapping[str, Any],
) -> _VolOfVolExhaustionFadeConfig:
    return _VolOfVolExhaustionFadeConfig(
        vol_window=max(8, int(params.get("vol_window", 24))),
        vol_z_window=max(8, int(params.get("vol_z_window", 48))),
        return_z_window=max(8, int(params.get("return_z_window", 24))),
        vol_entry_z=float(params.get("vol_entry_z", 1.8)),
        return_entry_z=float(params.get("return_entry_z", 1.2)),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 8))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _vol_of_vol_exhaustion_fade_position_series(
    *,
    close: np.ndarray,
    config: _VolOfVolExhaustionFadeConfig,
) -> np.ndarray:
    realized_vol = _rolling_realized_vol(close, config.vol_window)
    vol_z = np.nan_to_num(
        _rolling_z(np.nan_to_num(realized_vol, nan=0.0), config.vol_z_window),
        nan=0.0,
    )
    return_z = np.nan_to_num(_rolling_z(close, config.return_z_window), nan=0.0)
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    hold_bars = 0

    for idx in range(close.size):
        step = _vol_of_vol_exhaustion_step_input(
            idx=idx,
            close=close,
            vol_z=vol_z,
            return_z=return_z,
        )
        mode, entry_price, hold_bars, position[idx] = _vol_of_vol_exhaustion_step(
            mode=mode,
            entry_price=entry_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )

    return position


def _vol_of_vol_exhaustion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    vol_z: np.ndarray,
    return_z: np.ndarray,
) -> _VolOfVolExhaustionStepInput:
    return _VolOfVolExhaustionStepInput(
        close_i=float(close[idx]),
        vol_z_i=float(vol_z[idx]),
        return_z_i=float(return_z[idx]),
    )


def _vol_of_vol_exhaustion_step(
    *,
    mode: int,
    entry_price: float | None,
    hold_bars: int,
    step: _VolOfVolExhaustionStepInput,
    config: _VolOfVolExhaustionFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, hold_bars, 0.0

    if mode == 1 and entry_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        revert_hit = step.return_z_i >= 0.0
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_hold_bars, 1.0

    if mode == -1 and entry_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        revert_hit = step.return_z_i <= 0.0
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_hold_bars, -1.0

    if mode == 0 and step.vol_z_i >= config.vol_entry_z:
        if step.return_z_i <= -config.return_entry_z:
            return 1, step.close_i, 0, 1.0
        if config.allow_short and step.return_z_i >= config.return_entry_z:
            return -1, step.close_i, 0, -1.0
    return mode, entry_price, hold_bars, 0.0


def _apply_vol_of_vol_exhaustion_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_vol_of_vol_exhaustion_fade_config(params)
    for s_idx, symbol in enumerate(symbols):
        exposures[s_idx] = _vol_of_vol_exhaustion_fade_position_series(
            close=np.asarray(aligned[f"{symbol}:close"], dtype=float),
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _MultiHorizonTrendExhaustionFadeConfig:
    short_window: int
    entry_z: float
    exit_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _MultiHorizonTrendExhaustionStepInput:
    close_i: float
    z_i: float
    long_i: float


def _resolve_multi_horizon_trend_exhaustion_fade_config(
    params: Mapping[str, Any],
) -> _MultiHorizonTrendExhaustionFadeConfig:
    return _MultiHorizonTrendExhaustionFadeConfig(
        short_window=max(4, int(params.get("short_window", 16))),
        entry_z=float(params.get("entry_z", 1.6)),
        exit_z=max(0.0, float(params.get("exit_z", 0.3))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 10))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _multi_horizon_trend_exhaustion_fade_position_series(
    *,
    close: np.ndarray,
    config: _MultiHorizonTrendExhaustionFadeConfig,
) -> np.ndarray:
    short_z = np.nan_to_num(_rolling_z(close, config.short_window), nan=0.0)
    long_mom = _composite_momentum_series(
        close,
        windows=(8, 21, 55),
        weights=(0.5, 0.3, 0.2),
    )
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    hold_bars = 0

    for idx in range(close.size):
        step = _multi_horizon_trend_exhaustion_step_input(
            idx=idx,
            close=close,
            short_z=short_z,
            long_mom=long_mom,
        )
        mode, entry_price, hold_bars, position[idx] = _multi_horizon_trend_exhaustion_step(
            mode=mode,
            entry_price=entry_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )

    return position


def _multi_horizon_trend_exhaustion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    short_z: np.ndarray,
    long_mom: np.ndarray,
) -> _MultiHorizonTrendExhaustionStepInput:
    return _MultiHorizonTrendExhaustionStepInput(
        close_i=float(close[idx]),
        z_i=float(short_z[idx]),
        long_i=float(long_mom[idx]) if np.isfinite(long_mom[idx]) else 0.0,
    )


def _multi_horizon_trend_exhaustion_step(
    *,
    mode: int,
    entry_price: float | None,
    hold_bars: int,
    step: _MultiHorizonTrendExhaustionStepInput,
    config: _MultiHorizonTrendExhaustionFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, hold_bars, 0.0

    if mode == 1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            step.z_i >= -config.exit_z
            or step.long_i < 0.0
            or next_hold_bars >= config.max_hold_bars
            or (
                entry_price is not None
                and step.close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_hold_bars, 1.0

    if mode == -1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            step.z_i <= config.exit_z
            or step.long_i > 0.0
            or next_hold_bars >= config.max_hold_bars
            or (
                entry_price is not None
                and step.close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_hold_bars, -1.0

    if step.z_i >= config.entry_z and step.long_i <= 0.0 and config.allow_short:
        return -1, step.close_i, 0, -1.0
    if step.z_i <= -config.entry_z and step.long_i >= 0.0:
        return 1, step.close_i, 0, 1.0
    return 0, entry_price, hold_bars, 0.0


def _apply_multi_horizon_trend_exhaustion_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_multi_horizon_trend_exhaustion_fade_config(params)
    for s_idx, symbol in enumerate(symbols):
        exposures[s_idx] = _multi_horizon_trend_exhaustion_fade_position_series(
            close=np.asarray(aligned[f"{symbol}:close"], dtype=float),
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _BreadthThrustFailureReversalConfig:
    momentum_lookback: int
    breadth_entry: float
    breadth_exit: float
    basket_return_floor: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _BreadthThrustStepInput:
    basket_close: float
    breadth: float
    mean_ret: float


def _resolve_breadth_thrust_failure_reversal_config(
    params: Mapping[str, Any],
) -> _BreadthThrustFailureReversalConfig:
    return _BreadthThrustFailureReversalConfig(
        momentum_lookback=max(2, int(params.get("momentum_lookback", 16))),
        breadth_entry=float(params.get("breadth_entry", 0.80)),
        breadth_exit=float(params.get("breadth_exit", 0.55)),
        basket_return_floor=float(params.get("basket_return_floor", 0.003)),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 8))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _apply_breadth_thrust_failure_reversal_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_breadth_thrust_failure_reversal_config(params)
    close_map_np = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float)
        for symbol in symbols
    }

    basket_position = _breadth_thrust_failure_reversal_position_series(
        close_map_np=close_map_np,
        config=config,
    )
    exposures[:] = basket_position
    meta["cross_sectional"] = True


def _breadth_thrust_failure_reversal_position_series(
    *,
    close_map_np: Mapping[str, np.ndarray],
    config: _BreadthThrustFailureReversalConfig,
) -> np.ndarray:
    if not close_map_np:
        return np.zeros(0, dtype=float)

    n = len(next(iter(close_map_np.values())))
    basket_position = np.zeros(n, dtype=float)

    basket_state = 0.0
    basket_entry = np.nan
    hold_bars = 0
    for idx in range(n):
        step = _breadth_thrust_step_input(
            idx=idx,
            close_map_np=close_map_np,
            config=config,
        )
        basket_state, basket_entry, hold_bars, basket_position[idx] = _breadth_thrust_step(
            idx=idx,
            basket_state=basket_state,
            basket_entry=basket_entry,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )

    return basket_position


def _breadth_thrust_step_input(
    *,
    idx: int,
    close_map_np: Mapping[str, np.ndarray],
    config: _BreadthThrustFailureReversalConfig,
) -> _BreadthThrustStepInput:
    close_map = {symbol: float(close[idx]) for symbol, close in close_map_np.items()}
    basket_close = float(np.mean(list(close_map.values())))
    if idx < config.momentum_lookback:
        return _BreadthThrustStepInput(
            basket_close=basket_close,
            breadth=float("nan"),
            mean_ret=float("nan"),
        )

    breadth_values: list[float] = []
    basket_returns: list[float] = []
    for close in close_map_np.values():
        latest = float(close[idx])
        base = float(close[idx - config.momentum_lookback])
        if not np.isfinite(latest) or not np.isfinite(base) or base <= 0.0:
            continue
        ret = (latest / base) - 1.0
        basket_returns.append(float(ret))
        breadth_values.append(1.0 if ret > 0.0 else 0.0)
    if not breadth_values:
        return _BreadthThrustStepInput(
            basket_close=basket_close,
            breadth=float("nan"),
            mean_ret=float("nan"),
        )
    return _BreadthThrustStepInput(
        basket_close=basket_close,
        breadth=float(np.mean(np.asarray(breadth_values, dtype=float))),
        mean_ret=float(np.mean(np.asarray(basket_returns, dtype=float))),
    )


def _breadth_thrust_step(
    *,
    idx: int,
    basket_state: float,
    basket_entry: float,
    hold_bars: int,
    step: _BreadthThrustStepInput,
    config: _BreadthThrustFailureReversalConfig,
) -> tuple[float, float, int, float]:
    if basket_state != 0.0 and np.isfinite(basket_entry):
        next_hold_bars = hold_bars + 1
        stop_long = (
            basket_state > 0.0
            and step.basket_close <= float(basket_entry) * (1.0 - config.stop_loss_pct)
        )
        stop_short = (
            basket_state < 0.0
            and step.basket_close >= float(basket_entry) * (1.0 + config.stop_loss_pct)
        )
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_long or stop_short or timeout_hit:
            basket_state = 0.0
            basket_entry = np.nan
            hold_bars = 0
        else:
            hold_bars = next_hold_bars

    if idx < config.momentum_lookback or not (
        np.isfinite(step.breadth) and np.isfinite(step.mean_ret)
    ):
        return basket_state, basket_entry, hold_bars, basket_state

    if basket_state == 0.0:
        if (
            config.allow_short
            and step.breadth >= config.breadth_entry
            and step.mean_ret <= -config.basket_return_floor
        ):
            return -1.0, step.basket_close, 0, -1.0
        if (
            step.breadth <= (1.0 - config.breadth_entry)
            and step.mean_ret >= config.basket_return_floor
        ):
            return 1.0, step.basket_close, 0, 1.0
    elif abs(step.breadth - 0.5) <= abs(config.breadth_exit - 0.5):
        return 0.0, np.nan, 0, 0.0

    return basket_state, basket_entry, hold_bars, basket_state


@dataclass(frozen=True, slots=True)
class _LagConvergenceConfig:
    lag_bars: int
    entry_threshold: float
    exit_threshold: float
    stop_threshold: float
    max_hold_bars: int


def _resolve_lag_convergence_config(params: Mapping[str, Any]) -> _LagConvergenceConfig:
    entry_threshold = float(params.get("entry_threshold", 0.015))
    return _LagConvergenceConfig(
        lag_bars=max(1, int(params.get("lag_bars", 3))),
        entry_threshold=entry_threshold,
        exit_threshold=max(0.0, float(params.get("exit_threshold", 0.004))),
        stop_threshold=max(entry_threshold + 1e-9, float(params.get("stop_threshold", 0.05))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 96))),
    )


def _lag_convergence_pair_positions(
    *,
    spread: np.ndarray,
    n: int,
    config: _LagConvergenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x_pos = np.zeros(n, dtype=float)
    y_pos = np.zeros(n, dtype=float)
    mode = 0
    bars_held = 0
    for idx in range(n):
        if idx < config.lag_bars:
            continue
        value = float(spread[idx])
        if mode == 0:
            if value <= -config.entry_threshold:
                mode = 1
                bars_held = 0
            elif value >= config.entry_threshold:
                mode = -1
                bars_held = 0
        else:
            bars_held += 1
            if (
                abs(value) <= config.exit_threshold
                or abs(value) >= config.stop_threshold
                or bars_held >= config.max_hold_bars
            ):
                mode = 0
                bars_held = 0
        if mode == 1:
            x_pos[idx] = 1.0
            y_pos[idx] = -1.0
        elif mode == -1:
            x_pos[idx] = -1.0
            y_pos[idx] = 1.0
    return x_pos, y_pos


def _apply_lag_convergence_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
) -> None:
    symbol_x, symbol_y, x_idx, y_idx = _resolve_symbol_pair(symbols, params)
    config = _resolve_lag_convergence_config(params)

    x_close = np.asarray(aligned[f"{symbol_x}:close"], dtype=float)
    y_close = np.asarray(aligned[f"{symbol_y}:close"], dtype=float)
    x_base = np.roll(x_close, config.lag_bars)
    y_base = np.roll(y_close, config.lag_bars)
    x_base[: config.lag_bars] = x_close[: config.lag_bars]
    y_base[: config.lag_bars] = y_close[: config.lag_bars]
    mom_x = np.divide(
        x_close,
        np.clip(x_base, 1e-12, np.inf),
        out=np.ones_like(x_close),
    ) - 1.0
    mom_y = np.divide(
        y_close,
        np.clip(y_base, 1e-12, np.inf),
        out=np.ones_like(y_close),
    ) - 1.0
    spread = np.nan_to_num(mom_x - mom_y, nan=0.0)

    x_pos, y_pos = _lag_convergence_pair_positions(
        spread=spread,
        n=n,
        config=config,
    )
    exposures[x_idx] = x_pos
    exposures[y_idx] = y_pos


@dataclass(frozen=True, slots=True)
class _PerpCrowdingCarryConfig:
    window: int
    mild_funding: float
    extreme_funding: float
    entry_threshold: float
    exit_threshold: float
    stop_loss_pct: float
    max_hold_bars: int
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _CrowdingSupportInputs:
    close: np.ndarray
    funding_rate: np.ndarray
    open_interest: np.ndarray
    liquidation_long_notional: np.ndarray
    liquidation_short_notional: np.ndarray
    mark_price: np.ndarray | None
    index_price: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _PerpCarryStepInput:
    funding_i: float
    score_i: float
    oi_delta_z_i: float
    close_i: float


def _resolve_perp_crowding_carry_config(
    params: Mapping[str, Any],
) -> _PerpCrowdingCarryConfig:
    return _PerpCrowdingCarryConfig(
        window=int(params.get("window", 96)),
        mild_funding=float(params.get("mild_funding", 0.0002)),
        extreme_funding=float(params.get("extreme_funding", 0.0012)),
        entry_threshold=float(params.get("entry_threshold", 0.30)),
        exit_threshold=float(params.get("exit_threshold", 0.10)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        max_hold_bars=int(params.get("max_hold_bars", 72)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _resolve_crowding_support_inputs(
    *,
    aligned: Mapping[str, np.ndarray],
    symbol: str,
) -> _CrowdingSupportInputs | None:
    funding = aligned.get(f"{symbol}:funding_rate")
    open_interest = aligned.get(f"{symbol}:open_interest")
    liquidation_long = aligned.get(f"{symbol}:liquidation_long_notional")
    liquidation_short = aligned.get(f"{symbol}:liquidation_short_notional")
    close = aligned.get(f"{symbol}:close")
    if (
        funding is None
        or open_interest is None
        or liquidation_long is None
        or liquidation_short is None
        or close is None
    ):
        return None

    mark_price = aligned.get(f"{symbol}:mark_price")
    index_price = aligned.get(f"{symbol}:index_price")
    return _CrowdingSupportInputs(
        close=np.asarray(close, dtype=float),
        funding_rate=np.asarray(funding, dtype=float),
        open_interest=np.asarray(open_interest, dtype=float),
        liquidation_long_notional=np.asarray(liquidation_long, dtype=float),
        liquidation_short_notional=np.asarray(liquidation_short, dtype=float),
        mark_price=None if mark_price is None else np.asarray(mark_price, dtype=float),
        index_price=None if index_price is None else np.asarray(index_price, dtype=float),
    )


def _note_support_data_symbol(
    meta: dict[str, Any],
    *,
    symbol: str,
    values: np.ndarray,
) -> None:
    if np.any(np.isfinite(values)):
        meta.setdefault("support_data_symbols", []).append(symbol)


def _finalize_missing_support_symbols(
    meta: dict[str, Any],
    *,
    missing_symbols: Sequence[str],
) -> None:
    if missing_symbols:
        meta["missing_support_data"] = True
        meta["missing_support_symbols"] = list(missing_symbols)


def _apply_perp_crowding_carry_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_perp_crowding_carry_config(params)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        support_inputs = _resolve_crowding_support_inputs(aligned=aligned, symbol=symbol)
        if support_inputs is None:
            missing_symbols.append(symbol)
            continue
        position, support = _perp_carry_position_series(
            support_inputs=support_inputs,
            config=config,
        )
        exposures[s_idx] = position
        _note_support_data_symbol(
            meta,
            symbol=symbol,
            values=np.asarray(support["crowding_score"], dtype=float),
        )

    _finalize_missing_support_symbols(meta, missing_symbols=missing_symbols)


def _pair_spread_z(px: np.ndarray, py: np.ndarray, window: int = 96) -> np.ndarray:
    n = min(px.size, py.size)
    out = np.full(n, np.nan, dtype=float)
    x = np.log(np.clip(px[-n:], 1e-12, np.inf))
    y = np.log(np.clip(py[-n:], 1e-12, np.inf))
    win = max(16, int(window))
    if n < win + 2:
        return out
    for idx in range(win, n + 1):
        x_tail = x[idx - win : idx]
        y_tail = y[idx - win : idx]
        vx = float(np.var(x_tail))
        if vx <= 1e-12:
            beta = 1.0
        else:
            beta = float(np.cov(x_tail, y_tail)[0, 1] / vx)
        spread_tail = y_tail - (beta * x_tail)
        mean = float(np.mean(spread_tail[:-1]))
        std = _safe_std(spread_tail[:-1])
        if std <= 1e-12:
            out[idx - 1] = 0.0
        else:
            out[idx - 1] = (float(spread_tail[-1]) - mean) / std
    return out


def _beta_neutral_residual_series(
    values: np.ndarray,
    benchmark: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    arr = np.asarray(values, dtype=float)
    bench = np.asarray(benchmark, dtype=float)
    n = min(arr.size, bench.size)
    if n < 3:
        return out

    residual_price = 100.0
    win = max(8, int(window))
    for idx in range(1, n):
        start = max(0, idx - win)
        asset_tail = arr[start : idx + 1]
        bench_tail = bench[start : idx + 1]
        if asset_tail.size < 3 or bench_tail.size != asset_tail.size:
            continue
        if not (
            np.all(np.isfinite(asset_tail))
            and np.all(np.isfinite(bench_tail))
            and np.all(asset_tail > 0.0)
            and np.all(bench_tail > 0.0)
        ):
            continue
        asset_rets = (asset_tail[1:] / asset_tail[:-1]) - 1.0
        bench_rets = (bench_tail[1:] / bench_tail[:-1]) - 1.0
        mean_asset = float(np.mean(asset_rets))
        mean_bench = float(np.mean(bench_rets))
        bench_var = float(np.mean((bench_rets - mean_bench) ** 2))
        if bench_var <= 1e-12:
            beta = 0.0
        else:
            cov = float(np.mean((asset_rets - mean_asset) * (bench_rets - mean_bench)))
            beta = cov / bench_var
        residual_ret = float(asset_rets[-1]) - (beta * float(bench_rets[-1]))
        residual_price = max(1e-9, residual_price * (1.0 + residual_ret))
        out[idx] = residual_price
    return out


@dataclass(frozen=True, slots=True)
class _ResidualBasketReversionConfig:
    residual_window: int
    entry_z: float
    exit_z: float
    rebalance_bars: int
    max_longs: int
    max_shorts: int
    stop_loss_pct: float
    allow_short: bool


def _resolve_residual_basket_reversion_config(
    params: Mapping[str, Any],
    *,
    residual_window_default: int = 48,
) -> _ResidualBasketReversionConfig:
    return _ResidualBasketReversionConfig(
        residual_window=max(8, int(params.get("residual_window", residual_window_default))),
        entry_z=float(params.get("entry_z", 1.8)),
        exit_z=max(0.0, float(params.get("exit_z", 0.4))),
        rebalance_bars=max(1, int(params.get("rebalance_bars", 2))),
        max_longs=max(0, int(params.get("max_longs", 1))),
        max_shorts=max(0, int(params.get("max_shorts", 1))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _apply_residual_basket_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
    entry_gate: np.ndarray | None = None,
    session_gated: bool = False,
) -> None:
    config = _resolve_residual_basket_reversion_config(
        params,
        residual_window_default=64 if session_gated else 48,
    )
    btc_symbol = canonical_symbol(str(params.get("btc_symbol") or "BTC/USDT"))
    if btc_symbol not in symbols:
        btc_symbol = canonical_symbol(str(symbols[0]))

    btc_close = np.asarray(aligned[f"{btc_symbol}:close"], dtype=float)
    close_map_np = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float)
        for symbol in symbols
    }
    residual_z_map = _residual_basket_reversion_z_map(
        symbols=symbols,
        close_map_np=close_map_np,
        btc_symbol=btc_symbol,
        btc_close=btc_close,
        config=config,
    )
    position_state = np.zeros(len(symbols), dtype=float)
    entry_price = np.full(len(symbols), np.nan, dtype=float)

    for idx in range(len(aligned["datetime"])):
        close_map = {symbol: float(close[idx]) for symbol, close in close_map_np.items()}
        _apply_residual_basket_reversion_exits(
            symbols=symbols,
            idx=idx,
            close_map=close_map,
            residual_z_map=residual_z_map,
            position_state=position_state,
            entry_price=entry_price,
            config=config,
        )

        if not _residual_basket_reversion_rebalance_due(
            idx=idx,
            config=config,
            entry_gate=entry_gate,
        ):
            exposures[:, idx] = position_state
            continue

        long_set, shorts = _residual_basket_reversion_targets(
            symbols=symbols,
            btc_symbol=btc_symbol,
            residual_z_map=residual_z_map,
            idx=idx,
            config=config,
        )

        _apply_residual_basket_reversion_targets_to_state(
            symbols=symbols,
            close_map=close_map,
            long_set=long_set,
            shorts=shorts,
            position_state=position_state,
            entry_price=entry_price,
        )
        exposures[:, idx] = position_state

    meta["cross_sectional"] = True
    meta["residualized_cross_sectional"] = True
    meta["btc_symbol"] = btc_symbol
    if session_gated:
        meta["session_gated"] = True


def _residual_basket_reversion_rebalance_due(
    *,
    idx: int,
    config: _ResidualBasketReversionConfig,
    entry_gate: np.ndarray | None,
) -> bool:
    return (
        idx >= config.residual_window
        and (idx + 1) % config.rebalance_bars == 0
        and (entry_gate is None or bool(entry_gate[idx]))
    )


def _apply_residual_basket_reversion_targets_to_state(
    *,
    symbols: Sequence[str],
    close_map: Mapping[str, float],
    long_set: set[str],
    shorts: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
) -> None:
    short_set = set(shorts)
    for s_idx, symbol in enumerate(symbols):
        next_position = 1.0 if symbol in long_set else -1.0 if symbol in short_set else 0.0
        if next_position == 0.0:
            position_state[s_idx] = 0.0
            entry_price[s_idx] = np.nan
            continue
        if position_state[s_idx] != next_position or not np.isfinite(entry_price[s_idx]):
            entry_price[s_idx] = close_map[symbol]
        position_state[s_idx] = next_position


def _residual_basket_reversion_z_map(
    *,
    symbols: Sequence[str],
    close_map_np: Mapping[str, np.ndarray],
    btc_symbol: str,
    btc_close: np.ndarray,
    config: _ResidualBasketReversionConfig,
) -> dict[str, np.ndarray]:
    residual_z_map: dict[str, np.ndarray] = {}
    for symbol in symbols:
        close = close_map_np[symbol]
        if symbol == btc_symbol:
            residual_z_map[symbol] = np.zeros(close.shape, dtype=float)
            continue
        residual_series = _beta_neutral_residual_series(close, btc_close, window=config.residual_window)
        residual_z_map[symbol] = np.nan_to_num(
            _rolling_z(residual_series, config.residual_window),
            nan=0.0,
        )
    return residual_z_map


def _apply_residual_basket_reversion_exits(
    *,
    symbols: Sequence[str],
    idx: int,
    close_map: Mapping[str, float],
    residual_z_map: Mapping[str, np.ndarray],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    config: _ResidualBasketReversionConfig,
) -> None:
    for s_idx, symbol in enumerate(symbols):
        if position_state[s_idx] == 0.0 or not np.isfinite(entry_price[s_idx]):
            continue
        close_i = close_map[symbol]
        stop_long = (
            position_state[s_idx] > 0.0
            and close_i <= float(entry_price[s_idx]) * (1.0 - config.stop_loss_pct)
        )
        stop_short = (
            position_state[s_idx] < 0.0
            and close_i >= float(entry_price[s_idx]) * (1.0 + config.stop_loss_pct)
        )
        z_i = float(residual_z_map[symbol][idx]) if symbol in residual_z_map else 0.0
        exit_hit = abs(z_i) <= config.exit_z
        if stop_long or stop_short or exit_hit:
            position_state[s_idx] = 0.0
            entry_price[s_idx] = np.nan


def _residual_basket_reversion_targets(
    *,
    symbols: Sequence[str],
    btc_symbol: str,
    residual_z_map: Mapping[str, np.ndarray],
    idx: int,
    config: _ResidualBasketReversionConfig,
) -> tuple[set[str], list[str]]:
    ranked = [
        (float(residual_z_map[symbol][idx]), symbol)
        for symbol in symbols
        if symbol != btc_symbol
    ]
    ranked.sort(key=lambda item: item[0])
    longs = [symbol for z_i, symbol in ranked if z_i <= -config.entry_z][: config.max_longs]
    shorts = [symbol for z_i, symbol in reversed(ranked) if z_i >= config.entry_z][: config.max_shorts]
    if not config.allow_short:
        shorts = []
    long_set = set(longs)
    shorts = [symbol for symbol in shorts if symbol not in long_set]
    return long_set, shorts


@dataclass(frozen=True, slots=True)
class _CrossAssetLiquidationContagionFadeConfig:
    window: int
    leader_liq_z_min: float
    return_shock_pct: float
    exit_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _CrossAssetLiquidationStepInput:
    leader_liq: float
    ret_z: float
    close_i: float


def _resolve_cross_asset_liquidation_contagion_fade_config(
    params: Mapping[str, Any],
) -> _CrossAssetLiquidationContagionFadeConfig:
    return _CrossAssetLiquidationContagionFadeConfig(
        window=max(8, int(params.get("window", 64))),
        leader_liq_z_min=float(params.get("leader_liq_z_min", 1.2)),
        return_shock_pct=float(params.get("return_shock_pct", 0.006)),
        exit_z=max(0.0, float(params.get("exit_z", 0.3))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _apply_cross_asset_liquidation_contagion_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_cross_asset_liquidation_contagion_fade_config(params)

    liq_z_map: dict[str, np.ndarray] = {}
    return_z_map: dict[str, np.ndarray] = {}
    close_map_np: dict[str, np.ndarray] = {}
    valid_symbols: list[str] = []
    for symbol in symbols:
        liq_long = aligned.get(f"{symbol}:liquidation_long_notional")
        liq_short = aligned.get(f"{symbol}:liquidation_short_notional")
        close = aligned.get(f"{symbol}:close")
        if liq_long is None or liq_short is None or close is None:
            continue
        support = _crowding_support_series(
            funding_rate=np.zeros(len(close), dtype=float),
            open_interest=np.ones(len(close), dtype=float),
            liquidation_long_notional=np.asarray(liq_long, dtype=float),
            liquidation_short_notional=np.asarray(liq_short, dtype=float),
            window=config.window,
        )
        liq_z_map[symbol] = np.asarray(support["liquidation_imbalance_z"], dtype=float)
        close_arr = np.asarray(close, dtype=float)
        close_map_np[symbol] = close_arr
        prev_close = np.r_[close_arr[0], close_arr[:-1]]
        returns = np.divide(
            close_arr,
            np.clip(prev_close, 1e-12, np.inf),
            out=np.ones_like(close_arr),
            where=np.isfinite(prev_close),
        ) - 1.0
        return_z_map[symbol] = np.nan_to_num(_rolling_z(np.nan_to_num(returns, nan=0.0), config.window), nan=0.0)
        valid_symbols.append(symbol)

    for s_idx, symbol in enumerate(symbols):
        if symbol not in valid_symbols:
            continue
        exposures[s_idx] = _cross_asset_liquidation_contagion_position_series(
            symbol=symbol,
            close_arr=close_map_np[symbol],
            valid_symbols=valid_symbols,
            liq_z_map=liq_z_map,
            return_z_map=return_z_map,
            config=config,
        )


def _cross_asset_liquidation_contagion_position_series(
    *,
    symbol: str,
    close_arr: np.ndarray,
    valid_symbols: Sequence[str],
    liq_z_map: Mapping[str, np.ndarray],
    return_z_map: Mapping[str, np.ndarray],
    config: _CrossAssetLiquidationContagionFadeConfig,
) -> np.ndarray:
    pos = np.zeros(close_arr.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    hold_bars = 0
    for idx in range(close_arr.size):
        step = _cross_asset_liquidation_step_input(
            idx=idx,
            symbol=symbol,
            valid_symbols=valid_symbols,
            liq_z_map=liq_z_map,
            return_z_map=return_z_map,
            close_arr=close_arr,
        )
        mode, entry_price, hold_bars, pos[idx] = _cross_asset_liquidation_step(
            mode=mode,
            entry_price=entry_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )
    return pos


def _cross_asset_liquidation_step_input(
    *,
    idx: int,
    symbol: str,
    valid_symbols: Sequence[str],
    liq_z_map: Mapping[str, np.ndarray],
    return_z_map: Mapping[str, np.ndarray],
    close_arr: np.ndarray,
) -> _CrossAssetLiquidationStepInput:
    leader_vals = [
        float(liq_z_map[leader][idx])
        for leader in valid_symbols
        if leader != symbol and np.isfinite(liq_z_map[leader][idx])
    ]
    leader_liq = float(np.mean(np.asarray(leader_vals, dtype=float))) if leader_vals else np.nan
    ret_z = float(return_z_map[symbol][idx]) if np.isfinite(return_z_map[symbol][idx]) else np.nan
    return _CrossAssetLiquidationStepInput(
        leader_liq=leader_liq,
        ret_z=ret_z,
        close_i=float(close_arr[idx]),
    )


def _cross_asset_liquidation_step(
    *,
    mode: int,
    entry_price: float | None,
    hold_bars: int,
    step: _CrossAssetLiquidationStepInput,
    config: _CrossAssetLiquidationContagionFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, hold_bars, 0.0

    if mode == 1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            (np.isfinite(step.ret_z) and step.ret_z >= -config.exit_z)
            or (next_hold_bars >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_hold_bars, 1.0

    if mode == -1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            (np.isfinite(step.ret_z) and step.ret_z <= config.exit_z)
            or (next_hold_bars >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_hold_bars, -1.0

    if not (np.isfinite(step.leader_liq) and np.isfinite(step.ret_z)):
        return 0, entry_price, hold_bars, 0.0
    if (
        step.leader_liq >= config.leader_liq_z_min
        and step.ret_z >= config.return_shock_pct
        and config.allow_short
    ):
        return -1, step.close_i, 0, -1.0
    if step.leader_liq <= -config.leader_liq_z_min and step.ret_z <= -config.return_shock_pct:
        return 1, step.close_i, 0, 1.0
    return 0, entry_price, hold_bars, 0.0


def _minute_of_day(values: np.ndarray) -> np.ndarray:
    datetimes = np.asarray(values, dtype="datetime64[m]")
    return np.mod(datetimes.astype("int64"), 1440)


def _rolling_realized_vol(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < 3:
        return out
    returns = np.zeros(arr.shape, dtype=float)
    valid = np.isfinite(arr[1:]) & np.isfinite(arr[:-1]) & (arr[1:] > 0.0) & (arr[:-1] > 0.0)
    returns[1:][valid] = np.log(arr[1:][valid] / arr[:-1][valid])
    win = max(4, int(window))
    for idx in range(win, arr.size + 1):
        tail = returns[idx - win + 1 : idx]
        if tail.size < win:
            continue
        out[idx - 1] = _safe_std(tail)
    return out


def _align_bundles(
    bundles: Sequence[SeriesBundle],
    *,
    feature_cache: Mapping[str, pl.DataFrame] | None = None,
) -> dict[str, np.ndarray] | None:
    if not bundles:
        return None
    common_datetime = _common_bundle_datetime(bundles)
    if common_datetime is None:
        return None

    aligned: dict[str, np.ndarray] = {
        "datetime": common_datetime,
    }
    for bundle in bundles:
        prefix = bundle.symbol
        indices = _aligned_bundle_indices(bundle, common_datetime)
        if indices is None:
            return None
        aligned[f"{prefix}:open"] = bundle.open[indices]
        aligned[f"{prefix}:high"] = bundle.high[indices]
        aligned[f"{prefix}:low"] = bundle.low[indices]
        aligned[f"{prefix}:close"] = bundle.close[indices]
        aligned[f"{prefix}:volume"] = bundle.volume[indices]
        feature_frame = None if feature_cache is None else feature_cache.get(prefix)
        _augment_aligned_bundle_features(
            aligned,
            prefix=prefix,
            common_datetime=common_datetime,
            feature_frame=feature_frame,
        )
    return aligned


def _common_bundle_datetime(
    bundles: Sequence[SeriesBundle],
) -> np.ndarray | None:
    if not bundles:
        return None
    common_datetime = np.asarray(bundles[0].datetime, dtype="datetime64[ms]")
    for bundle in bundles[1:]:
        common_datetime = np.intersect1d(
            common_datetime,
            np.asarray(bundle.datetime, dtype="datetime64[ms]"),
            assume_unique=False,
        )
        if common_datetime.size < _MIN_BARS:
            return None
    if common_datetime.size < _MIN_BARS:
        return None
    return common_datetime


def _aligned_bundle_indices(
    bundle: SeriesBundle,
    common_datetime: np.ndarray,
) -> np.ndarray | None:
    bundle_datetime = np.asarray(bundle.datetime, dtype="datetime64[ms]")
    indices = np.searchsorted(bundle_datetime, common_datetime)
    if np.any(indices >= bundle_datetime.size) or np.any(bundle_datetime[indices] != common_datetime):
        return None
    return indices


def _aligned_feature_points(
    *,
    common_datetime: np.ndarray,
    feature_frame: pl.DataFrame,
) -> pl.DataFrame | None:
    if feature_frame.is_empty():
        return None

    target = pl.DataFrame({"datetime": pl.Series("datetime", common_datetime)})
    feature_points = feature_frame.select(["datetime", *_FEATURE_POINT_COLUMNS])
    target_dtype = target.schema.get("datetime")
    feature_dtype = feature_points.schema.get("datetime")
    if target_dtype is not None and feature_dtype is not None and target_dtype != feature_dtype:
        feature_points = feature_points.with_columns(pl.col("datetime").cast(target_dtype))
    return target.join_asof(
        feature_points.sort("datetime"),
        on="datetime",
        strategy="backward",
    )


def _augment_aligned_bundle_features(
    aligned: dict[str, np.ndarray],
    *,
    prefix: str,
    common_datetime: np.ndarray,
    feature_frame: pl.DataFrame | None,
) -> None:
    if feature_frame is None:
        return

    joined = _aligned_feature_points(
        common_datetime=common_datetime,
        feature_frame=feature_frame,
    )
    if joined is None:
        return

    for field in _FEATURE_POINT_COLUMNS:
        aligned[f"{prefix}:{field}"] = joined.get_column(field).to_numpy()

    support = _crowding_support_series(
        funding_rate=np.asarray(joined.get_column("funding_rate").to_numpy(), dtype=float),
        open_interest=np.asarray(joined.get_column("open_interest").to_numpy(), dtype=float),
        mark_price=np.asarray(joined.get_column("mark_price").to_numpy(), dtype=float),
        index_price=np.asarray(joined.get_column("index_price").to_numpy(), dtype=float),
        liquidation_long_notional=np.asarray(
            joined.get_column("liquidation_long_notional").to_numpy(),
            dtype=float,
        ),
        liquidation_short_notional=np.asarray(
            joined.get_column("liquidation_short_notional").to_numpy(),
            dtype=float,
        ),
    )
    for key, values in support.items():
        aligned[f"{prefix}:{key}"] = values


def _perp_carry_position_series(
    *,
    support_inputs: _CrowdingSupportInputs,
    config: _PerpCrowdingCarryConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    support = _crowding_support_series(
        funding_rate=support_inputs.funding_rate,
        open_interest=support_inputs.open_interest,
        mark_price=support_inputs.mark_price,
        index_price=support_inputs.index_price,
        liquidation_long_notional=support_inputs.liquidation_long_notional,
        liquidation_short_notional=support_inputs.liquidation_short_notional,
        window=config.window,
    )
    score = np.asarray(support["crowding_score"], dtype=float)
    oi_delta_z = np.asarray(support["oi_delta_z"], dtype=float)
    funding = support_inputs.funding_rate
    close_arr = support_inputs.close

    position = np.zeros(close_arr.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0

    for idx in range(close_arr.size):
        step = _perp_carry_step_input(
            idx=idx,
            funding=funding,
            score=score,
            oi_delta_z=oi_delta_z,
            close_arr=close_arr,
        )
        mode, entry_price, bars_held, position[idx] = _perp_carry_step(
            mode=mode,
            entry_price=entry_price,
            bars_held=bars_held,
            step=step,
            config=config,
        )

    return position, support


def _perp_carry_step_input(
    *,
    idx: int,
    funding: np.ndarray,
    score: np.ndarray,
    oi_delta_z: np.ndarray,
    close_arr: np.ndarray,
) -> _PerpCarryStepInput:
    return _PerpCarryStepInput(
        funding_i=float(funding[idx]),
        score_i=float(score[idx]),
        oi_delta_z_i=float(oi_delta_z[idx]),
        close_i=float(close_arr[idx]),
    )


def _perp_carry_step(
    *,
    mode: int,
    entry_price: float | None,
    bars_held: int,
    step: _PerpCarryStepInput,
    config: _PerpCrowdingCarryConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, bars_held, float(mode)

    if mode != 0:
        next_bars_held = bars_held + 1
        if _perp_carry_should_exit(
            mode=mode,
            score_i=step.score_i,
            funding_i=step.funding_i,
            close_i=step.close_i,
            entry_price=entry_price,
            bars_held=next_bars_held,
            config=config,
        ):
            return 0, None, 0, 0.0
        return mode, entry_price, next_bars_held, float(mode)

    next_mode = _perp_carry_entry_mode(
        funding_i=step.funding_i,
        score_i=step.score_i,
        oi_delta_z_i=step.oi_delta_z_i,
        config=config,
    )
    if next_mode == 0:
        return 0, entry_price, 0, 0.0
    return next_mode, step.close_i, 0, float(next_mode)


def _perp_carry_should_exit(
    *,
    mode: int,
    score_i: float,
    funding_i: float,
    close_i: float,
    entry_price: float | None,
    bars_held: int,
    config: _PerpCrowdingCarryConfig,
) -> bool:
    if mode == 1:
        return (
            (np.isfinite(score_i) and float(score_i) <= config.exit_threshold)
            or (np.isfinite(funding_i) and float(funding_i) >= config.extreme_funding)
            or (bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
    return (
        (np.isfinite(score_i) and float(score_i) >= -config.exit_threshold)
        or (np.isfinite(funding_i) and float(funding_i) <= -config.extreme_funding)
        or (bars_held >= config.max_hold_bars)
        or (
            entry_price is not None
            and close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
        )
    )


def _perp_carry_entry_mode(
    *,
    funding_i: float,
    score_i: float,
    oi_delta_z_i: float,
    config: _PerpCrowdingCarryConfig,
) -> int:
    if not (np.isfinite(funding_i) and np.isfinite(score_i) and np.isfinite(oi_delta_z_i)):
        return 0

    carry_long = (
        (funding_i > 0.0)
        and (funding_i <= config.mild_funding)
        and (score_i >= config.entry_threshold)
    )
    crowded_long = (
        (funding_i >= config.extreme_funding)
        and (oi_delta_z_i > 0.0)
        and (score_i >= config.entry_threshold)
    )
    carry_short = (
        (funding_i < 0.0)
        and (abs(funding_i) <= config.mild_funding)
        and (score_i <= -config.entry_threshold)
    )
    crowded_short = (
        (funding_i <= -config.extreme_funding)
        and (oi_delta_z_i < 0.0)
        and (score_i <= -config.entry_threshold)
    )

    if carry_long and not crowded_long:
        return 1
    if config.allow_short and (crowded_long or (carry_short and not crowded_short)):
        return -1
    if crowded_short:
        return 1
    return 0


def _returns_from_close(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.zeros(closes.shape, dtype=float)
    return np.diff(closes, prepend=closes[0]) / np.clip(np.r_[closes[0], closes[:-1]], 1e-12, np.inf)


def _resolve_strategy_anchor_symbol(
    raw_symbol: Any,
    symbols: Sequence[str],
    *,
    default: str,
) -> str:
    resolved = canonical_symbol(str(raw_symbol or default))
    if resolved not in symbols and symbols:
        return canonical_symbol(str(symbols[0]))
    return resolved


@dataclass(frozen=True, slots=True)
class _CompositeTrendStrategyConfig:
    long_threshold: float
    short_threshold: float
    exit_score_cross: float
    te_min: float
    vr_min: float
    risk_target_vol: float
    max_signal_strength: float
    vol_window: int
    max_hold_bars: int
    allow_short: bool
    benchmark_regime_ma: int
    benchmark_symbol: str
    crowding_reduce_threshold: float
    crowding_block_threshold: float


@dataclass(frozen=True, slots=True)
class _CompositeTrendStepInput:
    close_i: float
    score_i: float
    long_gate_i: bool
    short_gate_i: bool
    strength: float
    blocked: bool


def _resolve_composite_trend_strategy_config(
    params: Mapping[str, Any],
    symbols: Sequence[str],
) -> _CompositeTrendStrategyConfig:
    return _CompositeTrendStrategyConfig(
        long_threshold=float(params.get("long_threshold", 0.55)),
        short_threshold=float(params.get("short_threshold", 0.55)),
        exit_score_cross=float(params.get("exit_score_cross", 0.05)),
        te_min=float(params.get("te_min", 0.25)),
        vr_min=float(params.get("vr_min", 0.85)),
        risk_target_vol=float(params.get("risk_target_vol", 0.004)),
        max_signal_strength=float(params.get("max_signal_strength", 2.0)),
        vol_window=int(params.get("vol_window", 120)),
        max_hold_bars=int(params.get("max_hold_bars", 640)),
        allow_short=bool(params.get("allow_short", True)),
        benchmark_regime_ma=max(0, int(params.get("benchmark_regime_ma", 0))),
        benchmark_symbol=_resolve_strategy_anchor_symbol(
            params.get("benchmark_symbol"),
            symbols,
            default="BTC/USDT",
        ),
        crowding_reduce_threshold=float(params.get("crowding_reduce_threshold", 0.55)),
        crowding_block_threshold=float(params.get("crowding_block_threshold", 0.85)),
    )


def _composite_trend_benchmark_long_gate(
    *,
    aligned: Mapping[str, np.ndarray],
    n: int,
    config: _CompositeTrendStrategyConfig,
) -> np.ndarray:
    gate = np.ones(n, dtype=bool)
    if config.benchmark_regime_ma <= 0:
        return gate

    benchmark_close = np.asarray(aligned[f"{config.benchmark_symbol}:close"], dtype=float)
    for idx in range(config.benchmark_regime_ma - 1, n):
        window = benchmark_close[idx - config.benchmark_regime_ma + 1 : idx + 1]
        if np.all(np.isfinite(window)):
            gate[idx] = bool(float(benchmark_close[idx]) >= _safe_mean(window))
    return gate


def _apply_composite_trend_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_composite_trend_strategy_config(params, symbols)
    benchmark_long_gate = _composite_trend_benchmark_long_gate(
        aligned=aligned,
        n=n,
        config=config,
    )

    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)

        ret8 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 8), 1e-12, np.inf))
        ret21 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 21), 1e-12, np.inf))
        ret55 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 55), 1e-12, np.inf))
        z8 = _rolling_z(ret8, 120)
        z21 = _rolling_z(ret21, 120)
        z55 = _rolling_z(ret55, 120)
        mom = (0.5 * np.nan_to_num(z8)) + (0.3 * np.nan_to_num(z21)) + (0.2 * np.nan_to_num(z55))
        te = np.nan_to_num(_trend_efficiency_series(close, 55), nan=0.0)
        vr = np.nan_to_num(_vol_ratio_series(close, 8, 55), nan=0.0)

        vol_shock = _rolling_z(np.nan_to_num(volume, nan=0.0), 64)
        score = mom * (1.0 + 0.25 * np.tanh(np.nan_to_num(vol_shock) / 2.0)) * np.clip(te, 0.0, 1.0)
        gate = (te >= config.te_min) & (vr >= config.vr_min)
        long_gate = gate & benchmark_long_gate

        crowding = aligned.get(f"{symbol}:crowding_score")
        exposures[s_idx] = _composite_trend_position_series(
            close=close,
            score=np.asarray(score, dtype=float),
            gate=np.asarray(gate, dtype=bool),
            long_gate=np.asarray(long_gate, dtype=bool),
            short_gate=np.asarray(gate, dtype=bool),
            crowding=None if crowding is None else np.asarray(crowding, dtype=float),
            config=config,
        )

    if config.benchmark_regime_ma > 0:
        meta["benchmark_regime_ma"] = config.benchmark_regime_ma
        meta["benchmark_symbol"] = config.benchmark_symbol
        meta["crash_aware_gate"] = True


@dataclass(frozen=True, slots=True)
class _TopCapTimeSeriesMomentumConfig:
    lookback_bars: int
    rebalance_bars: int
    signal_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    max_longs: int
    max_shorts: int
    min_price: float
    btc_regime_ma: int
    benchmark_drawdown_window: int
    benchmark_drawdown_limit: float
    residualize_btc: bool
    residualize_mean: bool
    btc_symbol: str


def _resolve_topcap_tsmom_config(
    params: Mapping[str, Any],
    symbols: Sequence[str],
) -> _TopCapTimeSeriesMomentumConfig:
    return _TopCapTimeSeriesMomentumConfig(
        lookback_bars=max(2, int(params.get("lookback_bars", 16))),
        rebalance_bars=max(1, int(params.get("rebalance_bars", 4))),
        signal_threshold=float(params.get("signal_threshold", 0.04)),
        stop_loss_pct=max(0.0, float(params.get("stop_loss_pct", 0.08))),
        take_profit_pct=max(0.0, float(params.get("take_profit_pct", 0.0))),
        max_longs=max(0, int(params.get("max_longs", 2))),
        max_shorts=max(0, int(params.get("max_shorts", 2))),
        min_price=max(0.0, float(params.get("min_price", 0.10))),
        btc_regime_ma=max(0, int(params.get("btc_regime_ma", 0))),
        benchmark_drawdown_window=max(0, int(params.get("benchmark_drawdown_window", 0))),
        benchmark_drawdown_limit=max(0.0, float(params.get("benchmark_drawdown_limit", 0.0))),
        residualize_btc=bool(params.get("residualize_btc", False)),
        residualize_mean=bool(params.get("residualize_mean", False)),
        btc_symbol=_resolve_strategy_anchor_symbol(
            params.get("btc_symbol"),
            symbols,
            default="BTC/USDT",
        ),
    )


def _apply_topcap_risk_exits(
    *,
    current_close_map: Mapping[str, float],
    symbols: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    config: _TopCapTimeSeriesMomentumConfig,
) -> None:
    if config.stop_loss_pct <= 0.0 and config.take_profit_pct <= 0.0:
        return

    for s_idx, symbol in enumerate(symbols):
        if position_state[s_idx] == 0.0 or not np.isfinite(entry_price[s_idx]):
            continue

        close_i = current_close_map[symbol]
        if not np.isfinite(close_i) or close_i <= 0.0:
            continue

        stop_long = position_state[s_idx] > 0.0 and close_i <= float(entry_price[s_idx]) * (1.0 - config.stop_loss_pct)
        stop_short = position_state[s_idx] < 0.0 and close_i >= float(entry_price[s_idx]) * (1.0 + config.stop_loss_pct)
        tp_long = (
            config.take_profit_pct > 0.0
            and position_state[s_idx] > 0.0
            and close_i >= float(entry_price[s_idx]) * (1.0 + config.take_profit_pct)
        )
        tp_short = (
            config.take_profit_pct > 0.0
            and position_state[s_idx] < 0.0
            and close_i <= float(entry_price[s_idx]) * (1.0 - config.take_profit_pct)
        )
        if stop_long or stop_short or tp_long or tp_short:
            position_state[s_idx] = 0.0
            entry_price[s_idx] = np.nan


def _topcap_ranked_momentum_rows(
    *,
    close_by_symbol: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    idx: int,
    config: _TopCapTimeSeriesMomentumConfig,
) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for symbol in symbols:
        close_arr = close_by_symbol[symbol]
        latest = float(close_arr[idx])
        base = float(close_arr[idx - config.lookback_bars])
        if latest < config.min_price or base <= 0.0 or not np.isfinite(latest) or not np.isfinite(base):
            continue
        momentum = (latest / base) - 1.0
        if np.isfinite(momentum):
            rows.append((float(momentum), symbol))
    return rows


def _topcap_market_regime(
    btc_close: np.ndarray,
    *,
    idx: int,
    config: _TopCapTimeSeriesMomentumConfig,
) -> str:
    regime = "BOTH"
    if (
        config.benchmark_drawdown_window > 0
        and config.benchmark_drawdown_limit > 0.0
        and idx >= config.benchmark_drawdown_window - 1
    ):
        window = btc_close[idx - config.benchmark_drawdown_window + 1 : idx + 1]
        if np.all(np.isfinite(window)):
            peak = float(np.max(window))
            if peak > 0.0:
                drawdown = (float(btc_close[idx]) / peak) - 1.0
                if drawdown <= -config.benchmark_drawdown_limit:
                    regime = "RISK_OFF"

    if regime != "RISK_OFF" and config.btc_regime_ma > 0 and idx >= config.btc_regime_ma:
        window = btc_close[idx - config.btc_regime_ma + 1 : idx + 1]
        if np.all(np.isfinite(window)):
            regime = "RISK_ON" if float(btc_close[idx]) >= _safe_mean(window) else "RISK_OFF"
    return regime


def _topcap_residualized_rows(
    momentum_rows: list[tuple[float, str]],
    *,
    config: _TopCapTimeSeriesMomentumConfig,
) -> list[tuple[float, str]]:
    if not momentum_rows:
        return momentum_rows

    if not config.residualize_btc and not config.residualize_mean:
        momentum_rows.sort(key=lambda item: item[0])
        return momentum_rows

    momentum_map = {symbol: momentum for momentum, symbol in momentum_rows}
    if config.residualize_btc and config.btc_symbol in momentum_map:
        btc_momentum = float(momentum_map[config.btc_symbol])
        for symbol in list(momentum_map):
            momentum_map[symbol] = float(momentum_map[symbol]) - btc_momentum
    if config.residualize_mean and momentum_map:
        mean_momentum = _safe_mean(np.asarray(list(momentum_map.values()), dtype=float))
        for symbol in list(momentum_map):
            momentum_map[symbol] = float(momentum_map[symbol]) - mean_momentum

    residualized = [(float(momentum_map[symbol]), symbol) for _, symbol in momentum_rows]
    residualized.sort(key=lambda item: item[0])
    return residualized


def _topcap_target_sets(
    momentum_rows: list[tuple[float, str]],
    *,
    regime: str,
    config: _TopCapTimeSeriesMomentumConfig,
) -> tuple[set[str], set[str]]:
    longs = [symbol for momentum, symbol in reversed(momentum_rows) if momentum >= config.signal_threshold][
        : config.max_longs
    ]
    shorts = [symbol for momentum, symbol in momentum_rows if momentum <= -config.signal_threshold][
        : config.max_shorts
    ]
    if regime == "RISK_ON":
        shorts = []
    elif regime == "RISK_OFF":
        longs = []

    long_set = set(longs)
    short_set = {symbol for symbol in shorts if symbol not in long_set}
    return long_set, short_set


def _apply_topcap_target_positions(
    *,
    current_close_map: Mapping[str, float],
    symbols: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    long_set: set[str],
    short_set: set[str],
) -> None:
    for s_idx, symbol in enumerate(symbols):
        next_position = 0.0
        if symbol in long_set:
            next_position = 1.0
        elif symbol in short_set:
            next_position = -1.0

        if next_position == 0.0:
            if position_state[s_idx] != 0.0:
                entry_price[s_idx] = np.nan
            position_state[s_idx] = 0.0
            continue

        if position_state[s_idx] != next_position or not np.isfinite(entry_price[s_idx]):
            entry_price[s_idx] = current_close_map[symbol]
        position_state[s_idx] = next_position


def _topcap_rebalance_due(
    *,
    idx: int,
    config: _TopCapTimeSeriesMomentumConfig,
) -> bool:
    return idx >= config.lookback_bars and (idx + 1) % config.rebalance_bars == 0


def _rebalance_topcap_positions(
    *,
    idx: int,
    current_close_map: Mapping[str, float],
    close_by_symbol: Mapping[str, np.ndarray],
    btc_close: np.ndarray,
    symbols: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    config: _TopCapTimeSeriesMomentumConfig,
) -> None:
    momentum_rows = _topcap_ranked_momentum_rows(
        close_by_symbol=close_by_symbol,
        symbols=symbols,
        idx=idx,
        config=config,
    )
    regime = _topcap_market_regime(
        btc_close,
        idx=idx,
        config=config,
    )
    ranked_rows = _topcap_residualized_rows(momentum_rows, config=config)
    long_set, short_set = _topcap_target_sets(
        ranked_rows,
        regime=regime,
        config=config,
    )
    _apply_topcap_target_positions(
        current_close_map=current_close_map,
        symbols=symbols,
        position_state=position_state,
        entry_price=entry_price,
        long_set=long_set,
        short_set=short_set,
    )


def _apply_topcap_tsmom_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_topcap_tsmom_config(params, symbols)
    close_by_symbol = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float)
        for symbol in symbols
    }
    btc_close = close_by_symbol[config.btc_symbol]
    position_state = np.zeros(len(symbols), dtype=float)
    entry_price = np.full(len(symbols), np.nan, dtype=float)

    for idx in range(n):
        current_close_map = {
            symbol: float(close_by_symbol[symbol][idx])
            for symbol in symbols
        }
        _apply_topcap_risk_exits(
            current_close_map=current_close_map,
            symbols=symbols,
            position_state=position_state,
            entry_price=entry_price,
            config=config,
        )

        if not _topcap_rebalance_due(idx=idx, config=config):
            exposures[:, idx] = position_state
            continue

        _rebalance_topcap_positions(
            idx=idx,
            current_close_map=current_close_map,
            close_by_symbol=close_by_symbol,
            btc_close=btc_close,
            symbols=symbols,
            position_state=position_state,
            entry_price=entry_price,
            config=config,
        )
        exposures[:, idx] = position_state

    meta["cross_sectional"] = True
    if config.take_profit_pct > 0.0:
        meta["take_profit_pct"] = config.take_profit_pct
    if config.residualize_btc or config.residualize_mean:
        meta["residualized_cross_sectional"] = True
        meta["residualize_btc"] = config.residualize_btc
        meta["residualize_mean"] = config.residualize_mean
    if config.benchmark_drawdown_window > 0 and config.benchmark_drawdown_limit > 0.0:
        meta["crash_aware_gate"] = True
        meta["benchmark_drawdown_window"] = config.benchmark_drawdown_window
        meta["benchmark_drawdown_limit"] = config.benchmark_drawdown_limit


@dataclass(frozen=True, slots=True)
class _MeanReversionStdConfig:
    window: int
    entry_z: float
    exit_z: float
    stop_loss_pct: float
    allow_short: bool
    residualize_btc: bool
    btc_symbol: str


def _resolve_mean_reversion_std_config(
    params: Mapping[str, Any],
    symbols: Sequence[str],
) -> _MeanReversionStdConfig:
    return _MeanReversionStdConfig(
        window=max(8, int(params.get("window", 64))),
        entry_z=float(params.get("entry_z", 2.0)),
        exit_z=max(0.0, float(params.get("exit_z", 0.5))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", True)),
        residualize_btc=bool(params.get("residualize_btc", False)),
        btc_symbol=_resolve_strategy_anchor_symbol(
            params.get("btc_symbol"),
            symbols,
            default="BTC/USDT",
        ),
    )


def _mean_reversion_position_series(
    *,
    close: np.ndarray,
    signal_series: np.ndarray,
    window: int,
    entry_z: float,
    exit_z: float,
    stop_loss_pct: float,
    allow_short: bool,
) -> np.ndarray:
    zscore = np.nan_to_num(_rolling_z(signal_series, window), nan=0.0)
    pos = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _mean_reversion_step_input(
            idx=idx,
            close=close,
            zscore=zscore,
        )
        mode, entry_price, pos[idx] = _mean_reversion_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_loss_pct=stop_loss_pct,
            allow_short=allow_short,
        )
    return pos


@dataclass(frozen=True, slots=True)
class _MeanReversionStepInput:
    close_i: float
    z_i: float


def _mean_reversion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    zscore: np.ndarray,
) -> _MeanReversionStepInput:
    return _MeanReversionStepInput(
        close_i=float(close[idx]),
        z_i=float(zscore[idx]),
    )


def _mean_reversion_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _MeanReversionStepInput,
    entry_z: float,
    exit_z: float,
    stop_loss_pct: float,
    allow_short: bool,
) -> tuple[int, float | None, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, 0.0

    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - stop_loss_pct)
        revert_hit = step.z_i >= -exit_z
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + stop_loss_pct)
        revert_hit = step.z_i <= exit_z
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0:
        if step.z_i <= -entry_z:
            return 1, step.close_i, 1.0
        if allow_short and step.z_i >= entry_z:
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_mean_reversion_std_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_mean_reversion_std_config(params, symbols)
    btc_close = None
    if config.residualize_btc and config.btc_symbol in symbols:
        btc_close = np.asarray(aligned[f"{config.btc_symbol}:close"], dtype=float)

    for s_idx, symbol in enumerate(symbols):
        if config.residualize_btc and symbol == config.btc_symbol:
            continue

        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        signal_series = close
        if btc_close is not None:
            signal_series = _beta_neutral_residual_series(close, btc_close, window=config.window)
        exposures[s_idx] = _mean_reversion_position_series(
            close=close,
            signal_series=signal_series,
            window=config.window,
            entry_z=config.entry_z,
            exit_z=config.exit_z,
            stop_loss_pct=config.stop_loss_pct,
            allow_short=config.allow_short,
        )

    if config.residualize_btc:
        meta["residualized_single_asset"] = True
        meta["residualize_btc"] = True
        meta["btc_symbol"] = config.btc_symbol


@dataclass(frozen=True, slots=True)
class _ShockReversionFadeConfig:
    volume_window: int
    range_window: int
    volume_shock_z: float
    range_shock_z: float
    return_shock_pct: float
    revert_fraction: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _ShockReversionStepInput:
    close_i: float
    ret_i: float
    shock_ok: bool


def _resolve_shock_reversion_fade_config(
    params: Mapping[str, Any],
    *,
    volume_window_default: int,
    range_window_default: int,
    volume_shock_z_default: float,
    range_shock_z_default: float,
    return_shock_pct_default: float,
    revert_fraction_default: float,
) -> _ShockReversionFadeConfig:
    return _ShockReversionFadeConfig(
        volume_window=max(8, int(params.get("volume_window", volume_window_default))),
        range_window=max(8, int(params.get("range_window", range_window_default))),
        volume_shock_z=float(params.get("volume_shock_z", volume_shock_z_default)),
        range_shock_z=float(params.get("range_shock_z", range_shock_z_default)),
        return_shock_pct=max(1e-6, float(params.get("return_shock_pct", return_shock_pct_default))),
        revert_fraction=min(0.95, max(0.10, float(params.get("revert_fraction", revert_fraction_default)))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _range_pct_from_prev_close(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    prev_close = np.r_[close[0], close[:-1]]
    return np.divide(
        high - low,
        np.clip(prev_close, 1e-12, np.inf),
        out=np.zeros_like(close),
        where=np.isfinite(prev_close),
    )


def _shock_reversion_support_series(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    config: _ShockReversionFadeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns = _returns_from_close(close)
    range_pct = _range_pct_from_prev_close(close=close, high=high, low=low)
    vol_z = np.nan_to_num(_rolling_z(np.nan_to_num(volume, nan=0.0), config.volume_window), nan=0.0)
    range_z = np.nan_to_num(_rolling_z(np.nan_to_num(range_pct, nan=0.0), config.range_window), nan=0.0)
    return returns, vol_z, range_z


def _shock_reversion_position_series(
    *,
    close: np.ndarray,
    returns: np.ndarray,
    vol_z: np.ndarray,
    range_z: np.ndarray,
    config: _ShockReversionFadeConfig,
    entry_gate: np.ndarray | None = None,
) -> np.ndarray:
    pos = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    target_price: float | None = None
    hold_bars = 0
    gate = np.ones(close.shape, dtype=bool) if entry_gate is None else np.asarray(entry_gate, dtype=bool)

    for idx in range(close.size):
        step = _shock_reversion_step_input(
            idx=idx,
            close=close,
            returns=returns,
            vol_z=vol_z,
            range_z=range_z,
            gate=gate,
            config=config,
        )
        mode, entry_price, target_price, hold_bars, pos[idx] = _shock_reversion_step(
            mode=mode,
            entry_price=entry_price,
            target_price=target_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )
    return pos


def _shock_reversion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    returns: np.ndarray,
    vol_z: np.ndarray,
    range_z: np.ndarray,
    gate: np.ndarray,
    config: _ShockReversionFadeConfig,
) -> _ShockReversionStepInput:
    close_i = float(close[idx])
    ret_i = float(returns[idx]) if np.isfinite(returns[idx]) else 0.0
    shock_ok = bool(gate[idx]) and float(vol_z[idx]) >= config.volume_shock_z and float(range_z[idx]) >= config.range_shock_z
    return _ShockReversionStepInput(
        close_i=close_i,
        ret_i=ret_i,
        shock_ok=shock_ok,
    )


def _shock_reversion_step(
    *,
    mode: int,
    entry_price: float | None,
    target_price: float | None,
    hold_bars: int,
    step: _ShockReversionStepInput,
    config: _ShockReversionFadeConfig,
) -> tuple[int, float | None, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, target_price, hold_bars, 0.0

    if mode == 1 and entry_price is not None and target_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        revert_hit = step.close_i >= target_price
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, None, 0, 0.0
        return 1, entry_price, target_price, next_hold_bars, 1.0

    if mode == -1 and entry_price is not None and target_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        revert_hit = step.close_i <= target_price
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, None, 0, 0.0
        return -1, entry_price, target_price, next_hold_bars, -1.0

    if mode == 0 and step.shock_ok:
        if step.ret_i <= -config.return_shock_pct:
            return (
                1,
                step.close_i,
                step.close_i * (1.0 + (abs(step.ret_i) * config.revert_fraction)),
                0,
                1.0,
            )
        if config.allow_short and step.ret_i >= config.return_shock_pct:
            return (
                -1,
                step.close_i,
                step.close_i * (1.0 - (abs(step.ret_i) * config.revert_fraction)),
                0,
                -1.0,
            )
    return mode, entry_price, target_price, hold_bars, 0.0


def _apply_liquidity_shock_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_shock_reversion_fade_config(
        params,
        volume_window_default=64,
        range_window_default=48,
        volume_shock_z_default=1.5,
        range_shock_z_default=1.0,
        return_shock_pct_default=0.01,
        revert_fraction_default=0.50,
    )
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)
        returns, vol_z, range_z = _shock_reversion_support_series(
            close=close,
            high=high,
            low=low,
            volume=volume,
            config=config,
        )
        exposures[s_idx] = _shock_reversion_position_series(
            close=close,
            returns=returns,
            vol_z=vol_z,
            range_z=range_z,
            config=config,
        )


def _apply_session_liquidity_vacuum_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_shock_reversion_fade_config(
        params,
        volume_window_default=48,
        range_window_default=36,
        volume_shock_z_default=1.0,
        range_shock_z_default=0.8,
        return_shock_pct_default=0.006,
        revert_fraction_default=0.40,
    )
    session_window_minutes = max(5, int(params.get("session_window_minutes", 30)))
    transition_minutes = np.asarray((0, 480, 780), dtype=int)
    minute_of_day = _minute_of_day(np.asarray(aligned["datetime"]))
    session_gate = np.any(
        np.abs(minute_of_day[:, None] - transition_minutes[None, :]) <= session_window_minutes,
        axis=1,
    )

    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)
        returns, vol_z, range_z = _shock_reversion_support_series(
            close=close,
            high=high,
            low=low,
            volume=volume,
            config=config,
        )
        exposures[s_idx] = _shock_reversion_position_series(
            close=close,
            returns=returns,
            vol_z=vol_z,
            range_z=range_z,
            config=config,
            entry_gate=session_gate,
        )


@dataclass(frozen=True, slots=True)
class _FundingLiquidationCrowdingFadeConfig:
    window: int
    crowding_entry: float
    crowding_exit: float
    liquidation_z_min: float
    return_shock_pct: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _FundingLiquidationCrowdingStepInput:
    close_i: float
    score_i: float
    liq_i: float
    ret_i: float


def _resolve_funding_liquidation_crowding_fade_config(
    params: Mapping[str, Any],
) -> _FundingLiquidationCrowdingFadeConfig:
    return _FundingLiquidationCrowdingFadeConfig(
        window=int(params.get("window", 96)),
        crowding_entry=float(params.get("crowding_entry", 0.85)),
        crowding_exit=float(params.get("crowding_exit", 0.25)),
        liquidation_z_min=float(params.get("liquidation_z_min", 1.0)),
        return_shock_pct=float(params.get("return_shock_pct", 0.01)),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _funding_liquidation_crowding_position_series(
    *,
    close: np.ndarray,
    score: np.ndarray,
    liquidation_z: np.ndarray,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> np.ndarray:
    ret = _returns_from_close(close)
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0

    for idx in range(close.size):
        step = _funding_liquidation_crowding_step_input(
            idx=idx,
            close=close,
            score=score,
            liquidation_z=liquidation_z,
            returns=ret,
        )
        mode, entry_price, bars_held, position[idx] = _funding_liquidation_crowding_step(
            mode=mode,
            entry_price=entry_price,
            bars_held=bars_held,
            step=step,
            config=config,
        )
    return position


def _funding_liquidation_crowding_step_input(
    *,
    idx: int,
    close: np.ndarray,
    score: np.ndarray,
    liquidation_z: np.ndarray,
    returns: np.ndarray,
) -> _FundingLiquidationCrowdingStepInput:
    return _FundingLiquidationCrowdingStepInput(
        close_i=float(close[idx]),
        score_i=float(score[idx]) if np.isfinite(score[idx]) else np.nan,
        liq_i=float(liquidation_z[idx]) if np.isfinite(liquidation_z[idx]) else np.nan,
        ret_i=float(returns[idx]) if np.isfinite(returns[idx]) else 0.0,
    )


def _funding_liquidation_crowding_step(
    *,
    mode: int,
    entry_price: float | None,
    bars_held: int,
    step: _FundingLiquidationCrowdingStepInput,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, bars_held, float(mode)

    if mode != 0:
        next_bars_held = bars_held + 1
        if _funding_liquidation_crowding_should_exit(
            mode=mode,
            score_i=step.score_i,
            close_i=step.close_i,
            entry_price=entry_price,
            bars_held=next_bars_held,
            config=config,
        ):
            return 0, None, 0, 0.0
        return mode, entry_price, next_bars_held, float(mode)

    next_mode = _funding_liquidation_crowding_entry_mode(
        score_i=step.score_i,
        liq_i=step.liq_i,
        ret_i=step.ret_i,
        config=config,
    )
    if next_mode != 0:
        return next_mode, step.close_i, 0, float(next_mode)
    return mode, entry_price, bars_held, 0.0


def _funding_liquidation_crowding_should_exit(
    *,
    mode: int,
    score_i: float,
    close_i: float,
    entry_price: float | None,
    bars_held: int,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> bool:
    if mode == 1:
        return (
            (np.isfinite(score_i) and score_i >= -config.crowding_exit)
            or (bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
    return (
        (np.isfinite(score_i) and score_i <= config.crowding_exit)
        or (bars_held >= config.max_hold_bars)
        or (
            entry_price is not None
            and close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
        )
    )


def _funding_liquidation_crowding_entry_mode(
    *,
    score_i: float,
    liq_i: float,
    ret_i: float,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> int:
    if not (np.isfinite(score_i) and np.isfinite(liq_i)):
        return 0

    long_signal = (
        score_i <= -config.crowding_entry
        and liq_i <= -config.liquidation_z_min
        and ret_i <= -config.return_shock_pct
    )
    short_signal = (
        config.allow_short
        and score_i >= config.crowding_entry
        and liq_i >= config.liquidation_z_min
        and ret_i >= config.return_shock_pct
    )
    if long_signal:
        return 1
    if short_signal:
        return -1
    return 0


def _apply_funding_liquidation_crowding_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_funding_liquidation_crowding_fade_config(params)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        support_inputs = _resolve_crowding_support_inputs(aligned=aligned, symbol=symbol)
        if support_inputs is None:
            missing_symbols.append(symbol)
            continue

        support = _crowding_support_series(
            funding_rate=support_inputs.funding_rate,
            open_interest=support_inputs.open_interest,
            mark_price=support_inputs.mark_price,
            index_price=support_inputs.index_price,
            liquidation_long_notional=support_inputs.liquidation_long_notional,
            liquidation_short_notional=support_inputs.liquidation_short_notional,
            window=config.window,
        )
        score = np.asarray(support["crowding_score"], dtype=float)
        liquidation_z = np.asarray(support["liquidation_imbalance_z"], dtype=float)
        exposures[s_idx] = _funding_liquidation_crowding_position_series(
            close=support_inputs.close,
            score=score,
            liquidation_z=liquidation_z,
            config=config,
        )
        _note_support_data_symbol(meta, symbol=symbol, values=score)

    _finalize_missing_support_symbols(meta, missing_symbols=missing_symbols)


def _apply_vol_compression_vwap_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    entry_z = float(params.get("entry_z", 1.5))
    exit_z = float(params.get("exit_z", 0.35))
    compression_pct = float(params.get("compression_percentile", 0.30))
    comp_vol_ratio = float(
        params.get("compression_vol_ratio", params.get("compression_threshold", 0.85))
    )

    for s_idx, symbol in enumerate(symbols):
        close = aligned[f"{symbol}:close"]
        high = aligned[f"{symbol}:high"]
        low = aligned[f"{symbol}:low"]
        volume = aligned[f"{symbol}:volume"]

        dev_z = np.nan_to_num(
            _vwap_dev_z(high, low, close, volume, window=60, z_window=120),
            nan=0.0,
        )
        vr = np.nan_to_num(_vol_ratio_series(close, 12, 60), nan=0.0)
        bw = np.nan_to_num(_rolling_z(np.abs(_returns_from_close(close)), 64), nan=0.0)
        compression = (vr <= comp_vol_ratio) & (bw <= _normal_cdf(compression_pct) * 2.0)

        pos = np.zeros(close.shape, dtype=float)
        pos = np.where(compression & (dev_z <= -entry_z), 1.0, pos)
        pos = np.where(compression & (dev_z >= entry_z), -1.0, pos)
        pos = np.where(np.abs(dev_z) <= exit_z, 0.0, pos)
        exposures[s_idx] = pos


def _apply_leadlag_spillover_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    leader_symbols = [symbol for symbol in symbols if symbol in _LEADERS]
    laggards = [symbol for symbol in symbols if symbol not in _LEADERS and symbol not in _METALS]
    entry_score = float(params.get("entry_score", params.get("entry_spillover", 0.35)))
    exit_score = float(params.get("exit_score", params.get("exit_spillover", 0.08)))
    lag_order = max(1, int(params.get("max_lag", 3)))

    if not leader_symbols or not laggards:
        return

    n = exposures.shape[1]
    leader_ret = np.zeros((len(leader_symbols), n), dtype=float)
    for idx, sym in enumerate(leader_symbols):
        leader_ret[idx] = _returns_from_close(aligned[f"{sym}:close"])
    leader_mean = np.mean(leader_ret, axis=0)

    decay = np.exp(-np.arange(1, lag_order + 1, dtype=float))
    decay /= np.sum(decay)

    for symbol in laggards:
        s_idx = symbols.index(symbol)
        pred = np.zeros(n, dtype=float)
        for lag in range(1, lag_order + 1):
            shifted = np.roll(leader_mean, lag)
            pred += decay[lag - 1] * shifted
        follower_ret = _returns_from_close(aligned[f"{symbol}:close"])
        score = np.zeros(n, dtype=float)
        sigma = _safe_std(follower_ret)
        if sigma > 1e-12:
            score = pred / sigma
        pos = np.where(
            score >= entry_score,
            1.0,
            np.where(score <= -entry_score, -1.0, 0.0),
        )
        pos = np.where(np.abs(score) <= exit_score, 0.0, pos)
        exposures[s_idx] = pos


def _apply_session_gated_residual_basket_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    session_window_minutes = max(5, int(params.get("session_window_minutes", 30)))
    transition_minutes = np.asarray((0, 480, 780), dtype=int)
    minute_of_day = _minute_of_day(np.asarray(aligned["datetime"]))
    session_gate = np.any(
        np.abs(minute_of_day[:, None] - transition_minutes[None, :]) <= session_window_minutes,
        axis=1,
    )
    _apply_residual_basket_reversion_strategy(
        params=params,
        aligned=aligned,
        symbols=symbols,
        exposures=exposures,
        meta=meta,
        entry_gate=session_gate,
        session_gated=True,
    )


def _apply_pair_spread_strategy(
    *,
    strategy_class: str,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    symbol_x, symbol_y, x_idx, y_idx = _resolve_symbol_pair(symbols, params)

    try:
        simulated = _simulate_event_driven_strategy_exposures(
            _load_event_driven_strategy_impl(strategy_class),
            params=params,
            aligned=aligned,
            symbols=(symbol_x, symbol_y),
        )
        exposures[x_idx] = simulated[0]
        exposures[y_idx] = simulated[1]
        meta["event_driven_proxy"] = True
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.35))
        lookback = int(params.get("lookback_window", 96))
        x_pos, y_pos = _pair_spread_fallback_exposures(
            aligned=aligned,
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            length=n,
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=lookback,
        )

        exposures[x_idx] = x_pos
        exposures[y_idx] = y_pos
        meta["event_driven_proxy"] = False
        meta["event_driven_proxy_error"] = str(exc)


def _apply_micro_range_expansion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    lookback = max(8, int(params.get("lookback", 30)))
    range_z_threshold = float(params.get("range_z_threshold", 1.5))
    volume_z_threshold = float(params.get("volume_z_threshold", 1.0))

    for s_idx, symbol in enumerate(symbols):
        high = aligned[f"{symbol}:high"]
        low = aligned[f"{symbol}:low"]
        close = aligned[f"{symbol}:close"]
        volume = aligned[f"{symbol}:volume"]
        range_pct = np.divide(high - low, np.clip(close, 1e-12, np.inf))
        range_z = np.nan_to_num(_rolling_z(range_pct, lookback), nan=0.0)
        vol_z = np.nan_to_num(_rolling_z(volume, lookback), nan=0.0)
        ret = _returns_from_close(close)
        breakout = np.where(ret >= 0.0, 1.0, -1.0)
        active = (range_z >= range_z_threshold) & (vol_z >= volume_z_threshold)
        exposures[s_idx] = np.where(active, breakout, 0.0)


def _apply_alpha101_formula_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    simulated = _simulate_event_driven_strategy_exposures(
        _load_event_driven_strategy_impl("Alpha101FormulaStrategy"),
        params=params,
        aligned=aligned,
        symbols=symbols,
    )
    exposures[:] = simulated
    meta["event_driven_proxy"] = True
    meta["formulaic_alpha101"] = True
    meta["alpha_id"] = int(params.get("alpha_id", 101))
    meta["alpha_param_override_count"] = len(
        params.get("alpha_param_overrides")
        if isinstance(params.get("alpha_param_overrides"), Mapping)
        else {}
    )


def _wrap_strategy_handler(
    func,
    *,
    include_n: bool = False,
    include_meta: bool = False,
    extra_kwargs: Mapping[str, Any] | None = None,
):
    extras = dict(extra_kwargs or {})

    def _handler(
        params: dict[str, Any],
        aligned: dict[str, np.ndarray],
        symbols: Sequence[str],
        n: int,
        exposures: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        kwargs = {
            "params": params,
            "aligned": aligned,
            "symbols": symbols,
            "exposures": exposures,
            **extras,
        }
        if include_n:
            kwargs["n"] = n
        if include_meta:
            kwargs["meta"] = meta
        func(**kwargs)

    return _handler


_STRATEGY_SIGNAL_DISPATCHER = StrategySignalDispatcher(
    handlers={
        "Alpha101FormulaStrategy": _wrap_strategy_handler(
            _apply_alpha101_formula_strategy,
            include_meta=True,
        ),
        "CompositeTrendStrategy": _wrap_strategy_handler(
            _apply_composite_trend_strategy,
            include_n=True,
            include_meta=True,
        ),
        "VolCompressionVWAPReversionStrategy": _wrap_strategy_handler(
            _apply_vol_compression_vwap_reversion_strategy,
        ),
        "VolCompressionVwapReversionStrategy": _wrap_strategy_handler(
            _apply_vol_compression_vwap_reversion_strategy,
        ),
        "VolatilityCompressionReversionStrategy": _wrap_strategy_handler(
            _apply_vol_compression_vwap_reversion_strategy,
        ),
        "LeadLagSpilloverStrategy": _wrap_strategy_handler(_apply_leadlag_spillover_strategy),
        "TopCapTimeSeriesMomentumStrategy": _wrap_strategy_handler(
            _apply_topcap_tsmom_strategy,
            include_n=True,
            include_meta=True,
        ),
        "MeanReversionStdStrategy": _wrap_strategy_handler(
            _apply_mean_reversion_std_strategy,
            include_meta=True,
        ),
        "LiquidityShockReversionStrategy": _wrap_strategy_handler(
            _apply_liquidity_shock_reversion_strategy,
        ),
        "SessionLiquidityVacuumFadeStrategy": _wrap_strategy_handler(
            _apply_session_liquidity_vacuum_fade_strategy,
        ),
        "FundingLiquidationCrowdingFadeStrategy": _wrap_strategy_handler(
            _apply_funding_liquidation_crowding_fade_strategy,
            include_meta=True,
        ),
        "BasisSnapbackReversionStrategy": _wrap_strategy_handler(
            _apply_basis_snapback_reversion_strategy,
            include_meta=True,
        ),
        "VolOfVolExhaustionFadeStrategy": _wrap_strategy_handler(
            _apply_vol_of_vol_exhaustion_fade_strategy,
        ),
        "ResidualBasketReversionStrategy": _wrap_strategy_handler(
            _apply_residual_basket_reversion_strategy,
            include_meta=True,
        ),
        "SessionGatedResidualBasketReversionStrategy": _wrap_strategy_handler(
            _apply_session_gated_residual_basket_reversion_strategy,
            include_meta=True,
        ),
        "BreadthThrustFailureReversalStrategy": _wrap_strategy_handler(
            _apply_breadth_thrust_failure_reversal_strategy,
            include_n=True,
            include_meta=True,
        ),
        "CrossAssetLiquidationContagionFadeStrategy": _wrap_strategy_handler(
            _apply_cross_asset_liquidation_contagion_fade_strategy,
        ),
        "MultiHorizonTrendExhaustionFadeStrategy": _wrap_strategy_handler(
            _apply_multi_horizon_trend_exhaustion_fade_strategy,
        ),
        "VwapReversionStrategy": _wrap_strategy_handler(_apply_vwap_reversion_strategy),
        "RollingBreakoutStrategy": _wrap_strategy_handler(_apply_rolling_breakout_strategy),
        "RegimeBreakoutCandidateStrategy": _wrap_strategy_handler(
            _apply_regime_breakout_candidate_strategy,
        ),
        "PairSpreadZScoreStrategy": _wrap_strategy_handler(
            _apply_pair_spread_strategy,
            include_n=True,
            include_meta=True,
            extra_kwargs={"strategy_class": "PairSpreadZScoreStrategy"},
        ),
        "PairTradingZScoreStrategy": _wrap_strategy_handler(
            _apply_pair_spread_strategy,
            include_n=True,
            include_meta=True,
            extra_kwargs={"strategy_class": "PairTradingZScoreStrategy"},
        ),
        "LagConvergenceStrategy": _wrap_strategy_handler(
            _apply_lag_convergence_strategy,
            include_n=True,
        ),
        "PerpCrowdingCarryStrategy": _wrap_strategy_handler(
            _apply_perp_crowding_carry_strategy,
            include_meta=True,
        ),
        "MicroRangeExpansion1sStrategy": _wrap_strategy_handler(
            _apply_micro_range_expansion_strategy,
        ),
    },
    minimum_symbol_counts={
        "PairSpreadZScoreStrategy": 2,
        "PairTradingZScoreStrategy": 2,
        "LagConvergenceStrategy": 2,
    },
)


def _strategy_signal(
    candidate: dict[str, Any],
    *,
    aligned: dict[str, np.ndarray],
    symbols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return _STRATEGY_SIGNAL_DISPATCHER.dispatch(candidate, aligned=aligned, symbols=symbols)


def _candidate_cost_rate(candidate: dict[str, Any]) -> float:
    strategy = str(candidate.get("strategy_class") or candidate.get("strategy") or "").lower()
    if "micro" in strategy:
        return 0.0012
    if "pair" in strategy:
        return 0.0008
    if "lagconvergence" in strategy or "lag_convergence" in strategy:
        return 0.0007
    if "leadlag" in strategy:
        return 0.0007
    return 0.0005


def _candidate_symbols_and_timeframe(
    candidate: Mapping[str, Any],
) -> tuple[list[str], str]:
    symbols = canonicalize_symbol_list(list(candidate.get("symbols") or []))
    timeframe = str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "1m")
    return symbols, timeframe


def _candidate_bundle_list(
    *,
    symbols: Sequence[str],
    timeframe: str,
    cache: Mapping[tuple[str, str], SeriesBundle],
) -> list[SeriesBundle]:
    bundles: list[SeriesBundle] = []
    for symbol in symbols:
        bundle = cache.get((symbol, timeframe))
        if bundle is None:
            continue
        bundles.append(bundle)
    return bundles


def _insufficient_candidate_result(
    candidate: dict[str, Any],
    *,
    symbols: Sequence[str],
    timeframe: str,
    cache: Mapping[tuple[str, str], SeriesBundle],
) -> dict[str, Any]:
    return {
        "error": "insufficient_data",
        "candidate": candidate,
        "returns": np.asarray([], dtype=float),
        "turnover": np.asarray([], dtype=float),
        "exposure": np.asarray([], dtype=float),
        "metadata": {
            "missing_symbols": [symbol for symbol in symbols if (symbol, timeframe) not in cache]
        },
    }


def _candidate_benchmark_series(
    *,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    timeframe: str,
    timestamps: np.ndarray,
    returns_size: int,
) -> np.ndarray:
    benchmark_entry = benchmark_cache.get(timeframe)
    if isinstance(benchmark_entry, Mapping):
        benchmark_datetime = benchmark_entry.get("datetime")
        benchmark_returns = benchmark_entry.get("returns")
        return _align_series_to_timestamps(
            timestamps,
            source_timestamps=np.asarray(
                benchmark_datetime if benchmark_datetime is not None else [],
                dtype="datetime64[ms]",
            ),
            values=np.asarray(benchmark_returns if benchmark_returns is not None else [], dtype=float),
        )

    benchmark = np.asarray(benchmark_entry if benchmark_entry is not None else [], dtype=float)
    if benchmark.size < returns_size:
        benchmark = np.zeros(returns_size, dtype=float)
    return benchmark[-returns_size:]


def _metrics_for_mask(
    *,
    returns: np.ndarray,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    timestamps: np.ndarray,
    mask: np.ndarray,
    periods_per_year: int,
    candidate_count: int,
) -> dict[str, Any]:
    return _compute_metrics(
        returns[mask],
        turnover=turnover[mask],
        exposure=exposure[mask],
        benchmark_returns=benchmark_returns[mask],
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
        metric_config=BacktestConfig,
        timestamps=timestamps[mask],
    )


def _candidate_stage_metrics(
    *,
    returns: np.ndarray,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark: np.ndarray,
    timestamps: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    periods_per_year: int,
    candidate_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_metrics = _metrics_for_mask(
        returns=returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark,
        timestamps=timestamps,
        mask=split_masks["train"],
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
    )
    val_metrics = _metrics_for_mask(
        returns=returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark,
        timestamps=timestamps,
        mask=split_masks["val"],
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
    )
    oos_metrics = _metrics_for_mask(
        returns=returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark,
        timestamps=timestamps,
        mask=split_masks["oos"],
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
    )
    return train_metrics, val_metrics, oos_metrics


def _candidate_oos_cost_stress_metrics(
    *,
    returns_raw: np.ndarray,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark: np.ndarray,
    timestamps: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    cost_rate: float,
    periods_per_year: int,
    candidate_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oos_mask = split_masks["oos"]
    oos_turnover = turnover[oos_mask]
    oos_x2 = returns_raw[oos_mask] - (oos_turnover * (cost_rate * 2.0))
    oos_x3 = returns_raw[oos_mask] - (oos_turnover * (cost_rate * 3.0))
    oos_stress_x2 = _compute_metrics(
        oos_x2,
        turnover=oos_turnover,
        exposure=exposure[oos_mask],
        benchmark_returns=benchmark[oos_mask],
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
        metric_config=BacktestConfig,
        timestamps=timestamps[oos_mask],
    )
    oos_stress_x3 = _compute_metrics(
        oos_x3,
        turnover=oos_turnover,
        exposure=exposure[oos_mask],
        benchmark_returns=benchmark[oos_mask],
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
        metric_config=BacktestConfig,
        timestamps=timestamps[oos_mask],
    )
    return oos_stress_x2, oos_stress_x3


def _apply_cost_stress_hard_rejects(
    *,
    hard_reject: dict[str, Any],
    oos_stress_x2: Mapping[str, Any],
    oos_stress_x3: Mapping[str, Any],
    scoring_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(hard_reject)
    cfg = _resolve_score_config(scoring_config)
    stress_cfg = dict(cfg.get("cost_stress_thresholds") or {})
    if float(oos_stress_x2.get("sharpe", 0.0)) < float(stress_cfg.get("x2_sharpe_min", 0.0)):
        updated["stress_x2_sharpe"] = float(oos_stress_x2.get("sharpe", 0.0))
    if float(oos_stress_x3.get("sharpe", 0.0)) < float(stress_cfg.get("x3_sharpe_min", -0.25)):
        updated["stress_x3_sharpe"] = float(oos_stress_x3.get("sharpe", 0.0))
    return updated


def _stage1_prefilter_score(
    result: Mapping[str, Any],
    *,
    stage1_weights: Mapping[str, Any],
    stage1_error_score: float,
) -> float:
    if result.get("error"):
        return float(stage1_error_score)
    train = dict(result.get("train") or {})
    return float(
        (float(stage1_weights["sharpe_weight"]) * float(train.get("sharpe", 0.0)))
        + (float(stage1_weights["return_weight"]) * float(train.get("return", 0.0)))
        - (float(stage1_weights["pbo_penalty"]) * float(train.get("pbo", 1.0)))
    )


@dataclass(frozen=True, slots=True)
class _CandidateSignalPayload:
    symbols: list[str]
    timeframe: str
    timestamps: np.ndarray
    returns_raw: np.ndarray
    returns: np.ndarray
    turnover: np.ndarray
    exposure: np.ndarray
    meta: dict[str, Any]
    cost_rate: float


@dataclass(frozen=True, slots=True)
class _CandidateMetricPayload:
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    oos_metrics: dict[str, Any]
    oos_stress_x2: dict[str, Any]
    oos_stress_x3: dict[str, Any]


def _load_candidate_signal_payload(
    candidate: dict[str, Any],
    *,
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
) -> _CandidateSignalPayload | None:
    symbols, timeframe = _candidate_symbols_and_timeframe(candidate)
    bundles = _candidate_bundle_list(symbols=symbols, timeframe=timeframe, cache=cache)
    aligned = _align_bundles(bundles, feature_cache=feature_cache)
    if aligned is None:
        return None

    returns_raw, turnover, exposure, meta = _strategy_signal(candidate, aligned=aligned, symbols=symbols)
    cost_rate = _candidate_cost_rate(candidate)
    timestamps = np.asarray(aligned.get("datetime"), dtype="datetime64[ms]")
    return _CandidateSignalPayload(
        symbols=symbols,
        timeframe=timeframe,
        timestamps=timestamps,
        returns_raw=returns_raw,
        returns=returns_raw - (turnover * cost_rate),
        turnover=turnover,
        exposure=exposure,
        meta=meta,
        cost_rate=float(cost_rate),
    )


def _evaluate_candidate_metric_payload(
    signal_payload: _CandidateSignalPayload,
    *,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    candidate_count: int,
    split: Mapping[str, Any] | None,
) -> _CandidateMetricPayload:
    split_masks = _split_masks_from_datetimes(signal_payload.timestamps, split=split)
    periods_per_year = int(_PERIODS_PER_YEAR.get(signal_payload.timeframe, 365))
    benchmark = _candidate_benchmark_series(
        benchmark_cache=benchmark_cache,
        timeframe=signal_payload.timeframe,
        timestamps=signal_payload.timestamps,
        returns_size=signal_payload.returns.size,
    )
    train_metrics, val_metrics, oos_metrics = _candidate_stage_metrics(
        returns=signal_payload.returns,
        turnover=signal_payload.turnover,
        exposure=signal_payload.exposure,
        benchmark=benchmark,
        timestamps=signal_payload.timestamps,
        split_masks=split_masks,
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
    )
    oos_stress_x2, oos_stress_x3 = _candidate_oos_cost_stress_metrics(
        returns_raw=signal_payload.returns_raw,
        turnover=signal_payload.turnover,
        exposure=signal_payload.exposure,
        benchmark=benchmark,
        timestamps=signal_payload.timestamps,
        split_masks=split_masks,
        cost_rate=signal_payload.cost_rate,
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
    )
    return _CandidateMetricPayload(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        oos_metrics=oos_metrics,
        oos_stress_x2=oos_stress_x2,
        oos_stress_x3=oos_stress_x3,
    )


def _evaluate_candidate_hurdles(
    metric_payload: _CandidateMetricPayload,
    *,
    scoring_config: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], bool, dict[str, Any]]:
    hurdle_fields, passed, hard_reject = _hurdle_fields(
        metric_payload.train_metrics,
        metric_payload.val_metrics,
        metric_payload.oos_metrics,
        scoring_config=scoring_config,
    )
    hard_reject = _apply_cost_stress_hard_rejects(
        hard_reject=hard_reject,
        oos_stress_x2=metric_payload.oos_stress_x2,
        oos_stress_x3=metric_payload.oos_stress_x3,
        scoring_config=scoring_config,
    )
    passed = bool(passed and not hard_reject)
    for stage in ("train", "val", "oos"):
        hurdle_fields[stage]["pass"] = bool(hurdle_fields[stage]["pass"] and passed)
    return hurdle_fields, passed, hard_reject


def _candidate_result_payload(
    candidate: dict[str, Any],
    *,
    signal_payload: _CandidateSignalPayload,
    metric_payload: _CandidateMetricPayload,
    hurdle_fields: dict[str, dict[str, Any]],
    passed: bool,
    hard_reject: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "timestamps": signal_payload.timestamps,
        "returns": signal_payload.returns,
        "turnover": signal_payload.turnover,
        "exposure": signal_payload.exposure,
        "train": metric_payload.train_metrics,
        "val": metric_payload.val_metrics,
        "oos": metric_payload.oos_metrics,
        "oos_cost_stress": {
            "x2": {
                "sharpe": float(metric_payload.oos_stress_x2.get("sharpe", 0.0)),
                "return": float(metric_payload.oos_stress_x2.get("return", 0.0)),
            },
            "x3": {
                "sharpe": float(metric_payload.oos_stress_x3.get("sharpe", 0.0)),
                "return": float(metric_payload.oos_stress_x3.get("return", 0.0)),
            },
        },
        "hurdle_fields": hurdle_fields,
        "pass": bool(passed),
        "hard_reject_reasons": hard_reject,
        "metadata": {
            "strategy_family": _family_from_strategy(str(candidate.get("strategy_class") or "")),
            "cost_rate": float(signal_payload.cost_rate),
            "aligned_bars": int(signal_payload.timestamps.size),
            **signal_payload.meta,
        },
    }


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    candidate_count: int,
    scoring_config: Mapping[str, Any] | None = None,
    split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    signal_payload = _load_candidate_signal_payload(
        candidate,
        cache=cache,
        feature_cache=feature_cache,
    )
    if signal_payload is None:
        symbols, timeframe = _candidate_symbols_and_timeframe(candidate)
        return _insufficient_candidate_result(
            candidate,
            symbols=symbols,
            timeframe=timeframe,
            cache=cache,
        )

    metric_payload = _evaluate_candidate_metric_payload(
        signal_payload,
        benchmark_cache=benchmark_cache,
        candidate_count=candidate_count,
        split=split,
    )
    hurdle_fields, passed, hard_reject = _evaluate_candidate_hurdles(
        metric_payload,
        scoring_config=scoring_config,
    )
    return _candidate_result_payload(
        candidate,
        signal_payload=signal_payload,
        metric_payload=metric_payload,
        hurdle_fields=hurdle_fields,
        passed=passed,
        hard_reject=hard_reject,
    )


def _candidate_instability_penalty(
    row: dict[str, Any],
    *,
    scoring_config: Mapping[str, Any] | None = None,
) -> float:
    cfg = _resolve_score_config(scoring_config)
    weights = dict(cfg["candidate_rank_score_weights"])
    val = dict(row.get("val") or {})
    oos = dict(row.get("oos") or {})
    sharpe_gap = max(0.0, float(val.get("sharpe", 0.0)) - float(oos.get("sharpe", 0.0)))
    return_gap = max(0.0, float(val.get("return", 0.0)) - float(oos.get("return", 0.0)))
    turnover_gap = max(0.0, float(oos.get("turnover", 0.0)) - float(val.get("turnover", 0.0)))
    return float(
        (float(weights["instability_sharpe_penalty"]) * sharpe_gap)
        + (float(weights["instability_return_penalty"]) * return_gap)
        + (float(weights["instability_turnover_penalty"]) * turnover_gap)
    )


def _candidate_rank_score(row: dict[str, Any], *, scoring_config: Mapping[str, Any] | None = None) -> float:
    cfg = _resolve_score_config(scoring_config)
    weights = dict(cfg["candidate_rank_score_weights"])
    oos = dict(row.get("oos") or {})
    return float(
        (float(weights["sharpe_weight"]) * float(oos.get("sharpe", 0.0)))
        + (float(weights["deflated_sharpe_weight"]) * float(oos.get("deflated_sharpe", 0.0)))
        - (float(weights["pbo_penalty"]) * float(oos.get("pbo", 1.0)))
        + (float(weights["return_weight"]) * float(oos.get("return", 0.0)))
        - (
            float(weights["turnover_penalty"])
            * max(0.0, float(oos.get("turnover", 0.0)) - float(weights["turnover_threshold"]))
        )
        - (float(weights["drawdown_penalty"]) * float(oos.get("mdd", 0.0)))
        - _candidate_instability_penalty(row, scoring_config=scoring_config)
    )


def _read_csv_ohlcv(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path)
    cols = {str(col).lower(): col for col in frame.columns}
    required = {
        "open": cols.get("open"),
        "high": cols.get("high"),
        "low": cols.get("low"),
        "close": cols.get("close"),
        "volume": cols.get("volume"),
    }
    if any(value is None for value in required.values()):
        return pl.DataFrame()

    if "datetime" in cols:
        dt_col = cols["datetime"]
        dt_expr = pl.col(dt_col).cast(pl.Datetime(time_unit="ms"), strict=False)
    elif "timestamp" in cols:
        dt_col = cols["timestamp"]
        dt_expr = pl.from_epoch(pl.col(dt_col).cast(pl.Int64), time_unit="s").cast(pl.Datetime(time_unit="ms"))
    else:
        # Synthesize monotonic timestamps.
        dt_expr = pl.int_range(0, frame.height, eager=False).cast(pl.Int64) * 1000
        dt_expr = pl.from_epoch(dt_expr, time_unit="ms").cast(pl.Datetime(time_unit="ms"))

    out = (
        frame.select(
            [
                dt_expr.alias("datetime"),
                pl.col(required["open"]).cast(pl.Float64).alias("open"),
                pl.col(required["high"]).cast(pl.Float64).alias("high"),
                pl.col(required["low"]).cast(pl.Float64).alias("low"),
                pl.col(required["close"]).cast(pl.Float64).alias("close"),
                pl.col(required["volume"]).cast(pl.Float64).alias("volume"),
            ]
        )
        .drop_nulls()
        .sort("datetime")
    )
    return out


def _synthetic_bundle(
    symbol: str,
    timeframe: str,
    *,
    bars: int = 2400,
    start_date: Any = None,
    end_date: Any = None,
) -> SeriesBundle:
    step_seconds, bars, start = _synthetic_bundle_window(
        timeframe=timeframe,
        bars=bars,
        start_date=start_date,
        end_date=end_date,
    )
    open_, high, low, close, volume, dt = _synthetic_bundle_arrays(
        symbol=symbol,
        timeframe=timeframe,
        bars=bars,
        start=start,
        step_seconds=step_seconds,
    )

    return SeriesBundle(
        symbol=symbol,
        timeframe=timeframe,
        datetime=dt,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _synthetic_bundle_window(
    *,
    timeframe: str,
    bars: int,
    start_date: Any,
    end_date: Any,
) -> tuple[int, int, datetime]:
    step_seconds = {
        "1s": 1,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14_400,
        "1d": 86_400,
    }.get(timeframe, 60)
    start_bound = _coerce_utc_datetime(start_date)
    end_bound = _coerce_utc_datetime(end_date, end_of_day=True)
    if start_bound is not None and end_bound is not None and end_bound > start_bound:
        requested_bars = int(((end_bound - start_bound).total_seconds()) // step_seconds) + 1
        bars = max(_MIN_BARS, min(max(bars, requested_bars), 20_000))
        start = start_bound if requested_bars <= bars else end_bound - timedelta(seconds=(bars - 1) * step_seconds)
        return step_seconds, bars, start
    return step_seconds, bars, datetime.now(UTC) - timedelta(seconds=bars * step_seconds)


def _synthetic_bundle_arrays(
    *,
    symbol: str,
    timeframe: str,
    bars: int,
    start: datetime,
    step_seconds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(_hash_seed("synthetic", symbol, timeframe))
    open_ = np.zeros(bars, dtype=float)
    high = np.zeros(bars, dtype=float)
    low = np.zeros(bars, dtype=float)
    close = np.zeros(bars, dtype=float)
    volume = np.zeros(bars, dtype=float)
    dt = np.zeros(bars, dtype="datetime64[ms]")

    base = 100.0 + (20.0 * _hash_unit_interval(symbol, timeframe, "base"))
    drift = 0.00002 + (0.00008 * _hash_unit_interval(symbol, timeframe, "drift"))
    phase = 2.0 * math.pi * _hash_unit_interval(symbol, timeframe, "phase")
    price = base
    for idx in range(bars):
        regime = math.sin((idx / max(50.0, bars / 18.0)) + phase)
        step = drift + (0.001 * regime) + rng.gauss(0.0, 0.0025)

        o = max(0.1, price)
        c = max(0.1, o * (1.0 + step))
        wiggle = abs(rng.gauss(0.0, 0.0018)) + 0.0003
        open_[idx] = o
        high[idx] = max(o, c) * (1.0 + wiggle)
        low[idx] = min(o, c) * (1.0 - wiggle)
        close[idx] = c
        volume[idx] = max(1.0, 1200.0 * (1.0 + abs(regime)) + rng.uniform(-200.0, 200.0))
        dt[idx] = np.datetime64((start + timedelta(seconds=idx * step_seconds)).replace(tzinfo=None), "ms")
        price = c
    return open_, high, low, close, volume, dt


def _frame_to_bundle(symbol: str, timeframe: str, frame: pl.DataFrame) -> SeriesBundle:
    sorted_frame = frame.sort("datetime")
    return SeriesBundle(
        symbol=symbol,
        timeframe=timeframe,
        datetime=sorted_frame["datetime"].to_numpy(),
        open=sorted_frame["open"].to_numpy(),
        high=sorted_frame["high"].to_numpy(),
        low=sorted_frame["low"].to_numpy(),
        close=sorted_frame["close"].to_numpy(),
        volume=sorted_frame["volume"].to_numpy(),
    )


def _load_timeframe_parquet_frames(
    *,
    symbols: Sequence[str],
    timeframe: str,
    parquet_root: str,
    exchange: str,
    start_date: Any,
    end_date: Any,
    data_mode: str,
) -> dict[str, pl.DataFrame]:
    try:
        return load_data_dict_from_parquet(
            parquet_root,
            exchange=exchange,
            symbol_list=list(symbols),
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            data_mode=str(data_mode or "legacy"),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return {}


def _csv_bundle_candidates(symbol: str) -> list[Path]:
    compact = symbol.replace("/", "")
    return [
        Path("data") / f"{compact}.csv",
        Path("data") / f"{symbol}.csv",
        Path("data") / f"{symbol.replace('/', '_')}.csv",
    ]


def _filter_csv_frame_date_bounds(
    frame_csv: pl.DataFrame,
    *,
    start_date: Any,
    end_date: Any,
) -> pl.DataFrame:
    if frame_csv.is_empty() or (start_date is None and end_date is None):
        return frame_csv

    start_bound = _coerce_utc_datetime(start_date)
    end_bound = _coerce_utc_datetime(end_date, end_of_day=True)
    if start_bound is not None:
        frame_csv = frame_csv.filter(pl.col("datetime") >= start_bound.replace(tzinfo=None))
    if end_bound is not None:
        frame_csv = frame_csv.filter(pl.col("datetime") <= end_bound.replace(tzinfo=None))
    return frame_csv


def _load_csv_bundle(
    *,
    symbol: str,
    timeframe: str,
    start_date: Any,
    end_date: Any,
    min_bars: int,
) -> SeriesBundle | None:
    for csv_path in _csv_bundle_candidates(symbol):
        if not csv_path.exists():
            continue
        try:
            frame_csv = _read_csv_ohlcv(csv_path)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            frame_csv = pl.DataFrame()
        frame_csv = _filter_csv_frame_date_bounds(
            frame_csv,
            start_date=start_date,
            end_date=end_date,
        )
        if frame_csv.is_empty() or frame_csv.height < max(1, int(min_bars)):
            continue
        return _frame_to_bundle(symbol, timeframe, frame_csv)
    return None


def _strict_missing_bundle_error(*, symbol: str, timeframe: str) -> RawFirstDataMissingError:
    return RawFirstDataMissingError(
        f"Real market data missing for {symbol}@{timeframe} in strict mode."
    )


def _synthetic_disabled_bundle_error(*, symbol: str, timeframe: str) -> RawFirstDataMissingError:
    return RawFirstDataMissingError(
        f"Synthetic fallback is disabled and real market data is missing for {symbol}@{timeframe}."
    )


def _bundle_from_frame(
    *,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame | None,
    min_bars: int,
) -> SeriesBundle | None:
    if frame is None or frame.is_empty() or frame.height < max(1, int(min_bars)):
        return None
    return _frame_to_bundle(symbol, timeframe, frame)


def _resolve_bundle_cache_entry(
    *,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame | None,
    start_date: Any,
    end_date: Any,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bars: int,
) -> tuple[SeriesBundle, str]:
    parquet_bundle = _bundle_from_frame(
        symbol=symbol,
        timeframe=timeframe,
        frame=frame,
        min_bars=min_bars,
    )
    if parquet_bundle is not None:
        return parquet_bundle, "parquet"

    if not allow_csv_fallback and not allow_synthetic_fallback:
        raise _strict_missing_bundle_error(symbol=symbol, timeframe=timeframe)

    if allow_csv_fallback:
        csv_bundle = _load_csv_bundle(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            min_bars=min_bars,
        )
        if csv_bundle is not None:
            return csv_bundle, "csv"

    if not allow_synthetic_fallback:
        raise _synthetic_disabled_bundle_error(symbol=symbol, timeframe=timeframe)

    return _synthetic_bundle(symbol, timeframe, start_date=start_date, end_date=end_date), "synthetic"


def _load_bundle_cache(
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    start_date: Any = None,
    end_date: Any = None,
    data_mode: str = "legacy",
    allow_csv_fallback: bool = True,
    allow_synthetic_fallback: bool = True,
    min_bars: int = _MIN_BARS,
    market_data_settings: Mapping[str, Any] | None = None,
) -> tuple[dict[tuple[str, str], SeriesBundle], dict[str, list[str]]]:
    cache: dict[tuple[str, str], SeriesBundle] = {}
    source_map: dict[str, list[str]] = {"parquet": [], "csv": [], "synthetic": []}

    defaults = _current_research_market_data_settings(market_data_settings)
    parquet_root = str(defaults["parquet_root"])
    exchange = str(defaults["exchange"])

    for timeframe in timeframes:
        loaded = _load_timeframe_parquet_frames(
            symbols=symbols,
            timeframe=timeframe,
            parquet_root=parquet_root,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            data_mode=data_mode,
        )

        for symbol in symbols:
            key = (symbol, timeframe)
            bundle, source = _resolve_bundle_cache_entry(
                symbol=symbol,
                timeframe=timeframe,
                frame=loaded.get(symbol),
                start_date=start_date,
                end_date=end_date,
                allow_csv_fallback=allow_csv_fallback,
                allow_synthetic_fallback=allow_synthetic_fallback,
                min_bars=min_bars,
            )
            cache[key] = bundle
            source_map[source].append(f"{symbol}@{timeframe}")

    return cache, source_map


def _benchmark_cache(
    cache: Mapping[tuple[str, str], SeriesBundle],
    timeframes: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for tf in timeframes:
        bundle = cache.get(("BTC/USDT", tf))
        if bundle is None:
            out[tf] = {
                "datetime": np.asarray([], dtype="datetime64[ms]"),
                "returns": np.asarray([], dtype=float),
            }
        else:
            out[tf] = {
                "datetime": np.asarray(bundle.datetime, dtype="datetime64[ms]"),
                "returns": _returns_from_close(bundle.close),
            }
    return out


def _build_default_split(strategy_timeframe: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    train_start = now - timedelta(days=360)
    train_end = now - timedelta(days=150)
    val_start = train_end + timedelta(days=1)
    val_end = now - timedelta(days=60)
    oos_start = val_end + timedelta(days=1)
    oos_end = now
    return {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "val_start": val_start.isoformat(),
        "val_end": val_end.isoformat(),
        "oos_start": oos_start.isoformat(),
        "oos_end": oos_end.isoformat(),
        "strategy_timeframe": str(strategy_timeframe),
    }


@dataclass(frozen=True, slots=True)
class _ResearchRunScoringConfig:
    resolved_scoring_config: dict[str, Any]
    keep_ratio_applied: float
    stage1_weights: dict[str, Any]
    stage1_error_score: float
    failed_candidate_selection_score: float
    sort_missing_selection_score: float


def _normalize_candidate_research_base_timeframe(base_timeframe: str) -> str:
    base_tf = str(base_timeframe).strip().lower() or "1s"
    return "1s" if base_tf != "1s" else base_tf


def _resolve_research_run_scoring_config(
    *,
    score_config: Mapping[str, Any] | None,
    stage1_keep_ratio: float,
) -> _ResearchRunScoringConfig:
    resolved_scoring_config = _resolve_score_config(score_config)
    keep_ratio_cfg = dict(resolved_scoring_config["keep_ratio_bounds"])
    score_fallbacks = dict(resolved_scoring_config["score_fallbacks"])
    return _ResearchRunScoringConfig(
        resolved_scoring_config=resolved_scoring_config,
        keep_ratio_applied=max(
            float(keep_ratio_cfg["min"]),
            min(float(keep_ratio_cfg["max"]), float(stage1_keep_ratio)),
        ),
        stage1_weights=dict(resolved_scoring_config["stage1_prefilter_weights"]),
        stage1_error_score=float(score_fallbacks["stage1_error_score"]),
        failed_candidate_selection_score=float(score_fallbacks["failed_candidate_selection_score"]),
        sort_missing_selection_score=float(score_fallbacks["sort_missing_selection_score"]),
    )


def _adapt_candidate_inputs(
    candidates: Iterable[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    adapted = [adapt_legacy_candidate(item) for item in candidates]
    if int(max_candidates) > 0:
        adapted = adapted[: int(max_candidates)]
    return adapted


def _empty_candidate_research_report(
    *,
    base_timeframe: str,
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
    stage1_keep_ratio: float,
    scoring: _ResearchRunScoringConfig,
    split: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized_timeframes = normalize_strategy_timeframes(
        strategy_timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
        required=CANONICAL_STRATEGY_TIMEFRAMES,
        strict_subset=True,
    )
    empty_split = _resolve_split_config(
        split,
        strategy_timeframe=normalized_timeframes[0] if normalized_timeframes else "1m",
    )
    return {
        "schema_version": "v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_timeframe": base_timeframe,
        "strategy_timeframes": normalized_timeframes,
        "symbol_universe": canonicalize_symbol_list(
            symbol_universe or _current_research_market_data_settings()["symbols"]
        ),
        "split": empty_split,
        "candidates": [],
        "stage1": {
            "input_count": 0,
            "selected_count": 0,
            "keep_ratio": float(stage1_keep_ratio),
            "keep_ratio_applied": float(scoring.keep_ratio_applied),
        },
        "scoring_config": scoring.resolved_scoring_config,
    }


def _resolve_research_run_timeframes_and_universe(
    *,
    adapted: Sequence[dict[str, Any]],
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    discovered_timeframes = sorted(
        {
            str(row.get("strategy_timeframe") or row.get("timeframe") or "1m").strip().lower()
            for row in adapted
        }
    )
    normalized_timeframes = normalize_strategy_timeframes(
        strategy_timeframes or discovered_timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
        required=CANONICAL_STRATEGY_TIMEFRAMES,
        strict_subset=True,
    )
    universe = canonicalize_symbol_list(
        symbol_universe or _current_research_market_data_settings()["symbols"]
    )
    candidate_symbols = canonicalize_symbol_list(
        itertools.chain.from_iterable(list(row.get("symbols") or []) for row in adapted)
    )
    if candidate_symbols:
        universe = canonicalize_symbol_list(
            list(dict.fromkeys(list(universe) + list(candidate_symbols)))
        )
    return normalized_timeframes, universe


def _research_resource_loader() -> ResearchResourceLoader:
    return ResearchResourceLoader(
        split_window_bounds=_split_window_bounds,
        datetime_to_iso_z=_datetime_to_iso_z,
        load_bundle_cache=_load_bundle_cache,
        load_feature_cache=_load_feature_cache,
        benchmark_cache=_benchmark_cache,
        canonicalize_symbol_list=canonicalize_symbol_list,
    )


def _load_research_run_resources(
    *,
    adapted: Sequence[dict[str, Any]],
    normalized_timeframes: Sequence[str],
    universe: Sequence[str],
    resolved_split: Mapping[str, Any],
    data_mode: str,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bundle_bars: int,
    market_data_settings: Mapping[str, Any] | None = None,
) -> tuple[
    dict[tuple[str, str], SeriesBundle],
    dict[str, list[str]],
    dict[str, pl.DataFrame],
    dict[str, dict[str, np.ndarray]],
]:
    return _research_resource_loader().load(
        adapted=adapted,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
        market_data_settings=market_data_settings,
    )


def _evaluate_candidate_with_optional_split(
    candidate: dict[str, Any],
    *,
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    candidate_count: int,
    scoring_config: Mapping[str, Any] | None,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    return ResearchStageSelector(
        evaluate_candidate=_evaluate_candidate,
        stage1_prefilter_score=_stage1_prefilter_score,
    ).evaluate_candidate_with_optional_split(
        candidate,
        cache=cache,
        feature_cache=feature_cache,
        benchmark_cache=benchmark_cache,
        candidate_count=candidate_count,
        scoring_config=scoring_config,
        split=split,
    )


def _select_stage2_results(
    *,
    adapted: Sequence[dict[str, Any]],
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    benchmark: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    scoring: _ResearchRunScoringConfig,
    resolved_split: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return ResearchStageSelector(
        evaluate_candidate=_evaluate_candidate,
        stage1_prefilter_score=_stage1_prefilter_score,
    ).select_stage2_results(
        adapted=adapted,
        cache=cache,
        feature_cache=feature_cache,
        benchmark=benchmark,
        scoring=scoring,
        resolved_split=resolved_split,
    )


def _research_report_builder() -> ResearchReportBuilder:
    return ResearchReportBuilder(
        split_masks_from_datetimes=_split_masks_from_datetimes,
        split_lengths=_split_lengths,
        compute_metrics=_compute_metrics,
        hurdle_fields=_hurdle_fields,
        family_from_strategy=_family_from_strategy,
        canonicalize_symbol_list=canonicalize_symbol_list,
        series_to_stream=_series_to_stream,
        candidate_rank_score=_candidate_rank_score,
        correlation=_correlation,
        periods_per_year=_PERIODS_PER_YEAR,
        metric_config=BacktestConfig,
    )


def _result_timestamps_and_split_masks(
    result: Mapping[str, Any],
    *,
    resolved_split: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    return _research_report_builder().result_timestamps_and_split_masks(
        result,
        resolved_split=resolved_split,
    )


def _error_candidate_report_payload(
    *,
    result: Mapping[str, Any],
    resolved_scoring_config: Mapping[str, Any],
    failed_candidate_selection_score: float,
    candidate_count: int,
) -> dict[str, Any]:
    return _research_report_builder().error_candidate_report_payload(
        result=result,
        resolved_scoring_config=resolved_scoring_config,
        failed_candidate_selection_score=failed_candidate_selection_score,
        candidate_count=candidate_count,
    )


def _candidate_return_streams(
    *,
    returns: np.ndarray,
    timestamps: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    has_aligned_timestamps: bool,
) -> dict[str, list[dict[str, float | int]]]:
    return _research_report_builder().candidate_return_streams(
        returns=returns,
        timestamps=timestamps,
        split_masks=split_masks,
        has_aligned_timestamps=has_aligned_timestamps,
    )


def _successful_candidate_report_payload(
    *,
    result: Mapping[str, Any],
    resolved_split: Mapping[str, Any],
    resolved_scoring_config: Mapping[str, Any],
) -> dict[str, Any]:
    return _research_report_builder().successful_candidate_report_payload(
        result=result,
        resolved_split=resolved_split,
        resolved_scoring_config=resolved_scoring_config,
    )


def _report_candidates_from_stage2_results(
    *,
    stage2_results: Sequence[dict[str, Any]],
    candidate_count: int,
    resolved_split: Mapping[str, Any],
    scoring: _ResearchRunScoringConfig,
) -> list[dict[str, Any]]:
    from lumina_quant.strategy_factory.research_entrypoints import (
        _report_candidates_from_stage2_results as _report_candidates_from_stage2_results_impl,
    )

    return _report_candidates_from_stage2_results_impl(
        stage2_results=stage2_results,
        candidate_count=candidate_count,
        resolved_split=resolved_split,
        scoring=scoring,
    )


def _attach_cross_candidate_correlations(report_candidates: Sequence[dict[str, Any]]) -> None:
    from lumina_quant.strategy_factory.research_entrypoints import (
        _attach_cross_candidate_correlations as _attach_cross_candidate_correlations_impl,
    )

    _attach_cross_candidate_correlations_impl(report_candidates)


def _sorted_report_candidates(
    report_candidates: Sequence[dict[str, Any]],
    *,
    scoring: _ResearchRunScoringConfig,
) -> list[dict[str, Any]]:
    from lumina_quant.strategy_factory.research_entrypoints import (
        _sorted_report_candidates as _sorted_report_candidates_impl,
    )

    return _sorted_report_candidates_impl(
        report_candidates=report_candidates,
        scoring=scoring,
    )


def _candidate_research_report_payload(
    *,
    base_tf: str,
    normalized_timeframes: Sequence[str],
    universe: Sequence[str],
    resolved_split: Mapping[str, Any],
    adapted: Sequence[dict[str, Any]],
    stage2_results: Sequence[dict[str, Any]],
    stage1_keep_ratio: float,
    scoring: _ResearchRunScoringConfig,
    data_sources: dict[str, list[str]],
    report_candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    from lumina_quant.strategy_factory.research_entrypoints import (
        _candidate_research_report_payload as _candidate_research_report_payload_impl,
    )

    return _candidate_research_report_payload_impl(
        base_tf=base_tf,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        adapted=adapted,
        stage2_results=stage2_results,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        data_sources=data_sources,
        report_candidates=report_candidates,
    )


def _run_candidate_research_with_adapted_candidates(
    *,
    base_tf: str,
    adapted: Sequence[dict[str, Any]],
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
    stage1_keep_ratio: float,
    scoring: _ResearchRunScoringConfig,
    split: Mapping[str, Any] | None,
    data_mode: str,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bundle_bars: int,
    market_data_settings: Mapping[str, Any],
) -> dict[str, Any]:
    from lumina_quant.strategy_factory.research_entrypoints import (
        _run_candidate_research_with_adapted_candidates as _run_candidate_research_with_adapted_candidates_impl,
    )

    return _run_candidate_research_with_adapted_candidates_impl(
        base_tf=base_tf,
        adapted=adapted,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        split=split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
        market_data_settings=market_data_settings,
    )


def run_candidate_research(
    *,
    candidates: Iterable[dict[str, Any]],
    base_timeframe: str = "1s",
    strategy_timeframes: Sequence[str] | None = None,
    symbol_universe: Sequence[str] | None = None,
    stage1_keep_ratio: float = 0.35,
    max_candidates: int = 512,
    score_config: Mapping[str, Any] | None = None,
    split: Mapping[str, Any] | None = None,
    data_mode: str = "legacy",
    allow_csv_fallback: bool = True,
    allow_synthetic_fallback: bool = True,
    min_bundle_bars: int = _MIN_BARS,
) -> dict[str, Any]:
    """Evaluate candidate manifest into train/val/OOS report contract (v2)."""
    from lumina_quant.strategy_factory.research_entrypoints import (
        run_candidate_research as _run_candidate_research_entrypoint,
    )

    return _run_candidate_research_entrypoint(
        candidates=candidates,
        base_timeframe=base_timeframe,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
        stage1_keep_ratio=stage1_keep_ratio,
        max_candidates=max_candidates,
        score_config=score_config,
        split=split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
    )


def build_default_candidate_rows(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    max_candidates: int = 512,
) -> list[dict[str, Any]]:
    """Build candidate rows from strategy-factory candidate library."""
    from lumina_quant.strategy_factory.research_entrypoints import (
        build_default_candidate_rows as _build_default_candidate_rows_entrypoint,
    )

    return _build_default_candidate_rows_entrypoint(
        symbols=symbols,
        timeframes=timeframes,
        max_candidates=max_candidates,
    )


__all__ = [
    "adapt_legacy_candidate",
    "build_default_candidate_rows",
    "run_candidate_research",
]
