"""Shared portfolio optimizer helpers.

The script-level optimizers keep ownership of CLI/reporting behavior.  This
module owns the reusable, allocation-hot-path pieces that must stay consistent
across portfolio tuning, Optuna, and validator-facing reports.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np

LOCKED_OOS_OBJECTIVE_POLICY = "train_val_only_locked_oos_report"
DIAGNOSTIC_OOS_OBJECTIVE_POLICY = "diagnostic_oos_in_objective_not_selection_authority"
LOCKED_OOS_LABEL = "locked_oos_report_only"

AggregatedStream = dict[str, dict[str, Any]]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def canonical_split(value: Any, *, default: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return default
    alias_map = {
        "train": "train",
        "validation": "val",
        "val": "val",
        "oos": "oos",
        "test": "oos",
        "test_oos": "oos",
    }
    return alias_map.get(token, default)


def coerce_stream_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    numeric_value: float | None = None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
    elif isinstance(value, str) and value.strip():
        token = value.strip()
        try:
            numeric_value = float(token)
        except Exception:
            normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)

    if numeric_value is None or not math.isfinite(numeric_value):
        return None

    magnitude = abs(numeric_value)
    if magnitude >= 1e15:
        return datetime.fromtimestamp(numeric_value / 1_000_000.0, tz=UTC)
    if magnitude >= 1e12:
        return datetime.fromtimestamp(numeric_value / 1_000.0, tz=UTC)
    if magnitude >= 1e9:
        return datetime.fromtimestamp(numeric_value, tz=UTC)
    return None


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def stream_point_timestamp(raw_point: dict[str, Any]) -> Any:
    return raw_point.get("t", raw_point.get("timestamp", raw_point.get("datetime")))


def normalize_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw_point in enumerate(stream):
        value = safe_float(raw_point.get("v"), 0.0)
        raw_timestamp = stream_point_timestamp(raw_point)
        dt = coerce_stream_datetime(raw_timestamp)
        if dt is not None:
            timestamp = isoformat_utc(dt)
            normalized.append(
                {
                    "token": f"dt:{timestamp}",
                    "sort_key": (0, float(dt.timestamp()), float(idx)),
                    "t": timestamp,
                    "datetime": timestamp,
                    "v": float(value),
                }
            )
            continue

        numeric_timestamp = None
        if isinstance(raw_timestamp, (int, float)):
            numeric_timestamp = float(raw_timestamp)
        elif isinstance(raw_timestamp, str) and raw_timestamp.strip():
            try:
                numeric_timestamp = float(raw_timestamp.strip())
            except Exception:
                numeric_timestamp = None

        if numeric_timestamp is not None and math.isfinite(numeric_timestamp):
            normalized.append(
                {
                    "token": f"num:{numeric_timestamp:.12g}",
                    "sort_key": (1, float(numeric_timestamp), float(idx)),
                    "t": float(numeric_timestamp),
                    "datetime": None,
                    "v": float(value),
                }
            )
            continue

        seq = float(idx)
        normalized.append(
            {
                "token": f"seq:{seq:.12g}",
                "sort_key": (2, float(seq), float(idx)),
                "t": float(seq),
                "datetime": None,
                "v": float(value),
            }
        )
    normalized.sort(key=lambda item: item["sort_key"])
    return normalized


def aggregate_stream(stream: list[dict[str, Any]]) -> AggregatedStream:
    aggregated: AggregatedStream = {}
    for point in normalize_stream(stream):
        token = str(point["token"])
        if token not in aggregated:
            aggregated[token] = {
                "token": token,
                "sort_key": point["sort_key"],
                "t": point["t"],
                "datetime": point.get("datetime"),
                "v": 0.0,
            }
        aggregated[token]["v"] += float(point["v"])
    return aggregated


def stream_to_array(stream: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([safe_float(item.get("v"), 0.0) for item in stream], dtype=float)


def split_metrics(row: dict[str, Any], split: str) -> dict[str, Any]:
    token = canonical_split(split, default="oos")
    if token == "val":
        return dict(row.get("val") or row.get("validation") or {})
    if token == "train":
        return dict(row.get("train") or {})
    return dict(row.get("oos") or {})


def split_stream(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    token = canonical_split(split, default="oos")
    streams = row.get("return_streams") or {}
    raw_key = str(split or "").strip()
    aliases = {
        "train": ("train",),
        "val": ("val", "validation"),
        "oos": ("oos", "test", "test_oos"),
    }
    for key in (*aliases.get(token, (token,)), raw_key):
        if key and streams.get(key):
            return list(streams.get(key) or [])
    return []


def _as_aggregated(stream: list[dict[str, Any]] | AggregatedStream) -> AggregatedStream:
    if isinstance(stream, dict):
        return stream
    return aggregate_stream(stream)


def aligned_stream_arrays(
    lhs_stream: list[dict[str, Any]] | AggregatedStream,
    rhs_stream: list[dict[str, Any]] | AggregatedStream,
) -> tuple[np.ndarray, np.ndarray]:
    lhs = _as_aggregated(lhs_stream)
    rhs = _as_aggregated(rhs_stream)
    if not lhs and not rhs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    merged = dict(lhs)
    for token, point in rhs.items():
        if token not in merged:
            merged[token] = point

    ordered_tokens = [
        token for token, _ in sorted(merged.items(), key=lambda item: item[1]["sort_key"])
    ]
    lhs_values = np.asarray(
        [safe_float((lhs.get(token) or {}).get("v"), 0.0) for token in ordered_tokens],
        dtype=float,
    )
    rhs_values = np.asarray(
        [safe_float((rhs.get(token) or {}).get("v"), 0.0) for token in ordered_tokens],
        dtype=float,
    )
    return lhs_values, rhs_values


def corr_arrays(x: np.ndarray, y: np.ndarray) -> float:
    n = min(x.size, y.size)
    if n < 8:
        return 0.0
    xa = x[-n:]
    ya = y[-n:]
    sx = float(np.std(xa, ddof=1))
    sy = float(np.std(ya, ddof=1))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    value = float(np.corrcoef(xa, ya)[0, 1])
    if not math.isfinite(value):
        return 0.0
    return value


def corr_streams(
    lhs_stream: list[dict[str, Any]] | AggregatedStream,
    rhs_stream: list[dict[str, Any]] | AggregatedStream,
) -> float:
    lhs, rhs = aligned_stream_arrays(lhs_stream, rhs_stream)
    return corr_arrays(lhs, rhs)


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    dd = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    return float(np.max(dd)) if dd.size else 0.0


def metrics(returns: np.ndarray, periods_per_year: int = 365) -> dict[str, float]:
    if returns.size == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
        }
    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = max(1.0 / periods_per_year, returns.size / periods_per_year)
    cagr = float(math.exp(math.log1p(max(-0.999999, total_return)) / years) - 1.0)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = 0.0 if sigma <= 1e-12 else (mu / sigma) * math.sqrt(periods_per_year)

    downside = returns[returns < 0.0]
    dsigma = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = 0.0 if dsigma <= 1e-12 else (mu / dsigma) * math.sqrt(periods_per_year)

    mdd = max_drawdown(returns)
    calmar = 0.0 if mdd <= 1e-12 else cagr / mdd

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "volatility": float(sigma * math.sqrt(periods_per_year)),
    }


class StreamCache:
    """Cache normalized stream payloads for repeated portfolio combinations."""

    def __init__(self) -> None:
        self._aggregates: dict[tuple[str, str], AggregatedStream] = {}
        self._arrays: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _row_key(row: dict[str, Any]) -> str:
        key = row.get("candidate_id") or row.get("name")
        if key is not None and str(key).strip():
            return str(key)
        return f"row:{id(row)}"

    def aggregate_for_row(self, row: dict[str, Any], split: str) -> AggregatedStream:
        token = canonical_split(split, default="oos")
        key = (self._row_key(row), token)
        cached = self._aggregates.get(key)
        if cached is None:
            cached = aggregate_stream(split_stream(row, split))
            self._aggregates[key] = cached
        return cached

    def array_for_row(self, row: dict[str, Any], split: str) -> np.ndarray:
        token = canonical_split(split, default="oos")
        key = (self._row_key(row), token)
        cached = self._arrays.get(key)
        if cached is None:
            aggregate = self.aggregate_for_row(row, token)
            ordered = sorted(aggregate.values(), key=lambda item: item["sort_key"])
            cached = np.asarray([safe_float(point.get("v"), 0.0) for point in ordered], dtype=float)
            self._arrays[key] = cached
        return cached


def build_portfolio_stream(
    weights: dict[str, float],
    rows: dict[str, dict[str, Any]],
    *,
    split: str,
    cache: StreamCache | None = None,
) -> list[dict[str, Any]]:
    stream_cache = cache or StreamCache()
    aggregated: AggregatedStream = {}
    for cid, weight in weights.items():
        if weight <= 0.0:
            continue
        row = rows.get(cid) or {}
        for token, point in stream_cache.aggregate_for_row(row, split).items():
            if token not in aggregated:
                aggregated[token] = {
                    "token": token,
                    "sort_key": point["sort_key"],
                    "t": point["t"],
                    "datetime": point.get("datetime"),
                    "v": 0.0,
                }
            aggregated[token]["v"] += float(weight) * safe_float(point.get("v"), 0.0)

    out: list[dict[str, Any]] = []
    for point in sorted(aggregated.values(), key=lambda item: item["sort_key"]):
        row = {"t": point["t"], "v": float(point["v"])}
        if point.get("datetime"):
            row["datetime"] = point["datetime"]
        out.append(row)
    return out


def build_portfolio_returns(
    weights: dict[str, float],
    rows: dict[str, dict[str, Any]],
    *,
    split: str,
    cache: StreamCache | None = None,
) -> np.ndarray:
    return stream_to_array(build_portfolio_stream(weights, rows, split=split, cache=cache))


def cluster_by_correlation(
    ids: list[str],
    stream_map: dict[str, list[dict[str, Any]] | AggregatedStream],
    threshold: float = 0.60,
) -> list[list[str]]:
    clusters: list[list[str]] = []
    for cid in ids:
        added = False
        for cluster in clusters:
            if any(abs(corr_streams(stream_map[cid], stream_map[member])) >= threshold for member in cluster):
                cluster.append(cid)
                added = True
                break
        if not added:
            clusters.append([cid])
    return clusters


def objective_policy_payload(profile: str, *, oos_is_objective_input: bool) -> dict[str, Any]:
    policy = DIAGNOSTIC_OOS_OBJECTIVE_POLICY if oos_is_objective_input else LOCKED_OOS_OBJECTIVE_POLICY
    return {
        "objective_profile": str(profile),
        "objective_policy": policy,
        "selection_label": "diagnostic_oos_in_objective"
        if oos_is_objective_input
        else "train_val_validation_only",
        "locked_oos_label": LOCKED_OOS_LABEL,
        "oos_is_objective_input": bool(oos_is_objective_input),
    }
