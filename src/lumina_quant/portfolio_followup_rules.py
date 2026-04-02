from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.eval.exact_window_suite import _metrics_daily

ROBUST_PROMOTION_GATES = {
    "train_total_return_min_exclusive": 0.0,
    "val_total_return_min_exclusive": 0.0,
    "train_sharpe_min": -0.10,
    "oos_total_return_delta_min_exclusive": 0.0,
    "oos_monthly_mean_min": 0.02,
    "oos_sharpe_relief_min": 0.50,
}

VALIDATION_OBJECTIVE_FORMULA = (
    "(1.0 * val_sharpe) + (0.35 * val_sortino) + (0.10 * val_calmar) + "
    "(10.0 * val_total_return) - (4.0 * val_max_drawdown) - (0.75 * val_volatility)"
)

MEMORY_LEDGER_ARTIFACT_KIND = "portfolio_meta_search_memory_ledger_row"
MEMORY_LEDGER_SCHEMA_VERSION = "1.0"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    for key in ("candidate_key", "candidate_id", "name", "label"):
        token = str(candidate.get(key) or "").strip()
        if token:
            return token
    return "candidate"


def _candidate_tokens(candidate: Mapping[str, Any]) -> str:
    metadata = dict(candidate.get("metadata") or {})
    tokens = [
        _candidate_identity(candidate),
        str(candidate.get("selection_basis") or ""),
        str(metadata.get("basis_family") or ""),
        str(metadata.get("basis_variant") or ""),
    ]
    lineage = dict(candidate.get("basis_lineage") or metadata.get("basis_lineage") or {})
    tokens.extend(
        [
            str(lineage.get("family") or ""),
            str(lineage.get("variant") or ""),
            str(lineage.get("basis_universe") or ""),
        ]
    )
    normalized = " ".join(token for token in tokens if token).lower()
    return normalized.replace("-", "_")


def infer_basis_lineage(candidate: Mapping[str, Any]) -> dict[str, str]:
    metadata = dict(candidate.get("metadata") or {})
    lineage = dict(candidate.get("basis_lineage") or metadata.get("basis_lineage") or {})

    family = str(lineage.get("family") or metadata.get("basis_family") or "").strip().lower()
    variant = str(lineage.get("variant") or metadata.get("basis_variant") or "").strip().lower()
    basis_universe = str(
        lineage.get("basis_universe") or metadata.get("basis_universe") or ""
    ).strip().lower()

    tokens = _candidate_tokens(candidate)
    if not family or not variant or not basis_universe:
        if "55_45" in tokens or "55/45" in tokens:
            family = family or "static_blend"
            variant = variant or "raw_55_45"
            basis_universe = basis_universe or "raw_basis"
        elif "80_20" in tokens or "80/20" in tokens:
            family = family or "static_blend"
            variant = variant or "derived_80_20"
            basis_universe = basis_universe or "derived_basis"
        elif "incumbent_autoresearch_static_blend" in tokens or "current_one_shot_incumbent" in tokens:
            family = family or "incumbent"
            variant = variant or "incumbent_autoresearch_static_blend"
            basis_universe = basis_universe or "shared"
        elif "soft_allocator" in tokens or "soft allocator" in tokens:
            family = family or "soft_allocator"
            variant = variant or "soft_allocator"
            basis_universe = basis_universe or "shared"
        elif "regime_switch" in tokens or "regime switch" in tokens:
            family = family or "regime_switch"
            variant = variant or "regime_switch"
            basis_universe = basis_universe or "shared"
        elif "grouped_base" in tokens or "grouped base" in tokens:
            family = family or "grouped_base"
            variant = variant or "grouped_base"
            basis_universe = basis_universe or "shared"

    if not family:
        family = _candidate_identity(candidate).strip().lower().replace("-", "_").replace(" ", "_")
    if not variant:
        variant = family
    if not basis_universe:
        basis_universe = "shared"

    return {
        "family": family,
        "variant": variant,
        "basis_universe": basis_universe,
    }


def annotate_basis_lineage(candidate: Mapping[str, Any]) -> dict[str, Any]:
    annotated = dict(candidate)
    metadata = dict(annotated.get("metadata") or {})
    lineage = infer_basis_lineage(annotated)
    annotated["basis_lineage"] = dict(lineage)
    metadata.setdefault("basis_family", lineage["family"])
    metadata.setdefault("basis_variant", lineage["variant"])
    metadata.setdefault("basis_universe", lineage["basis_universe"])
    annotated["metadata"] = metadata
    return annotated


def validate_basis_universe(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    members: list[dict[str, str]] = []
    for row in list(rows or []):
        lineage = infer_basis_lineage(row)
        candidate_key = _candidate_identity(row)
        members.append({"candidate_key": candidate_key, **lineage})
        family = lineage["family"]
        variant = lineage["variant"]
        seen = families.get(family)
        if seen is not None and seen != variant:
            conflicts.append(
                {
                    "family": family,
                    "existing_variant": seen,
                    "conflicting_variant": variant,
                    "candidate_key": candidate_key,
                }
            )
        else:
            families[family] = variant
    return {
        "ok": not conflicts,
        "conflicts": conflicts,
        "members": members,
    }


def build_basis_search_universes(rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    shared: list[dict[str, Any]] = []
    raw_basis: list[dict[str, Any]] = []
    derived_basis: list[dict[str, Any]] = []
    for row in list(rows or []):
        annotated = annotate_basis_lineage(row)
        universe = str((annotated.get("basis_lineage") or {}).get("basis_universe") or "shared")
        if universe == "raw_basis":
            raw_basis.append(annotated)
        elif universe == "derived_basis":
            derived_basis.append(annotated)
        else:
            shared.append(annotated)

    universes = {
        "raw_basis": [*shared, *raw_basis],
        "derived_basis": [*shared, *derived_basis],
    }
    deduped: dict[str, list[dict[str, Any]]] = {}
    for universe_name, universe_rows in universes.items():
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in universe_rows:
            key = _candidate_identity(row)
            if key in seen:
                continue
            ordered.append(dict(row))
            seen.add(key)
        deduped[universe_name] = ordered
    return deduped


def _point_datetime(point: Mapping[str, Any]) -> datetime | None:
    raw_datetime = point.get("datetime")
    if isinstance(raw_datetime, str) and raw_datetime.strip():
        normalized = raw_datetime.replace("Z", "+00:00") if raw_datetime.endswith("Z") else raw_datetime
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    raw_t = point.get("t")
    try:
        numeric = float(raw_t)
    except (TypeError, ValueError):
        return None
    scale = 1000.0 if abs(numeric) >= 1e12 else 1.0
    return datetime.fromtimestamp(numeric / scale, tz=UTC)


def compound_returns(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def _weighted_return_stream(
    rows: list[Mapping[str, Any]],
    split: str,
    *,
    weight_key: str = "_saved_weight",
) -> list[dict[str, Any]]:
    bucket: dict[datetime, float] = defaultdict(float)
    for row in list(rows or []):
        weight = safe_float(row.get(weight_key, row.get("weight")), 0.0)
        if weight <= 0.0:
            continue
        streams = extract_portfolio_streams(row)
        for point in list(streams.get(split) or []):
            dt = _point_datetime(point)
            if dt is None:
                continue
            bucket[dt] += weight * safe_float(point.get("v"), 0.0)
    return [
        {
            "datetime": dt.isoformat().replace("+00:00", "Z"),
            "t": float(int(dt.timestamp() * 1000.0)),
            "v": float(bucket[dt]),
        }
        for dt in sorted(bucket)
    ]


def _daily_return_stream(stream: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    daily: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        dt = _point_datetime(point)
        if dt is None:
            continue
        daily[dt.date().isoformat()].append(safe_float(point.get("v"), 0.0))
    rows: list[dict[str, Any]] = []
    for day_key in sorted(daily):
        day_dt = datetime.fromisoformat(f"{day_key}T00:00:00+00:00").astimezone(UTC)
        rows.append(
            {
                "datetime": day_dt.isoformat().replace("+00:00", "Z"),
                "t": float(int(day_dt.timestamp() * 1000.0)),
                "v": compound_returns(daily[day_key]),
            }
        )
    return rows


def monthly_returns(stream: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    monthly: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        dt = _point_datetime(point)
        if dt is None:
            continue
        monthly[dt.strftime("%Y-%m")].append(safe_float(point.get("v"), 0.0))
    rows: list[dict[str, Any]] = []
    for month in sorted(monthly):
        values = monthly[month]
        rows.append(
            {
                "month": month,
                "total_return": compound_returns(values),
                "days": len(values),
            }
        )
    return rows


def mean_monthly_return(rows: list[Mapping[str, Any]]) -> float:
    values = [
        safe_float(row.get("total_return", row.get("return")), 0.0)
        for row in list(rows or [])
        if isinstance(row, Mapping)
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def extract_split_metrics(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    for key in ("portfolio_metrics", "split_metrics", "metrics"):
        metric_block = payload.get(key)
        if not isinstance(metric_block, Mapping):
            continue
        for split in ("train", "val", "oos"):
            split_payload = metric_block.get(split)
            if isinstance(split_payload, Mapping):
                extracted[split] = dict(split_payload)
    for split in ("train", "val", "oos"):
        if split not in extracted and isinstance(payload.get(split), Mapping):
            extracted[split] = dict(payload.get(split) or {})
    return extracted


def extract_portfolio_streams(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    for key in ("portfolio_return_streams", "return_streams", "portfolio_daily_return_streams"):
        stream_block = payload.get(key)
        if not isinstance(stream_block, Mapping):
            continue
        return {
            split: [dict(item) for item in list(stream_block.get(split) or []) if isinstance(item, Mapping)]
            for split in ("train", "val", "oos")
        }
    return {split: [] for split in ("train", "val", "oos")}


def extract_oos_monthly_returns(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = payload.get("oos_monthly_returns")
    if isinstance(explicit, list):
        return [dict(item) for item in explicit if isinstance(item, Mapping)]
    streams = extract_portfolio_streams(payload)
    oos_stream = list(streams.get("oos") or [])
    if not oos_stream:
        return []
    return monthly_returns(oos_stream)


def evaluate_weighted_portfolio(
    rows: list[Mapping[str, Any]],
    *,
    weight_key: str = "_saved_weight",
) -> dict[str, Any]:
    raw_streams = {
        split: _weighted_return_stream(rows, split, weight_key=weight_key)
        for split in ("train", "val", "oos")
    }
    daily_streams = {split: _daily_return_stream(stream) for split, stream in raw_streams.items()}

    metrics: dict[str, dict[str, Any]] = {}
    weighted_component_summaries: dict[str, dict[str, float]] = {}
    for split, stream in daily_streams.items():
        returns = np.asarray([safe_float(item.get("v"), 0.0) for item in stream], dtype=float)
        split_metrics = dict(_metrics_daily(returns))
        split_metrics["return"] = float(split_metrics.get("total_return", 0.0))
        metrics[split] = split_metrics
        weighted_component_summaries[split] = {
            "trade_count": float(
                sum(
                    safe_float((row.get(split) or {}).get("trade_count"), 0.0)
                    * safe_float(row.get(weight_key, row.get("weight")), 0.0)
                    for row in rows
                )
            ),
            "turnover": float(
                sum(
                    safe_float((row.get(split) or {}).get("turnover"), 0.0)
                    * safe_float(row.get(weight_key, row.get("weight")), 0.0)
                    for row in rows
                )
            ),
            "benchmark_corr": float(
                sum(
                    safe_float((row.get(split) or {}).get("benchmark_corr"), 0.0)
                    * safe_float(row.get(weight_key, row.get("weight")), 0.0)
                    for row in rows
                )
            ),
        }

    return {
        "portfolio_metrics": metrics,
        "weighted_component_summaries": weighted_component_summaries,
        "portfolio_return_streams": raw_streams,
        "portfolio_daily_return_streams": daily_streams,
        "oos_monthly_returns": monthly_returns(daily_streams["oos"]),
    }


def validation_objective(metrics: Mapping[str, Any]) -> float:
    return float(
        (1.0 * safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * safe_float(metrics.get("sortino"), 0.0))
        + (0.10 * safe_float(metrics.get("calmar"), 0.0))
        + (10.0 * safe_float(metrics.get("total_return", metrics.get("return")), 0.0))
        - (4.0 * safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0))
        - (0.75 * safe_float(metrics.get("volatility"), 0.0))
    )


def evaluate_robustness_gates(
    candidate_payload: Mapping[str, Any],
    incumbent_payload: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_metrics = extract_split_metrics(candidate_payload)
    incumbent_metrics = extract_split_metrics(incumbent_payload)

    candidate_train = dict(candidate_metrics.get("train") or {})
    candidate_val = dict(candidate_metrics.get("val") or {})
    candidate_oos = dict(candidate_metrics.get("oos") or {})
    incumbent_oos = dict(incumbent_metrics.get("oos") or {})

    oos_total_return_delta = safe_float(
        candidate_oos.get("total_return", candidate_oos.get("return")), 0.0
    ) - safe_float(incumbent_oos.get("total_return", incumbent_oos.get("return")), 0.0)
    oos_max_drawdown_delta = safe_float(
        candidate_oos.get("max_drawdown", candidate_oos.get("mdd")), 0.0
    ) - safe_float(incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")), 0.0)
    oos_sharpe_delta = safe_float(candidate_oos.get("sharpe"), 0.0) - safe_float(
        incumbent_oos.get("sharpe"), 0.0
    )

    monthly_rows = extract_oos_monthly_returns(candidate_payload)
    oos_monthly_mean = mean_monthly_return(monthly_rows)

    checks = {
        "train_total_return": safe_float(
            candidate_train.get("total_return", candidate_train.get("return")), 0.0
        ) > ROBUST_PROMOTION_GATES["train_total_return_min_exclusive"],
        "val_total_return": safe_float(
            candidate_val.get("total_return", candidate_val.get("return")), 0.0
        ) > ROBUST_PROMOTION_GATES["val_total_return_min_exclusive"],
        "train_sharpe": safe_float(candidate_train.get("sharpe"), 0.0)
        >= ROBUST_PROMOTION_GATES["train_sharpe_min"],
        "oos_total_return_delta": oos_total_return_delta
        > ROBUST_PROMOTION_GATES["oos_total_return_delta_min_exclusive"],
        "oos_monthly_mean": oos_monthly_mean >= ROBUST_PROMOTION_GATES["oos_monthly_mean_min"],
        "drawdown_or_sharpe_relief": (
            safe_float(candidate_oos.get("max_drawdown", candidate_oos.get("mdd")), 0.0)
            <= safe_float(incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")), 0.0)
        )
        or (
            safe_float(candidate_oos.get("sharpe"), 0.0)
            >= safe_float(incumbent_oos.get("sharpe"), 0.0)
            + ROBUST_PROMOTION_GATES["oos_sharpe_relief_min"]
        ),
    }

    rejection_reasons: list[str] = []
    if not checks["train_total_return"]:
        rejection_reasons.append("train_total_return_non_positive")
    if not checks["val_total_return"]:
        rejection_reasons.append("val_total_return_non_positive")
    if not checks["train_sharpe"]:
        rejection_reasons.append("train_sharpe_below_floor")
    if not checks["oos_total_return_delta"]:
        rejection_reasons.append("oos_total_return_not_above_incumbent")
    if not checks["oos_monthly_mean"]:
        rejection_reasons.append("oos_monthly_mean_below_floor")
    if not checks["drawdown_or_sharpe_relief"]:
        rejection_reasons.append("oos_drawdown_worse_without_sharpe_relief")

    return {
        "promotable": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "robustness_gate_checks": checks,
        "oos_monthly_mean": oos_monthly_mean,
        "oos_monthly_month_count": len(monthly_rows),
        "oos_total_return_delta": oos_total_return_delta,
        "oos_max_drawdown_delta": oos_max_drawdown_delta,
        "oos_sharpe_delta": oos_sharpe_delta,
    }


def build_memory_ledger_row(
    *,
    run_name: str,
    basis_universe: str,
    candidate_count: int,
    combination_count: int,
    budget_bytes: int,
    heavy_lock_path: str | Path,
    session_memory_lease_path: str | Path,
    status: str,
    started_at: str,
    completed_at: str,
    memory_summary_path: str | Path,
) -> dict[str, Any]:
    return {
        "artifact_kind": MEMORY_LEDGER_ARTIFACT_KIND,
        "schema_version": MEMORY_LEDGER_SCHEMA_VERSION,
        "run_name": str(run_name),
        "basis_universe": str(basis_universe),
        "candidate_count": int(candidate_count),
        "combination_count": int(combination_count),
        "budget_bytes": int(budget_bytes),
        "one_heavy_lane_only": True,
        "heavy_lock_path": str(Path(heavy_lock_path)),
        "session_memory_lease_path": str(Path(session_memory_lease_path)),
        "status": str(status),
        "started_at": str(started_at),
        "completed_at": str(completed_at),
        "memory_summary_path": str(Path(memory_summary_path)),
    }


def serialize_memory_ledger_row(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True)


def parse_memory_ledger_row(raw: str) -> dict[str, Any]:
    payload = dict(json.loads(str(raw)))
    if payload.get("artifact_kind") != MEMORY_LEDGER_ARTIFACT_KIND:
        raise ValueError("unexpected memory ledger artifact kind")
    if str(payload.get("schema_version") or "") != MEMORY_LEDGER_SCHEMA_VERSION:
        raise ValueError("unexpected memory ledger schema version")
    if not bool(payload.get("one_heavy_lane_only")):
        raise ValueError("memory ledger row must declare one-heavy-lane policy")
    return payload


__all__ = [
    "MEMORY_LEDGER_ARTIFACT_KIND",
    "MEMORY_LEDGER_SCHEMA_VERSION",
    "ROBUST_PROMOTION_GATES",
    "VALIDATION_OBJECTIVE_FORMULA",
    "annotate_basis_lineage",
    "build_basis_search_universes",
    "build_memory_ledger_row",
    "evaluate_robustness_gates",
    "evaluate_weighted_portfolio",
    "extract_oos_monthly_returns",
    "extract_portfolio_streams",
    "extract_split_metrics",
    "infer_basis_lineage",
    "mean_monthly_return",
    "monthly_returns",
    "parse_memory_ledger_row",
    "safe_float",
    "serialize_memory_ledger_row",
    "validate_basis_universe",
    "validation_objective",
]
