#!/usr/bin/env python3
"""Tune fresh-start multi-sleeve portfolios from replay survivors/candidates.

This is intentionally downstream of replay_profit_moonshot_fresh_start.py.  It
selects sleeve candidates using train/validation evidence only, combines their
stateful replay equity curves, and reports OOS as report-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.portfolio.optimizer_core import safe_float as _safe_float  # noqa: E402

from lumina_quant.portfolio_split_contract import (  # noqa: E402
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/h1_h2_calendar_allocator"
)
DEFAULT_CANDIDATE_CSV = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion"
    / "fresh_start_overhaul_replay_candidates.csv"
)
CURRENT_CHAMPION_TRAIN_RETURN = 0.035993
CURRENT_CHAMPION_VAL_RETURN = 0.026755
CURRENT_CHAMPION_OOS_RETURN = 0.012181
CURRENT_CHAMPION_OOS_MDD = 0.001662
CURRENT_CHAMPION_OOS_RETURN_RISK = CURRENT_CHAMPION_OOS_RETURN / CURRENT_CHAMPION_OOS_MDD
BASELINE_OOS_RETURN = CURRENT_CHAMPION_OOS_RETURN
SHADOW_OOS_MDD = 0.001778
MIN_STABLE_MONTHLY_RETURN = 0.02
MAX_ACCEPTABLE_OOS_MDD = 0.25
SUCCESS_SHARPE = 2.0
SUCCESS_SORTINO = 3.0
SUCCESS_SMART_SORTINO = 3.0
SUCCESS_CALMAR = 1.0
SUCCESS_TRAIN_SHARPE = 1.5
SUCCESS_TRAIN_SORTINO = 1.5
SUCCESS_TRAIN_CALMAR = 1.0
SUCCESS_VAL_SHARPE = 3.0
SUCCESS_VAL_SORTINO = 3.0
SUCCESS_VAL_CALMAR = 3.0
TARGET_BUDGET_TRAIN_RETURN = 0.05
TARGET_BUDGET_VAL_RETURN = 0.04
MIN_TARGET_BUDGET_LEVERAGE = 0.20
MAX_TRAIN_VAL_MONTHLY_BUDGET_LEVERAGE = 6.0
RUN_NAME = "profit_moonshot_fresh_portfolio_tuning"
LOCKBOX_POLICY = {
    "selection_label": "train_val_validation_only",
    "locked_oos_label": "locked_oos_report_only",
    "locked_oos_gate_label": "locked_oos_gate_only",
    "diagnostic_best_oos_label": "diagnostic_locked_oos_not_selection_authority",
    "diagnostic_not_promoted_label": "diagnostic_not_promoted",
    "improved_label": "train_val_ranked_locked_oos_gate_passed",
    "oos_is_report_only": True,
    "oos_is_gate_only": True,
    "current_champion_oos_return": CURRENT_CHAMPION_OOS_RETURN,
    "minimum_stable_monthly_return": MIN_STABLE_MONTHLY_RETURN,
    "maximum_acceptable_oos_mdd": MAX_ACCEPTABLE_OOS_MDD,
}


def _load_fresh_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_profit_moonshot_fresh_start", FRESH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {FRESH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validation_score(row: dict[str, str]) -> float:
    val_ret = _safe_float(row.get("val_total_return"))
    train_ret = _safe_float(row.get("train_total_return"))
    val_sharpe = _safe_float(row.get("val_sharpe"))
    val_mdd = _safe_float(row.get("val_max_drawdown"), 1.0)
    trips = max(1.0, _safe_float(row.get("val_round_trips"), 0.0))
    return val_ret * 100.0 + train_ret * 25.0 + val_sharpe * 0.15 - val_mdd * 50.0 + math.log1p(trips) * 0.01


def _row_filters(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("filters") or {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _calendar_neighborhood_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if str(row.get("family") or "").strip() != "calendar_rotation":
        return None
    filters = _row_filters(row)
    long_symbol = str(filters.get("calendar_long_symbol") or "dynamic_long").upper()
    short_symbol = str(filters.get("calendar_short_symbol") or "dynamic_short").upper()
    hold_bars = round(_safe_float(filters.get("hold_bars"), _safe_float(row.get("hold_bars"), 0.0)))
    hold_bucket = int(round(hold_bars / 24.0) * 24) if hold_bars > 0 else 0
    threshold = _safe_float(filters.get("threshold"), _safe_float(row.get("threshold"), 0.0))
    take_profit = _safe_float(filters.get("take_profit_pct"), _safe_float(row.get("take_profit_pct"), 0.0))
    threshold_bucket = round(threshold / 0.003) if threshold > 0.0 else 0
    take_profit_bucket = round(take_profit / 0.012) if take_profit > 0.0 else 0
    return (
        "calendar_rotation",
        long_symbol,
        short_symbol,
        hold_bucket,
        threshold_bucket,
        take_profit_bucket,
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(item) for item in values)
    if len(arr) == 1:
        return arr[0]
    pos = min(max(float(q), 0.0), 1.0) * float(len(arr) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return arr[lo]
    frac = pos - float(lo)
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def _calendar_neighborhood_score(group_rows: list[dict[str, str]]) -> float:
    train_returns = [_safe_float(row.get("train_total_return")) for row in group_rows]
    val_returns = [_safe_float(row.get("val_total_return")) for row in group_rows]
    val_mdds = [_safe_float(row.get("val_max_drawdown"), 1.0) for row in group_rows]
    trips = sum(max(0.0, _safe_float(row.get("val_round_trips"))) for row in group_rows)
    train_median = _quantile(train_returns, 0.50)
    val_median = _quantile(val_returns, 0.50)
    val_lq = _quantile(val_returns, 0.25)
    val_mdd_median = _quantile(val_mdds, 0.50)
    dispersion = float(np.std(np.asarray(val_returns, dtype=float))) if len(val_returns) > 1 else 0.0
    return (
        val_median * 100.0
        + train_median * 25.0
        + val_lq * 20.0
        - val_mdd_median * 50.0
        - dispersion * 75.0
        + math.log1p(max(1.0, trips)) * 0.01
    )


def _append_unique(
    selected: list[dict[str, str]],
    selected_names: set[str],
    rows: list[dict[str, str]],
    *,
    limit: int,
) -> None:
    for row in rows:
        if len(selected) >= limit:
            return
        name = str(row.get("name") or "")
        if name and name not in selected_names:
            selected.append(row)
            selected_names.add(name)


def _candidate_pool_with_metadata(
    rows: list[dict[str, str]],
    *,
    top_n: int,
    family_quota: int = 0,
    calendar_neighborhood_reps: int = 0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if _safe_float(row.get("train_total_return")) > 0.0
        and _safe_float(row.get("val_total_return")) > 0.0
        and _safe_float(row.get("val_round_trips")) >= 1.0
    ]
    eligible.sort(key=_validation_score, reverse=True)
    limit = max(0, int(top_n))
    metadata: dict[str, Any] = {
        "selection_basis": "train_val_only",
        "pool_order": "validation_score_descending_after_calendar_neighborhood_reps",
        "calendar_neighborhood_reps": max(0, int(calendar_neighborhood_reps)),
        "family_quota": int(family_quota),
        "uses_locked_oos_for_selection": False,
        "calendar_neighborhoods": [],
    }
    if limit == 0:
        return [], metadata

    selected: list[dict[str, str]] = []
    selected_names: set[str] = set()
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in eligible:
        key = _calendar_neighborhood_key(row)
        if key is not None:
            groups.setdefault(key, []).append(row)

    scored_groups: list[dict[str, Any]] = []
    representatives: list[dict[str, str]] = []
    for key, group_rows in groups.items():
        representative = max(group_rows, key=_validation_score)
        representatives.append(representative)
        scored_groups.append(
            {
                "key": list(key),
                "score": float(_calendar_neighborhood_score(group_rows)),
                "size": len(group_rows),
                "representative": representative.get("name"),
                "train_median_return": _quantile(
                    [_safe_float(row.get("train_total_return")) for row in group_rows], 0.50
                ),
                "val_median_return": _quantile(
                    [_safe_float(row.get("val_total_return")) for row in group_rows], 0.50
                ),
                "val_lower_quartile_return": _quantile(
                    [_safe_float(row.get("val_total_return")) for row in group_rows], 0.25
                ),
            }
        )
    scored_groups.sort(key=lambda item: (item["score"], item["size"]), reverse=True)
    metadata["calendar_neighborhoods"] = scored_groups[:25]

    if calendar_neighborhood_reps > 0:
        representatives_by_name = {str(row.get("name") or ""): row for row in representatives}
        ordered_reps = [
            representatives_by_name[str(item.get("representative") or "")]
            for item in scored_groups[: max(0, int(calendar_neighborhood_reps))]
            if str(item.get("representative") or "") in representatives_by_name
        ]
        _append_unique(selected, selected_names, ordered_reps, limit=limit)

    if family_quota > 0:
        families = sorted(
            {row.get("family") or "unknown" for row in eligible},
            key=lambda family: max(
                (_validation_score(row) for row in eligible if (row.get("family") or "unknown") == family),
                default=float("-inf"),
            ),
            reverse=True,
        )
        for family in families:
            family_rows = [row for row in eligible if (row.get("family") or "unknown") == family]
            _append_unique(selected, selected_names, family_rows[: max(0, int(family_quota))], limit=limit)
            if len(selected) >= limit:
                metadata["selected_names"] = [row.get("name") for row in selected]
                return selected, metadata

    _append_unique(selected, selected_names, eligible, limit=limit)
    metadata["selected_names"] = [row.get("name") for row in selected]
    return selected, metadata


def _candidate_pool(
    rows: list[dict[str, str]],
    *,
    top_n: int,
    family_quota: int = 0,
    calendar_neighborhood_reps: int = 0,
) -> list[dict[str, str]]:
    pool, _metadata = _candidate_pool_with_metadata(
        rows,
        top_n=top_n,
        family_quota=family_quota,
        calendar_neighborhood_reps=calendar_neighborhood_reps,
    )
    return pool


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _combine_equity(
    curves: list[list[float]],
    *,
    mode: str,
    weights: list[float] | None = None,
    leverage: float = 1.0,
) -> list[float]:
    if not curves:
        return []
    min_len = min(len(curve) for curve in curves)
    if min_len <= 0:
        return []
    returns = []
    for curve in curves:
        arr = np.asarray(curve[:min_len], dtype=float)
        returns.append(arr / 10_000.0 - 1.0)
    stacked = np.vstack(returns)
    if weights is None:
        if mode in {
            "additive_sleeves",
            "train_val_target_return_budget",
            "train_val_monthly_return_budget",
        }:
            weight_arr = np.ones(stacked.shape[0], dtype=float)
        elif mode == "equal_weight":
            weight_arr = np.full(stacked.shape[0], 1.0 / float(stacked.shape[0]), dtype=float)
        else:
            raise ValueError(f"weights required for combine mode: {mode}")
    else:
        weight_arr = np.asarray(weights, dtype=float)
    combined = float(leverage) * np.dot(weight_arr, stacked)
    return [float(10_000.0 * (1.0 + item)) for item in combined]


def _normalize_weights(raw_weights: list[float]) -> list[float]:
    clean = [max(0.0, _safe_float(item)) for item in raw_weights]
    total = sum(clean)
    if total <= 0.0:
        return [1.0 / float(len(clean)) for _ in clean] if clean else []
    return [float(item / total) for item in clean]


def _validation_return_risk_weights(
    *,
    combo_names: tuple[str, ...],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
) -> list[float]:
    raw: list[float] = []
    for name in combo_names:
        train_metrics = split_payloads[name]["train"].get("metrics") or {}
        val_metrics = split_payloads[name]["val"].get("metrics") or {}
        train_ret = max(0.0, _safe_float(train_metrics.get("total_return")))
        val_ret = max(0.0, _safe_float(val_metrics.get("total_return")))
        val_sharpe = max(0.0, _safe_float(val_metrics.get("sharpe")))
        val_mdd = max(1e-6, _safe_float(val_metrics.get("max_drawdown"), 1.0))
        raw.append((val_ret * 4.0 + train_ret + val_sharpe * 0.01) / val_mdd)
    return _normalize_weights(raw)


def _train_val_mdd_budget_weights(
    *,
    combo_names: tuple[str, ...],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
) -> list[float]:
    raw: list[float] = []
    for name in combo_names:
        train_metrics = split_payloads[name]["train"].get("metrics") or {}
        val_metrics = split_payloads[name]["val"].get("metrics") or {}
        train_ret = max(0.0, _safe_float(train_metrics.get("total_return")))
        val_ret = max(0.0, _safe_float(val_metrics.get("total_return")))
        val_sharpe = max(0.0, _safe_float(val_metrics.get("sharpe")))
        train_mdd = _safe_float(train_metrics.get("max_drawdown"), 1.0)
        val_mdd = _safe_float(val_metrics.get("max_drawdown"), 1.0)
        mdd_budget = max(1e-6, train_mdd, val_mdd)
        raw.append((val_ret * 4.0 + train_ret + val_sharpe * 0.01) / mdd_budget)
    return _normalize_weights(raw)


def _calendar_cluster_key(row: dict[str, Any] | None, name: str) -> tuple[Any, ...]:
    if not row or str(row.get("family") or "").strip() != "calendar_rotation":
        return ("non_calendar", name)
    filters = _row_filters(row)
    long_symbol = str(filters.get("calendar_long_symbol") or "dynamic_long").upper()
    short_symbol = str(filters.get("calendar_short_symbol") or "dynamic_short").upper()
    hold_bars = round(_safe_float(filters.get("hold_bars"), 0.0))
    take_profit = _safe_float(filters.get("take_profit_pct"), 0.0)
    take_profit_bucket = round(take_profit / 0.012) if take_profit > 0.0 else 0
    return ("calendar_rotation", long_symbol, short_symbol, hold_bars, take_profit_bucket)


def _curve_return_vector(curves: dict[str, list[float]]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for split_name in ("train", "val"):
        arr = np.asarray(curves.get(split_name) or [], dtype=float)
        if arr.size >= 2:
            vectors.append(np.diff(arr / 10_000.0))
    if not vectors:
        return np.asarray([], dtype=float)
    return np.concatenate(vectors)


def _curve_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    n = min(int(lhs.size), int(rhs.size))
    if n < 2:
        return 0.0
    left = lhs[-n:]
    right = rhs[-n:]
    if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    return corr if math.isfinite(corr) else 0.0


def _cluster_assignments(
    *,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    candidate_rows_by_name: dict[str, dict[str, Any]],
    correlation_threshold: float,
) -> dict[str, str]:
    clusters: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}
    for name in combo_names:
        key = _calendar_cluster_key(candidate_rows_by_name.get(name), name)
        vector = _curve_return_vector(split_curves.get(name, {}))
        assigned_id = ""
        for cluster in clusters:
            if cluster["key"] != key:
                continue
            if _curve_corr(vector, cluster["prototype"]) >= correlation_threshold:
                assigned_id = str(cluster["id"])
                break
        if not assigned_id:
            assigned_id = f"cluster_{len(clusters) + 1}"
            clusters.append({"id": assigned_id, "key": key, "prototype": vector})
        assignments[name] = assigned_id
    return assignments


def _cap_cluster_weights(
    raw_weights: list[float],
    *,
    cluster_ids: list[str],
    cluster_cap: float,
    sleeve_cap: float,
) -> list[float]:
    weights = _normalize_weights(raw_weights)
    if not weights:
        return []
    cluster_cap = max(0.01, min(1.0, float(cluster_cap)))
    sleeve_cap = max(0.01, min(1.0, float(sleeve_cap)))
    weights = [min(weight, sleeve_cap) for weight in weights]
    for _ in range(12):
        excess = 0.0
        for idx, weight in enumerate(list(weights)):
            if weight > sleeve_cap:
                excess += weight - sleeve_cap
                weights[idx] = sleeve_cap
        cluster_sums = {
            cluster_id: sum(weight for weight, cid in zip(weights, cluster_ids, strict=True) if cid == cluster_id)
            for cluster_id in set(cluster_ids)
        }
        for cluster_id, total in cluster_sums.items():
            if total <= cluster_cap:
                continue
            scale = cluster_cap / total
            for idx, cid in enumerate(cluster_ids):
                if cid == cluster_id:
                    old = weights[idx]
                    weights[idx] = old * scale
                    excess += old - weights[idx]
        if excess <= 1e-12:
            break
        cluster_sums = {
            cluster_id: sum(weight for weight, cid in zip(weights, cluster_ids, strict=True) if cid == cluster_id)
            for cluster_id in set(cluster_ids)
        }
        capacities = [
            max(0.0, min(sleeve_cap - weight, cluster_cap - cluster_sums[cid]))
            for weight, cid in zip(weights, cluster_ids, strict=True)
        ]
        total_capacity = sum(capacities)
        if total_capacity <= 1e-12:
            break
        for idx, capacity in enumerate(capacities):
            weights[idx] += excess * capacity / total_capacity
    total = sum(weights)
    if total > 0.0 and abs(total - 1.0) > 1e-9:
        missing = 1.0 - total
        if missing > 0.0:
            cluster_sums = {
                cluster_id: sum(
                    weight for weight, cid in zip(weights, cluster_ids, strict=True) if cid == cluster_id
                )
                for cluster_id in set(cluster_ids)
            }
            capacities = [
                max(0.0, min(sleeve_cap - weight, cluster_cap - cluster_sums[cid]))
                for weight, cid in zip(weights, cluster_ids, strict=True)
            ]
            total_capacity = sum(capacities)
            if total_capacity > 1e-12:
                for idx, capacity in enumerate(capacities):
                    weights[idx] += missing * capacity / total_capacity
        else:
            weights = [weight / total for weight in weights]
    return weights


def _cluster_capped_validation_weights(
    *,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
    candidate_rows_by_name: dict[str, dict[str, Any]],
    cluster_cap: float,
    sleeve_cap: float,
    correlation_threshold: float,
) -> tuple[list[float], dict[str, Any]]:
    raw_weights = _train_val_mdd_budget_weights(combo_names=combo_names, split_payloads=split_payloads)
    assignments = _cluster_assignments(
        combo_names=combo_names,
        split_curves=split_curves,
        candidate_rows_by_name=candidate_rows_by_name,
        correlation_threshold=correlation_threshold,
    )
    cluster_ids = [assignments[name] for name in combo_names]
    weights = _cap_cluster_weights(raw_weights, cluster_ids=cluster_ids, cluster_cap=cluster_cap, sleeve_cap=sleeve_cap)
    cluster_weights = {
        cluster_id: sum(weight for weight, cid in zip(weights, cluster_ids, strict=True) if cid == cluster_id)
        for cluster_id in sorted(set(cluster_ids))
    }
    diagnostics = {
        "selection_basis": "train_val_only_cluster_capped",
        "uses_locked_oos_for_selection": False,
        "cluster_cap": float(cluster_cap),
        "sleeve_cap": float(sleeve_cap),
        "correlation_threshold": float(correlation_threshold),
        "clusters": dict(assignments),
        "cluster_weights": cluster_weights,
        "raw_train_val_mdd_budget_weights": [float(item) for item in raw_weights],
    }
    return weights, diagnostics


def _train_val_target_return_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
) -> tuple[float, dict[str, Any]]:
    split_returns: dict[str, float] = {}
    for split_name in ("train", "val"):
        equity = _combine_equity(
            [split_curves[name][split_name] for name in combo_names],
            mode="additive_sleeves",
            weights=None,
            leverage=1.0,
        )
        metrics = fresh._metrics_from_equity_totals(
            equity,
            periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
        )
        split_returns[split_name] = _safe_float(metrics.get("total_return"))

    required_leverages: list[float] = []
    if split_returns["train"] > 0.0:
        required_leverages.append(TARGET_BUDGET_TRAIN_RETURN / split_returns["train"])
    if split_returns["val"] > 0.0:
        required_leverages.append(TARGET_BUDGET_VAL_RETURN / split_returns["val"])
    raw_leverage = max(required_leverages, default=1.0)
    if not math.isfinite(raw_leverage):
        raw_leverage = 1.0
    leverage = max(MIN_TARGET_BUDGET_LEVERAGE, min(1.0, float(raw_leverage)))
    return leverage, {
        "selection_basis": "train_val_target_return_budget",
        "uses_locked_oos_for_selection": False,
        "target_train_return": TARGET_BUDGET_TRAIN_RETURN,
        "target_val_return": TARGET_BUDGET_VAL_RETURN,
        "raw_train_return": split_returns["train"],
        "raw_val_return": split_returns["val"],
        "raw_required_leverage": float(raw_leverage),
        "min_leverage": MIN_TARGET_BUDGET_LEVERAGE,
    }


def _split_metrics_for_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_name: str,
    leverage: float,
) -> dict[str, Any]:
    equity = _combine_equity(
        [split_curves[name][split_name] for name in combo_names],
        mode="train_val_monthly_return_budget",
        leverage=leverage,
    )
    if not equity:
        return {}
    return fresh._metrics_from_equity_totals(
        equity,
        periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
    )


def _leverage_for_monthly_return(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_name: str,
    max_leverage: float,
) -> float:
    base_metrics = _split_metrics_for_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name=split_name,
        leverage=1.0,
    )
    if _monthlyized_return(base_metrics) >= MIN_STABLE_MONTHLY_RETURN:
        return 1.0
    raw_total_return = _safe_float(base_metrics.get("total_return"))
    raw_cagr = _safe_float(base_metrics.get("cagr"))
    if raw_total_return <= 0.0 or raw_cagr <= -1.0:
        return max_leverage
    target_cagr = (1.0 + MIN_STABLE_MONTHLY_RETURN) ** 12.0 - 1.0
    annualization = 1.0
    if raw_total_return > -1.0 and raw_cagr > -1.0:
        try:
            annualization = math.log1p(raw_cagr) / math.log1p(raw_total_return)
        except (ValueError, ZeroDivisionError):
            annualization = 1.0
    if not math.isfinite(annualization) or annualization <= 0.0:
        annualization = 1.0
    target_total_return = (1.0 + target_cagr) ** (1.0 / annualization) - 1.0
    required = target_total_return / raw_total_return
    if not math.isfinite(required):
        return max_leverage
    return max(1.0, min(max_leverage, float(required)))


def _max_train_val_mdd_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_name: str,
    max_leverage: float,
) -> float:
    base_metrics = _split_metrics_for_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name=split_name,
        leverage=1.0,
    )
    raw_mdd = _safe_float(base_metrics.get("max_drawdown"), 0.0)
    if raw_mdd <= 0.0:
        return max_leverage
    cap = MAX_ACCEPTABLE_OOS_MDD / raw_mdd
    if not math.isfinite(cap):
        return max_leverage
    return max(0.0, min(max_leverage, float(cap)))


def _train_val_monthly_return_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
) -> tuple[float, dict[str, Any]]:
    max_leverage = MAX_TRAIN_VAL_MONTHLY_BUDGET_LEVERAGE
    train_required = _leverage_for_monthly_return(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="train",
        max_leverage=max_leverage,
    )
    val_required = _leverage_for_monthly_return(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="val",
        max_leverage=max_leverage,
    )
    train_mdd_cap = _max_train_val_mdd_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="train",
        max_leverage=max_leverage,
    )
    val_mdd_cap = _max_train_val_mdd_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="val",
        max_leverage=max_leverage,
    )
    raw_required = max(train_required, val_required, 1.0)
    safe_cap = min(max_leverage, train_mdd_cap, val_mdd_cap)
    leverage = max(MIN_TARGET_BUDGET_LEVERAGE, min(raw_required, safe_cap))
    diagnostics = {
        "selection_basis": "train_val_monthly_return_budget",
        "uses_locked_oos_for_selection": False,
        "target_monthly_return": MIN_STABLE_MONTHLY_RETURN,
        "max_train_val_mdd_budget": MAX_ACCEPTABLE_OOS_MDD,
        "max_leverage": max_leverage,
        "train_required_leverage": float(train_required),
        "val_required_leverage": float(val_required),
        "train_mdd_cap_leverage": float(train_mdd_cap),
        "val_mdd_cap_leverage": float(val_mdd_cap),
        "raw_required_leverage": float(raw_required),
        "selected_leverage": float(leverage),
    }
    return leverage, diagnostics


def _mode_weights_and_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
    mode: str,
    candidate_rows_by_name: dict[str, dict[str, Any]] | None = None,
    cluster_cap: float = 0.50,
    sleeve_cap: float = 0.35,
    correlation_threshold: float = 0.85,
) -> tuple[list[float] | None, float, dict[str, Any]]:
    if mode in {"additive_sleeves", "equal_weight"}:
        return None, 1.0, {}
    if mode == "train_val_target_return_budget":
        leverage, diagnostics = _train_val_target_return_leverage(
            fresh=fresh,
            combo_names=combo_names,
            split_curves=split_curves,
        )
        return None, leverage, diagnostics
    if mode == "train_val_monthly_return_budget":
        leverage, diagnostics = _train_val_monthly_return_leverage(
            fresh=fresh,
            combo_names=combo_names,
            split_curves=split_curves,
        )
        return None, leverage, diagnostics
    weights = _validation_return_risk_weights(combo_names=combo_names, split_payloads=split_payloads)
    if mode == "validation_return_risk_weight":
        return weights, 1.0, {"selection_basis": "train_val_validation_return_risk"}
    if mode == "validation_drawdown_budget":
        validation_equity = _combine_equity(
            [split_curves[name]["val"] for name in combo_names],
            mode=mode,
            weights=weights,
            leverage=1.0,
        )
        validation_metrics = fresh._metrics_from_equity_totals(
            validation_equity,
            periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
        )
        val_mdd = _safe_float(validation_metrics.get("max_drawdown"), 1.0)
        target_val_mdd = SHADOW_OOS_MDD * 0.75
        leverage = 1.0 if val_mdd <= 0.0 else target_val_mdd / val_mdd
        return weights, max(0.20, min(6.0, float(leverage))), {
            "selection_basis": "train_val_validation_drawdown_budget",
            "target_val_mdd": target_val_mdd,
        }
    if mode == "cluster_capped_validation_weight":
        capped_weights, diagnostics = _cluster_capped_validation_weights(
            combo_names=combo_names,
            split_curves=split_curves,
            split_payloads=split_payloads,
            candidate_rows_by_name=candidate_rows_by_name or {},
            cluster_cap=cluster_cap,
            sleeve_cap=sleeve_cap,
            correlation_threshold=correlation_threshold,
        )
        return capped_weights, 1.0, diagnostics
    raise ValueError(f"unknown combine mode: {mode}")


def _return_risk_score(total_return: Any, max_drawdown: Any) -> float:
    return _safe_float(total_return) / max(1e-9, _safe_float(max_drawdown, 1.0))


def _monthlyized_return(metrics: dict[str, Any]) -> float:
    """Return geometric monthly return from the annualized split CAGR."""
    cagr = _safe_float(metrics.get("cagr"), -1.0)
    if cagr <= -1.0:
        return -1.0
    return float((1.0 + cagr) ** (1.0 / 12.0) - 1.0)


def _smart_sortino(metrics: dict[str, Any]) -> float:
    """Sortino penalized for missing the monthly return floor and for using MDD budget.

    This local "smart sortino" is deliberately stricter than raw Sortino: a low
    return / tiny-drawdown row cannot pass only because the denominator is small.
    """
    monthly_return = _monthlyized_return(metrics)
    return_floor_factor = max(0.0, min(1.0, monthly_return / MIN_STABLE_MONTHLY_RETURN))
    mdd = max(0.0, _safe_float(metrics.get("max_drawdown"), 1.0))
    drawdown_budget_factor = max(0.0, 1.0 - min(mdd, MAX_ACCEPTABLE_OOS_MDD) / MAX_ACCEPTABLE_OOS_MDD)
    return _safe_float(metrics.get("sortino")) * return_floor_factor * drawdown_budget_factor


def _improved_candidate_from_gates(gates: dict[str, bool]) -> bool:
    required = (
        "train_positive",
        "val_positive",
        "train_return_beats_current_champion",
        "val_return_beats_current_champion",
        "train_monthly_return_gte_2pct",
        "val_monthly_return_gte_2pct",
        "train_sharpe_high",
        "train_sortino_high",
        "train_calmar_high",
        "val_sharpe_high",
        "val_sortino_high",
        "val_calmar_high",
        "oos_monthly_return_gte_2pct",
        "oos_return_beats_current_champion",
        "oos_return_risk_beats_current_champion",
        "oos_mdd_within_25pct_budget",
        "oos_sharpe_high",
        "oos_sortino_high",
        "oos_smart_sortino_high",
        "oos_calmar_high",
        "oos_trades_not_starved",
    )
    return all(bool(gates.get(key)) for key in required)


def _with_promotion_labels(item: dict[str, Any]) -> dict[str, Any]:
    gates = {str(key): bool(value) for key, value in dict(item.get("gates") or {}).items()}
    failed_gates = [key for key, ok in gates.items() if not ok]
    improved = _improved_candidate_from_gates(gates)
    item["gates"] = gates
    item["failed_gates"] = failed_gates
    item["improved_candidate"] = improved
    item["success_candidate"] = improved
    item["diagnostic_not_promoted"] = not improved
    item["promotion_status"] = (
        "improved_success_candidate" if improved else LOCKBOX_POLICY["diagnostic_not_promoted_label"]
    )
    item["locked_oos_policy"] = {
        "selection_basis": "train_val_only",
        "locked_oos_label": LOCKBOX_POLICY["locked_oos_label"],
        "oos_is_report_only": True,
        "oos_is_gate_only": True,
        "uses_locked_oos_for_selection": False,
    }
    return item


def _combo_metrics(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
    mode: str,
    candidate_rows_by_name: dict[str, dict[str, Any]] | None = None,
    cluster_cap: float = 0.50,
    sleeve_cap: float = 0.35,
    correlation_threshold: float = 0.85,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": f"fresh_portfolio_{mode}_" + "__".join(combo_names),
        "mode": mode,
        "sleeves": list(combo_names),
        "sleeve_count": len(combo_names),
        "splits": {},
    }
    weights, leverage, allocator_diagnostics = _mode_weights_and_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_payloads=split_payloads,
        mode=mode,
        candidate_rows_by_name=candidate_rows_by_name,
        cluster_cap=cluster_cap,
        sleeve_cap=sleeve_cap,
        correlation_threshold=correlation_threshold,
    )
    out["weights"] = [float(item) for item in weights] if weights is not None else []
    out["leverage"] = float(leverage)
    if allocator_diagnostics:
        out["allocator_diagnostics"] = allocator_diagnostics
    for split_name in ("train", "val", "oos"):
        curves = [split_curves[name][split_name] for name in combo_names]
        equity = _combine_equity(curves, mode=mode, weights=weights, leverage=leverage)
        metrics = fresh._metrics_from_equity_totals(
            equity,
            periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
        )
        round_trips = sum(int(split_payloads[name][split_name].get("round_trips") or 0) for name in combo_names)
        fills = sum(int(split_payloads[name][split_name].get("fills") or 0) for name in combo_names)
        out["splits"][split_name] = {
            "metrics": metrics,
            "round_trips": round_trips,
            "fills": fills,
            "final_equity": float(equity[-1]) if equity else 10_000.0,
        }
    train = out["splits"]["train"]["metrics"]
    val = out["splits"]["val"]["metrics"]
    oos = out["splits"]["oos"]["metrics"]
    train_return = _safe_float(train.get("total_return"))
    val_return = _safe_float(val.get("total_return"))
    oos_return = _safe_float(oos.get("total_return"))
    oos_mdd = _safe_float(oos.get("max_drawdown"), 1.0)
    train_monthly_return = _monthlyized_return(train)
    val_monthly_return = _monthlyized_return(val)
    oos_monthly_return = _monthlyized_return(oos)
    oos_smart_sortino = _smart_sortino(oos)
    gates = {
        "train_positive": train_return > 0.0,
        "val_positive": val_return > 0.0,
        "train_return_beats_current_champion": train_return > CURRENT_CHAMPION_TRAIN_RETURN,
        "val_return_beats_current_champion": val_return > CURRENT_CHAMPION_VAL_RETURN,
        "train_monthly_return_gte_2pct": train_monthly_return >= MIN_STABLE_MONTHLY_RETURN,
        "val_monthly_return_gte_2pct": val_monthly_return >= MIN_STABLE_MONTHLY_RETURN,
        "train_sharpe_high": _safe_float(train.get("sharpe")) >= SUCCESS_TRAIN_SHARPE,
        "train_sortino_high": _safe_float(train.get("sortino")) >= SUCCESS_TRAIN_SORTINO,
        "train_calmar_high": _safe_float(train.get("calmar")) >= SUCCESS_TRAIN_CALMAR,
        "val_sharpe_high": _safe_float(val.get("sharpe")) >= SUCCESS_VAL_SHARPE,
        "val_sortino_high": _safe_float(val.get("sortino")) >= SUCCESS_VAL_SORTINO,
        "val_calmar_high": _safe_float(val.get("calmar")) >= SUCCESS_VAL_CALMAR,
        "oos_monthly_return_gte_2pct": oos_monthly_return >= MIN_STABLE_MONTHLY_RETURN,
        "oos_return_beats_current_champion": oos_return > CURRENT_CHAMPION_OOS_RETURN,
        "oos_return_risk_beats_current_champion": (
            _return_risk_score(oos_return, oos_mdd) > CURRENT_CHAMPION_OOS_RETURN_RISK
        ),
        "oos_mdd_within_25pct_budget": oos_mdd <= MAX_ACCEPTABLE_OOS_MDD,
        "oos_sharpe_high": _safe_float(oos.get("sharpe")) >= SUCCESS_SHARPE,
        "oos_sortino_high": _safe_float(oos.get("sortino")) >= SUCCESS_SORTINO,
        "oos_smart_sortino_high": oos_smart_sortino >= SUCCESS_SMART_SORTINO,
        "oos_calmar_high": _safe_float(oos.get("calmar")) >= SUCCESS_CALMAR,
        "oos_trades_not_starved": int(out["splits"]["oos"].get("round_trips") or 0) >= 5,
    }
    out["return_quality"] = {
        "train_monthlyized_return": train_monthly_return,
        "val_monthlyized_return": val_monthly_return,
        "oos_monthlyized_return": oos_monthly_return,
        "oos_smart_sortino": oos_smart_sortino,
        "minimum_stable_monthly_return": MIN_STABLE_MONTHLY_RETURN,
        "maximum_acceptable_oos_mdd": MAX_ACCEPTABLE_OOS_MDD,
        "minimum_oos_sharpe": SUCCESS_SHARPE,
        "minimum_oos_sortino": SUCCESS_SORTINO,
        "minimum_oos_smart_sortino": SUCCESS_SMART_SORTINO,
        "minimum_oos_calmar": SUCCESS_CALMAR,
        "minimum_train_sharpe": SUCCESS_TRAIN_SHARPE,
        "minimum_train_sortino": SUCCESS_TRAIN_SORTINO,
        "minimum_train_calmar": SUCCESS_TRAIN_CALMAR,
        "minimum_val_sharpe": SUCCESS_VAL_SHARPE,
        "minimum_val_sortino": SUCCESS_VAL_SORTINO,
        "minimum_val_calmar": SUCCESS_VAL_CALMAR,
    }
    out["validation_score"] = (
        _safe_float(val.get("total_return")) * 100.0
        + _safe_float(train.get("total_return")) * 25.0
        + _safe_float(val.get("sharpe")) * 0.15
        - _safe_float(val.get("max_drawdown"), 1.0) * 50.0
        + math.log1p(max(1, int(out["splits"]["val"].get("round_trips") or 0))) * 0.01
    )
    out["gates"] = gates
    return _with_promotion_labels(out)


def _flatten_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": item["name"],
        "mode": item["mode"],
        "sleeve_count": item["sleeve_count"],
        "sleeves": ",".join(item["sleeves"]),
        "weights": ",".join(f"{float(weight):.6f}" for weight in item.get("weights") or []),
        "leverage": _safe_float(item.get("leverage"), 1.0),
        "validation_score": item["validation_score"],
        "success_candidate": item["success_candidate"],
        "improved_candidate": item.get("improved_candidate", False),
        "diagnostic_not_promoted": item.get("diagnostic_not_promoted", False),
        "promotion_status": item.get("promotion_status", ""),
        "failed_gates": ",".join(item.get("failed_gates") or []),
    }
    if item.get("return_quality"):
        quality = dict(item["return_quality"])
        for key in (
            "train_monthlyized_return",
            "val_monthlyized_return",
            "oos_monthlyized_return",
            "oos_smart_sortino",
        ):
            row[key] = _safe_float(quality.get(key), 0.0)
    if item.get("allocator_diagnostics"):
        row["allocator_selection_basis"] = dict(item["allocator_diagnostics"]).get("selection_basis", "")
    for split_name, split in item["splits"].items():
        metrics = split["metrics"]
        for key in ("total_return", "cagr", "max_drawdown", "sharpe", "sortino", "calmar", "volatility"):
            row[f"{split_name}_{key}"] = _safe_float(metrics.get(key), 0.0)
        row[f"{split_name}_round_trips"] = int(split.get("round_trips") or 0)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):+.4%}"


def _fmt_float(value: Any) -> str:
    return f"{_safe_float(value):.6f}"


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    selected = payload.get("selected_by_validation") or {}
    best_success = payload.get("best_success_candidate") or {}
    best_oos = payload.get("diagnostic_best_oos") or {}
    quarantine = payload.get("diagnostic_quarantine") or []
    memory_policy = payload.get("memory_policy") or {}
    memory_summary = payload.get("memory_summary") or {}
    lockbox_policy = payload.get("lockbox_policy") or LOCKBOX_POLICY
    combo_limit = payload.get("combo_limit_policy") or {}
    lines = [
        "# Profit moonshot fresh portfolio tuning",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Policy",
        "",
        "- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.",
        "- Portfolio selection is validation-primary; locked-OOS is report-only / gate-only.",
        "- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.",
        f"- Selection label: `{lockbox_policy.get('selection_label')}`.",
        f"- Locked-OOS label: `{lockbox_policy.get('locked_oos_label')}`.",
        f"- Locked-OOS gate label: `{lockbox_policy.get('locked_oos_gate_label')}`.",
        f"- Diagnostic quarantine label: `{lockbox_policy.get('diagnostic_not_promoted_label')}`.",
        f"- Stable-return floor: train, validation, and locked-OOS monthlyized return `>={MIN_STABLE_MONTHLY_RETURN:.2%}`.",
        f"- MDD budget: locked-OOS max drawdown `≤{MAX_ACCEPTABLE_OOS_MDD:.2%}`.",
        f"- Quality floors: OOS Sharpe `≥{SUCCESS_SHARPE:.1f}`, Sortino `≥{SUCCESS_SORTINO:.1f}`, smart Sortino `≥{SUCCESS_SMART_SORTINO:.1f}`, Calmar `≥{SUCCESS_CALMAR:.1f}`.",
        f"- Incumbent improvement still requires current-champion return/risk improvement from OOS return `>{CURRENT_CHAMPION_OOS_RETURN:.4%}`.",
        "",
        "## Runtime guard",
        "",
        f"- Heavy-run lock: `{memory_policy.get('heavy_lock_path')}`",
        f"- Explicit memory budget: `{memory_policy.get('explicit_budget_bytes')}` bytes",
        f"- RSS summary: `{memory_summary.get('summary_path') or memory_summary.get('rss_log_path')}`",
        "",
        "## Summary",
        "",
        f"- Candidate sleeves considered: `{payload['candidate_sleeve_count']}`",
        f"- Portfolio specs evaluated: `{payload['portfolio_spec_count']}`",
        f"- Combo cap per size: `{combo_limit.get('max_combos_per_size', 'n/a')}`; skipped by size: `{combo_limit.get('limit_hits_by_size', {})}`",
        f"- Success candidates: `{payload['success_candidate_count']}`",
        f"- Peak RSS: `{payload['peak_rss_mib']:.3f} MiB`",
        "",
    ]
    if selected:
        split = selected.get("splits", {})
        lines.extend(
            [
                "## Selected by validation",
                "",
                f"- `{selected.get('name')}`",
                f"- sleeves: `{', '.join(selected.get('sleeves') or [])}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- locked OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                f"- monthlyized train/val/OOS: `{_fmt_pct((selected.get('return_quality') or {}).get('train_monthlyized_return'))}` / `{_fmt_pct((selected.get('return_quality') or {}).get('val_monthlyized_return'))}` / `{_fmt_pct((selected.get('return_quality') or {}).get('oos_monthlyized_return'))}`; smart Sortino `{_fmt_float((selected.get('return_quality') or {}).get('oos_smart_sortino'))}`",
                f"- promotion status: `{selected.get('promotion_status')}` / failed gates: `{','.join(selected.get('failed_gates') or [])}`",
                "",
            ]
        )
    if best_success:
        split = best_success.get("splits", {})
        lines.extend(
            [
                "## Best success candidate",
                "",
                "- Ranked by train/validation validation score after all locked-OOS gates pass.",
                f"- `{best_success.get('name')}`",
                f"- sleeves: `{', '.join(best_success.get('sleeves') or [])}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- locked OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                f"- monthlyized train/val/OOS: `{_fmt_pct((best_success.get('return_quality') or {}).get('train_monthlyized_return'))}` / `{_fmt_pct((best_success.get('return_quality') or {}).get('val_monthlyized_return'))}` / `{_fmt_pct((best_success.get('return_quality') or {}).get('oos_monthlyized_return'))}`; smart Sortino `{_fmt_float((best_success.get('return_quality') or {}).get('oos_smart_sortino'))}`",
                f"- promotion status: `{best_success.get('promotion_status')}` / failed gates: `{','.join(best_success.get('failed_gates') or [])}`",
                "",
            ]
        )
    if best_oos:
        split = best_oos.get("splits", {})
        lines.extend(
            [
                "## Diagnostic best OOS (not selection authority)",
                "",
                f"- `{best_oos.get('name')}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- locked OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                f"- monthlyized locked OOS: `{_fmt_pct((best_oos.get('return_quality') or {}).get('oos_monthlyized_return'))}`; smart Sortino `{_fmt_float((best_oos.get('return_quality') or {}).get('oos_smart_sortino'))}`",
                f"- promotion status: `{best_oos.get('promotion_status')}`",
                "",
            ]
        )
    if quarantine:
        lines.extend(
            [
                "## H6 diagnostic quarantine",
                "",
                "- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.",
                "- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.",
                "",
                "| rank | name | mode | locked OOS | locked OOS MDD | failed gates |",
                "|---:|---|---|---:|---:|---|",
            ]
        )
        for idx, item in enumerate(quarantine[:10], start=1):
            split = item.get("splits", {})
            oos = split.get("oos", {}).get("metrics", {})
            lines.append(
                f"| {idx} | `{item.get('name')}` | `{item.get('mode')}` | "
                f"{_fmt_pct(oos.get('total_return'))} | {_fmt_pct(oos.get('max_drawdown'))} | "
                f"`{','.join(item.get('failed_gates') or [])}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Top rows",
            "",
            "| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(rows[:15], start=1):
        lines.append(
            f"| {idx} | `{row['name']}` | `{row['mode']}` | {_fmt_float(row.get('leverage'))} | {row['success_candidate']} | "
            f"{_fmt_pct(row['train_total_return'])} | {_fmt_pct(row['val_total_return'])} | "
            f"{_fmt_pct(row['oos_total_return'])} | {_fmt_pct(row.get('oos_monthlyized_return'))} | "
            f"{_fmt_pct(row['oos_max_drawdown'])} | {_fmt_float(row['oos_sharpe'])} | "
            f"{_fmt_float(row.get('oos_smart_sortino'))} | `{row['failed_gates']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _portfolio_report_sort_key(item: dict[str, Any]) -> tuple[bool, float, bool, str]:
    return (
        not bool(item["success_candidate"]),
        -_safe_float(item.get("validation_score")),
        item.get("promotion_status") == LOCKBOX_POLICY["diagnostic_not_promoted_label"],
        str(item.get("name") or ""),
    )


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fresh = _load_fresh_module()
    rows = _read_rows(Path(args.candidate_csv))
    pool, pool_policy = _candidate_pool_with_metadata(
        rows,
        top_n=int(args.top_n),
        family_quota=int(args.family_quota),
        calendar_neighborhood_reps=int(args.calendar_neighborhood_reps),
    )
    oos_end = datetime.fromisoformat(str(args.oos_end_date)).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root), exchange=str(args.exchange), symbols=symbols, start=start, end=end
    )
    arrays = fresh._build_arrays(panel, symbols)
    specs_by_name = {spec.name: spec for spec in fresh._candidate_specs(arrays, symbols)}
    pool = [row for row in pool if row["name"] in specs_by_name]
    candidate_rows_by_name = {row["name"]: dict(row) for row in pool}

    split_curves: dict[str, dict[str, list[float]]] = {}
    split_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for row in pool:
        name = row["name"]
        split_curves[name] = {}
        split_payloads[name] = {}
        for split in splits:
            result = fresh._run_split(
                spec=specs_by_name[name], arrays=arrays, split=split, include_equity=True
            )
            split_curves[name][split.name] = list(result.get("equity_history") or [])
            split_payloads[name][split.name] = result

    portfolio_items: list[dict[str, Any]] = []
    names = [row["name"] for row in pool]
    max_k = max(2, min(int(args.max_sleeves), len(names))) if names else 0
    max_combos_per_size = max(1, int(args.max_combos_per_size))
    combo_limit_hits: dict[str, int] = {}
    for size in range(2, max_k + 1):
        total_for_size = math.comb(len(names), size)
        if total_for_size > max_combos_per_size:
            combo_limit_hits[str(size)] = int(total_for_size - max_combos_per_size)
        for combo in itertools.islice(itertools.combinations(names, size), max_combos_per_size):
            for mode in (
                "equal_weight",
                "additive_sleeves",
                "train_val_target_return_budget",
                "train_val_monthly_return_budget",
                "validation_return_risk_weight",
                "validation_drawdown_budget",
                "cluster_capped_validation_weight",
            ):
                portfolio_items.append(
                    _combo_metrics(
                        fresh=fresh,
                        combo_names=combo,
                        split_curves=split_curves,
                        split_payloads=split_payloads,
                        mode=mode,
                        candidate_rows_by_name=candidate_rows_by_name,
                        cluster_cap=float(args.cluster_cap),
                        sleeve_cap=float(args.sleeve_cap),
                        correlation_threshold=float(args.correlation_threshold),
                    )
                )
    portfolio_items.sort(key=_portfolio_report_sort_key)
    csv_rows = [_flatten_row(item) for item in portfolio_items]
    selected_by_validation = max(portfolio_items, key=lambda item: item["validation_score"], default={})
    success_candidates = [item for item in portfolio_items if bool(item["success_candidate"])]
    best_success_candidate = max(
        success_candidates,
        key=lambda item: item["validation_score"],
        default={},
    )
    diagnostic_best_oos = max(
        portfolio_items,
        key=lambda item: (
            _safe_float(item["splits"]["oos"]["metrics"].get("total_return")),
            _safe_float(item["splits"]["oos"]["metrics"].get("sharpe")),
        ),
        default={},
    )
    diagnostic_quarantine = sorted(
        [
            item
            for item in portfolio_items
            if bool(item.get("diagnostic_not_promoted"))
            and _safe_float(item["splits"]["oos"]["metrics"].get("total_return")) > CURRENT_CHAMPION_OOS_RETURN
        ],
        key=lambda item: (
            _safe_float(item["splits"]["oos"]["metrics"].get("total_return")),
            -_safe_float(item["splits"]["oos"]["metrics"].get("max_drawdown"), 1.0),
        ),
        reverse=True,
    )
    payload = {
        "artifact_kind": "profit_moonshot_fresh_portfolio_tuning",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "candidate_csv": str(args.candidate_csv),
        "candidate_sleeve_count": len(pool),
        "portfolio_spec_count": len(portfolio_items),
        "combo_limit_policy": {
            "max_combos_per_size": max_combos_per_size,
            "limit_hits_by_size": combo_limit_hits,
            "pool_order": pool_policy.get("pool_order"),
            "family_quota": int(args.family_quota),
        },
        "candidate_pool_policy": pool_policy,
        "allocator_policy": {
            "modes": [
                "equal_weight",
                "additive_sleeves",
                "train_val_target_return_budget",
                "train_val_monthly_return_budget",
                "validation_return_risk_weight",
                "validation_drawdown_budget",
                "cluster_capped_validation_weight",
            ],
            "cluster_cap": float(args.cluster_cap),
            "sleeve_cap": float(args.sleeve_cap),
            "correlation_threshold": float(args.correlation_threshold),
            "selection_basis": "train_val_only",
            "uses_locked_oos_for_selection": False,
        },
        "success_candidate_count": len(success_candidates),
        "best_success_candidate": best_success_candidate,
        "selected_by_validation": selected_by_validation,
        "diagnostic_best_oos": diagnostic_best_oos,
        "diagnostic_quarantine": diagnostic_quarantine[:50],
        "data_metadata": data_metadata,
        "peak_rss_mib": _rss_mib(),
        "lockbox_policy": dict(LOCKBOX_POLICY),
        "memory_policy": memory_policy_payload(budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES),
    }
    return payload, csv_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT")
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=18)
    parser.add_argument("--family-quota", type=int, default=0)
    parser.add_argument("--calendar-neighborhood-reps", type=int, default=8)
    parser.add_argument("--cluster-cap", type=float, default=0.50)
    parser.add_argument("--sleeve-cap", type=float, default=0.35)
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--max-sleeves", type=int, default=5)
    parser.add_argument("--max-combos-per-size", type=int, default=12_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fresh_portfolio_tuning_latest.json"
    csv_path = output_dir / "fresh_portfolio_tuning_candidates.csv"
    md_path = output_dir / "fresh_portfolio_tuning_latest.md"
    memory_guard = acquire_portfolio_memory_guard(
        run_name=RUN_NAME,
        output_dir=output_dir,
        input_path=args.candidate_csv,
        metadata={
            "script": Path(__file__).name,
            "top_n": int(args.top_n),
            "family_quota": int(args.family_quota),
            "calendar_neighborhood_reps": int(args.calendar_neighborhood_reps),
            "cluster_cap": float(args.cluster_cap),
            "sleeve_cap": float(args.sleeve_cap),
            "correlation_threshold": float(args.correlation_threshold),
            "max_sleeves": int(args.max_sleeves),
            "locked_oos_label": LOCKBOX_POLICY["locked_oos_label"],
            "locked_oos_gate_label": LOCKBOX_POLICY["locked_oos_gate_label"],
            "selection_label": LOCKBOX_POLICY["selection_label"],
        },
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    finalized = False
    try:
        memory_guard.checkpoint(
            "start",
            {
                "candidate_csv": str(args.candidate_csv),
                "top_n": int(args.top_n),
                "family_quota": int(args.family_quota),
                "calendar_neighborhood_reps": int(args.calendar_neighborhood_reps),
                "max_sleeves": int(args.max_sleeves),
            },
        )
        payload, rows = build_payload(args)
        payload["lockbox_policy"] = dict(LOCKBOX_POLICY)
        payload["memory_policy"] = memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        )
        payload["rss_log_path"] = str(memory_guard.rss_log_path)
        payload["memory_summary_path"] = str(memory_guard.summary_path)
        _write_csv(csv_path, rows)
        memory_guard.checkpoint(
            "artifacts_prepared",
            {
                "portfolio_spec_count": int(payload["portfolio_spec_count"]),
                "success_candidate_count": int(payload["success_candidate_count"]),
            },
        )
        memory_summary = memory_guard.finalize(
            status="completed",
            context={
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "csv_path": str(csv_path),
                "portfolio_spec_count": int(payload["portfolio_spec_count"]),
                "success_candidate_count": int(payload["success_candidate_count"]),
            },
        )
        finalized = True
        memory_summary["summary_path"] = str(memory_guard.summary_path)
        payload["memory_summary"] = memory_summary
        payload["peak_rss_mib"] = max(
            _safe_float(payload.get("peak_rss_mib")),
            _safe_float(memory_summary.get("peak_rss_bytes")) / (1024.0 * 1024.0),
        )
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_markdown(payload, rows) + "\n", encoding="utf-8")
    except Exception as exc:
        if not finalized:
            memory_guard.finalize(status="failed", error=str(exc), context={"script": Path(__file__).name})
        raise
    finally:
        memory_guard.release()
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
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
