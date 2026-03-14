"""Build a leakage-safe exact-window sleeve freeze artifact for portfolio tuning."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    OOS_START,
    REPORT_ROOT,
    TRAIN_END_EXCLUSIVE,
    VAL_END_EXCLUSIVE,
    VAL_START,
    memory_policy_payload,
    split_windows,
)

ONE_SHOT_BASELINE_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "portfolio_optimization_latest.json"
)
EQUAL_WEIGHT_BASELINE_PATH = FOLLOWUP_ROOT / "committee_portfolio_followup_latest.json"
PAIR_REVIEW_NOT_BEFORE = "2026-03-31"

DEFAULT_SLEEVE_SOURCES: dict[str, dict[str, Any]] = {
    "composite_trend_30m": {
        "path": REPORT_ROOT
        / "expansion_crypto_15m_30m_1h_20260310T115853Z"
        / "15m-30m-1h"
        / "exact_window_candidate_details_20260310T120626Z.json",
        "strategy_class": "CompositeTrendStrategy",
        "timeframe": "30m",
    },
    "topcap_tsmom_1h": {
        "path": REPORT_ROOT
        / "topcap_crypto_1h_focus_20260310T123813Z"
        / "1h"
        / "exact_window_candidate_details_20260310T124047Z.json",
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "timeframe": "1h",
    },
    "regime_breakout_1h": {
        "path": REPORT_ROOT
        / "topcap_crypto_1h_focus_20260310T123813Z"
        / "1h"
        / "exact_window_candidate_details_20260310T124047Z.json",
        "strategy_class": "RegimeBreakoutCandidateStrategy",
        "timeframe": "1h",
    },
    "rolling_breakout_30m": {
        "path": REPORT_ROOT
        / "expansion_crypto_15m_30m_1h_20260310T115853Z"
        / "15m-30m-1h"
        / "exact_window_candidate_details_20260310T120626Z.json",
        "strategy_class": "RollingBreakoutStrategy",
        "timeframe": "30m",
    },
}

ROLLING_GATE_PATH = FOLLOWUP_ROOT / "rolling_breakout_30m_gate_latest.json"
PAIR_RETUNE_PATH = FOLLOWUP_ROOT / "pair_spread_4h_xpt_xpd_retune_latest.json"

DENYLIST_FIELDS = {
    "committee",
    "candidate_pool_eligible",
    "promoted",
    "hurdle_fields",
    "btc_beating_candidate",
    "oos_btc_hurdle_pass",
    "oos_monthly_hurdle",
    "recent_three_month_two_pct_pass",
    "recent_three_months",
    "three_month_two_pct_candidate",
    "validation_btc_hurdle_pass",
    "validation_hurdle_pass",
    "validation_monthly_hurdle",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str) and value.strip():
        token = value.strip()
        try:
            numeric = float(token)
        except ValueError:
            normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
    else:
        return None

    magnitude = abs(numeric)
    if magnitude >= 1e15:
        return datetime.fromtimestamp(numeric / 1_000_000.0, tz=UTC)
    if magnitude >= 1e12:
        return datetime.fromtimestamp(numeric / 1_000.0, tz=UTC)
    if magnitude >= 1e9:
        return datetime.fromtimestamp(numeric, tz=UTC)
    return None


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _normalize_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, raw_point in enumerate(list(stream or [])):
        raw_ts = raw_point.get("datetime", raw_point.get("t", raw_point.get("timestamp")))
        dt = _canonical_timestamp(raw_ts)
        if dt is not None:
            ts = _isoformat_utc(dt)
            normalized.append(
                {
                    "token": f"dt:{ts}",
                    "sort_key": (0, float(dt.timestamp()), float(idx)),
                    "t": ts,
                    "datetime": ts,
                    "v": _safe_float(raw_point.get("v"), 0.0),
                }
            )
            continue
        normalized.append(
            {
                "token": f"seq:{idx}",
                "sort_key": (1, float(idx), float(idx)),
                "t": float(idx),
                "datetime": None,
                "v": _safe_float(raw_point.get("v"), 0.0),
            }
        )
    normalized.sort(key=lambda item: item["sort_key"])
    return normalized


def _aggregate_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for point in _normalize_stream(stream):
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

    rows: list[dict[str, Any]] = []
    for point in sorted(aggregated.values(), key=lambda item: item["sort_key"]):
        row = {"t": point["t"], "v": float(point["v"])}
        if point.get("datetime"):
            row["datetime"] = point["datetime"]
        rows.append(row)
    return rows


def _weighted_stream(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in rows:
        weight = _safe_float(row.get("_portfolio_weight"), 0.0)
        if weight <= 0.0:
            continue
        for point in _normalize_stream(list(((row.get("return_streams") or {}).get(split)) or [])):
            token = str(point["token"])
            if token not in aggregated:
                aggregated[token] = {
                    "token": token,
                    "sort_key": point["sort_key"],
                    "t": point["t"],
                    "datetime": point.get("datetime"),
                    "v": 0.0,
                }
            aggregated[token]["v"] += weight * _safe_float(point.get("v"), 0.0)

    out: list[dict[str, Any]] = []
    for point in sorted(aggregated.values(), key=lambda item: item["sort_key"]):
        row = {"t": point["t"], "v": float(point["v"])}
        if point.get("datetime"):
            row["datetime"] = point["datetime"]
        out.append(row)
    return out


def _split_window_summary(stream: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize_stream(stream)
    timestamps = [point.get("datetime") for point in normalized if point.get("datetime")]
    return {
        "point_count": len(normalized),
        "start": timestamps[0] if timestamps else None,
        "end": timestamps[-1] if timestamps else None,
    }


def _strict_reslice_streams(
    streams: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    train_end = _canonical_timestamp(TRAIN_END_EXCLUSIVE)
    val_start = _canonical_timestamp(VAL_START)
    val_end = _canonical_timestamp(VAL_END_EXCLUSIVE)
    oos_start = _canonical_timestamp(OOS_START)
    if train_end is None or val_start is None or val_end is None or oos_start is None:
        return (
            {
                split: _aggregate_stream(list(streams.get(split) or []))
                for split in ("train", "val", "oos")
            },
            [],
        )

    bucket: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "oos": [],
    }
    adjustments: list[str] = []
    for original_split in ("train", "val", "oos"):
        for point in _aggregate_stream(list(streams.get(original_split) or [])):
            dt = _canonical_timestamp(point.get("datetime", point.get("t")))
            if dt is None:
                continue
            if dt < train_end:
                target_split = "train"
            elif val_start <= dt < val_end:
                target_split = "val"
            elif dt >= oos_start:
                target_split = "oos"
            else:
                continue
            if target_split != original_split:
                adjustments.append(
                    f"moved {original_split} point at {point.get('datetime')} into {target_split} to enforce strict windows"
                )
            bucket[target_split].append(point)
    return {split: _aggregate_stream(points) for split, points in bucket.items()}, adjustments


def _canonical_row_id(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("name") or "")


def _freeze_score(row: dict[str, Any]) -> float:
    val = dict(row.get("val") or {})
    return float(
        (3.0 * _safe_float(val.get("sharpe"), 0.0))
        + (2.0 * _safe_float(val.get("deflated_sharpe"), 0.0))
        + (25.0 * _safe_float(val.get("return"), 0.0))
        - (2.5 * _safe_float(val.get("pbo"), 1.0))
        - (0.25 * _safe_float(val.get("turnover"), 0.0))
    )


def _secondary_score(row: dict[str, Any]) -> float:
    train = dict(row.get("train") or {})
    return float(
        _safe_float(train.get("deflated_sharpe"), 0.0)
        + (0.5 * _safe_float(train.get("sharpe"), 0.0))
    )


def _ranking_key(
    row: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, str]:
    val = dict(row.get("val") or {})
    return (
        -_freeze_score(row),
        -_secondary_score(row),
        _safe_float(val.get("pbo"), 1.0),
        -_safe_float(val.get("deflated_sharpe"), 0.0),
        -_safe_float(val.get("sharpe"), 0.0),
        -_safe_float(val.get("return"), 0.0),
        _safe_float(val.get("turnover"), 0.0),
        _canonical_row_id(row),
    )


def _trim_selected_row(row: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "candidate_id",
        "name",
        "family",
        "strategy_class",
        "strategy_timeframe",
        "timeframe",
        "symbols",
        "params",
        "pass",
        "metadata",
        "train",
        "val",
        "oos",
        "return_streams",
        "source_details_path",
        "source_summary_path",
    }
    trimmed = {key: value for key, value in row.items() if key in allowed}
    trimmed.pop("committee", None)
    trimmed.pop("candidate_pool_eligible", None)
    trimmed.pop("promoted", None)
    trimmed.pop("hurdle_fields", None)
    trimmed["pass"] = bool(trimmed.get("pass", True))
    if "timeframe" not in trimmed and "strategy_timeframe" in trimmed:
        trimmed["timeframe"] = trimmed["strategy_timeframe"]
    trimmed["selection_basis"] = "validation_only"
    trimmed["freeze_score"] = _freeze_score(row)
    trimmed["freeze_secondary_score"] = _secondary_score(row)
    trimmed["selection_inputs"] = {
        "candidate_id": row.get("candidate_id"),
        "name": row.get("name"),
        "train": {
            "sharpe": _safe_float(dict(row.get("train") or {}).get("sharpe"), 0.0),
            "deflated_sharpe": _safe_float(
                dict(row.get("train") or {}).get("deflated_sharpe"), 0.0
            ),
            "return": _safe_float(dict(row.get("train") or {}).get("return"), 0.0),
        },
        "val": {
            "sharpe": _safe_float(dict(row.get("val") or {}).get("sharpe"), 0.0),
            "deflated_sharpe": _safe_float(dict(row.get("val") or {}).get("deflated_sharpe"), 0.0),
            "return": _safe_float(dict(row.get("val") or {}).get("return"), 0.0),
            "pbo": _safe_float(dict(row.get("val") or {}).get("pbo"), 1.0),
            "turnover": _safe_float(dict(row.get("val") or {}).get("turnover"), 0.0),
        },
    }
    trimmed["present_denied_fields_in_source"] = sorted(
        field for field in DENYLIST_FIELDS if field in row
    )
    return trimmed


def _load_default_grouped_rows(
    report_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, str] = {}
    for sleeve, source in DEFAULT_SLEEVE_SOURCES.items():
        path = Path(source["path"])
        payload = _load_json(path)
        grouped[sleeve] = [
            dict(row)
            for row in list(payload or [])
            if str(row.get("strategy_class") or "") == str(source["strategy_class"])
            and str(row.get("strategy_timeframe") or row.get("timeframe") or "")
            == str(source["timeframe"])
        ]
        manifest[sleeve] = str(path.resolve())
    return grouped, manifest


def _coerce_grouped_rows(
    *,
    report_root: Path,
    grouped_rows: dict[str, list[dict[str, Any]]] | None,
    grouped_candidate_rows: dict[str, list[dict[str, Any]]] | None,
    candidate_groups: dict[str, list[dict[str, Any]]] | None,
    rows_by_sleeve: dict[str, list[dict[str, Any]]] | None,
    sleeve_rows: dict[str, list[dict[str, Any]]] | None,
    sleeve_groups: dict[str, list[dict[str, Any]]] | None,
    source_payloads: dict[str, list[dict[str, Any]]] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    for candidate in (
        grouped_rows,
        grouped_candidate_rows,
        candidate_groups,
        rows_by_sleeve,
        sleeve_rows,
        sleeve_groups,
        source_payloads,
    ):
        if isinstance(candidate, dict) and candidate:
            return {
                str(key): [dict(row) for row in list(value or [])]
                for key, value in candidate.items()
            }, {}
    return _load_default_grouped_rows(report_root)


def _coerce_pair_payload(
    *,
    pair_payload: dict[str, Any] | None,
    pair_spread_payload: dict[str, Any] | None,
    pairspread_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    for payload in (pair_payload, pair_spread_payload, pairspread_payload):
        if isinstance(payload, dict) and payload:
            return dict(payload)
    if PAIR_RETUNE_PATH.exists():
        return dict(_load_json(PAIR_RETUNE_PATH))
    return {}


def _coerce_rolling_gate_payload(
    *,
    rolling_gate_payload: dict[str, Any] | None,
    rolling_breakout_gate_payload: dict[str, Any] | None,
    gate_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    for payload in (rolling_gate_payload, rolling_breakout_gate_payload, gate_payload):
        if isinstance(payload, dict) and payload:
            return dict(payload)
    if ROLLING_GATE_PATH.exists():
        return dict(_load_json(ROLLING_GATE_PATH))
    return {}


def _ranking_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted([dict(row) for row in rows], key=_ranking_key)
    table: list[dict[str, Any]] = []
    for row in ranked:
        val = dict(row.get("val") or {})
        table.append(
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "freeze_score": _freeze_score(row),
                "secondary_score": _secondary_score(row),
                "val_sharpe": _safe_float(val.get("sharpe"), 0.0),
                "val_deflated_sharpe": _safe_float(val.get("deflated_sharpe"), 0.0),
                "val_return": _safe_float(val.get("return"), 0.0),
                "val_pbo": _safe_float(val.get("pbo"), 1.0),
                "val_turnover": _safe_float(val.get("turnover"), 0.0),
            }
        )
    return table


def _supplement_rolling_gate(
    selected_row: dict[str, Any], gate_payload: dict[str, Any]
) -> dict[str, Any]:
    if not gate_payload or not bool(gate_payload.get("survives")):
        return selected_row
    gated = dict(gate_payload.get("gated_candidate_row") or {})
    if not gated:
        return selected_row
    selected_id = _canonical_row_id(selected_row)
    gated_id = _canonical_row_id(gated)
    if selected_id and gated_id and selected_id != gated_id:
        return selected_row

    supplemented = dict(selected_row)
    split_adjustments: list[str] = []
    for split in ("train", "val", "oos"):
        if split in gated and isinstance(gated.get(split), dict):
            supplemented[split] = dict(gated.get(split) or {})
    if isinstance(gated.get("return_streams"), dict):
        gated_streams, split_adjustments = _strict_reslice_streams(
            dict(gated.get("return_streams") or {})
        )
        supplemented["return_streams"] = gated_streams
    metadata = dict(supplemented.get("metadata") or {})
    gated_metadata = dict(gated.get("metadata") or {})
    for key in (
        "activation_rule_id",
        "activation_rule_survives",
        "activation_rule_conditions",
        "activation_rule_label",
        "activation_signal_lag_days",
    ):
        if key in gated_metadata:
            metadata[key] = gated_metadata[key]
    if split_adjustments:
        metadata["activation_split_adjustments"] = list(split_adjustments)
    supplemented["metadata"] = metadata
    return supplemented


def _summary_from_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = dict(_load_json(path))
    return {
        "path": str(path.resolve()),
        "generated_at": payload.get("generated_at"),
        "portfolio_metrics": dict(payload.get("portfolio_metrics") or payload.get("metrics") or {}),
        "weight_count": len(list(payload.get("weights") or [])),
    }


def _equal_weight_baseline_support(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = dict(_load_json(path))
    selection_rows = [
        dict(row) for row in list(payload.get("selection") or []) if isinstance(row, dict)
    ]
    component_count = len(selection_rows)
    if component_count > 0:
        equal_weight = 1.0 / float(component_count)
        for row in selection_rows:
            row["_portfolio_weight"] = equal_weight
    combined_streams = (
        {split: _weighted_stream(selection_rows, split) for split in ("train", "val", "oos")}
        if selection_rows
        else {}
    )
    has_streams = any(
        bool(list((combined_streams.get(split)) or [])) for split in ("train", "val", "oos")
    )
    normalization_method = (
        "rebuilt_from_component_streams_default"
        if selection_rows and has_streams
        else "summary_only_missing_component_streams"
    )
    return {
        "path": str(path.resolve()),
        "generated_at": payload.get("generated_at"),
        "normalization_method": normalization_method,
        "component_count": component_count,
        "component_names": [row.get("name") for row in selection_rows],
        "portfolio_metrics": dict(payload.get("portfolio_metrics") or payload.get("metrics") or {}),
        "combined_streams": combined_streams if has_streams else {},
        "split_windows": {
            split: _split_window_summary(list((combined_streams.get(split)) or []))
            for split in ("train", "val", "oos")
        }
        if has_streams
        else {},
    }


def _one_shot_baseline_support(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = dict(_load_json(path))
    combined_streams = {
        split: _aggregate_stream(
            list(((payload.get("portfolio_return_streams") or {}).get(split)) or [])
        )
        for split in ("train", "val", "oos")
    }
    has_streams = any(combined_streams.values())
    return {
        "path": str(path.resolve()),
        "generated_at": payload.get("generated_at"),
        "normalization_method": "normalized_existing_portfolio_streams"
        if has_streams
        else "summary_only_missing_streams",
        "portfolio_metrics": dict(payload.get("portfolio_metrics") or payload.get("metrics") or {}),
        "combined_streams": combined_streams if has_streams else {},
        "split_windows": {
            split: _split_window_summary(list((combined_streams.get(split)) or []))
            for split in ("train", "val", "oos")
        }
        if has_streams
        else {},
        "weight_count": len(list(payload.get("weights") or [])),
    }


def _build_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    selected_rows = [
        dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)
    ]
    baseline_support = dict(payload.get("baseline_support") or {})
    pair_payload = dict(payload.get("pair_payload") or {})
    return {
        "generated_at": payload.get("generated_at"),
        "artifact_kind": "portfolio_exact_window_freeze_manifest",
        "schema_version": payload.get("schema_version"),
        "selection_basis": payload.get("selection_basis"),
        "split_windows": split_windows(),
        "memory_policy": memory_policy_payload(),
        "comparison_scope": list(payload.get("comparison_scope") or []),
        "source_manifest": dict(payload.get("source_manifest") or {}),
        "source_of_truth": dict((payload.get("selection") or {}).get("source_of_truth") or {}),
        "pairspread_exclusion": {
            "excluded": True,
            "review_not_before": pair_payload.get("review_not_before") or PAIR_REVIEW_NOT_BEFORE,
            "coverage_guard": dict(pair_payload.get("coverage_guard") or {}),
        },
        "baseline_support": baseline_support,
        "selected_rows": [
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "timeframe": row.get("timeframe") or row.get("strategy_timeframe"),
                "selection_basis": row.get("selection_basis"),
                "freeze_score": row.get("freeze_score"),
                "freeze_secondary_score": row.get("freeze_secondary_score"),
                "split_windows": {
                    split: _split_window_summary(
                        list(((row.get("return_streams") or {}).get(split)) or [])
                    )
                    for split in ("train", "val", "oos")
                },
            }
            for row in selected_rows
        ],
        "notes": list(payload.get("notes") or []),
    }


def build_portfolio_exact_window_freeze(
    *,
    report_root: Path | str = REPORT_ROOT,
    output_dir: Path | str | None = None,
    grouped_rows: dict[str, list[dict[str, Any]]] | None = None,
    grouped_candidate_rows: dict[str, list[dict[str, Any]]] | None = None,
    candidate_groups: dict[str, list[dict[str, Any]]] | None = None,
    rows_by_sleeve: dict[str, list[dict[str, Any]]] | None = None,
    sleeve_rows: dict[str, list[dict[str, Any]]] | None = None,
    sleeve_groups: dict[str, list[dict[str, Any]]] | None = None,
    source_payloads: dict[str, list[dict[str, Any]]] | None = None,
    component_rows: list[dict[str, Any]] | None = None,
    candidate_rows: list[dict[str, Any]] | None = None,
    rows: list[dict[str, Any]] | None = None,
    rolling_gate_payload: dict[str, Any] | None = None,
    rolling_breakout_gate_payload: dict[str, Any] | None = None,
    gate_payload: dict[str, Any] | None = None,
    pair_payload: dict[str, Any] | None = None,
    pair_spread_payload: dict[str, Any] | None = None,
    pairspread_payload: dict[str, Any] | None = None,
    include_one_shot_baseline: bool = True,
) -> dict[str, Any]:
    resolved_root = Path(report_root)
    _ = output_dir  # reserved for future use in callers/tests
    _ = component_rows
    _ = candidate_rows
    _ = rows

    grouped, source_manifest = _coerce_grouped_rows(
        report_root=resolved_root,
        grouped_rows=grouped_rows,
        grouped_candidate_rows=grouped_candidate_rows,
        candidate_groups=candidate_groups,
        rows_by_sleeve=rows_by_sleeve,
        sleeve_rows=sleeve_rows,
        sleeve_groups=sleeve_groups,
        source_payloads=source_payloads,
    )
    resolved_pair_payload = _coerce_pair_payload(
        pair_payload=pair_payload,
        pair_spread_payload=pair_spread_payload,
        pairspread_payload=pairspread_payload,
    )
    resolved_gate_payload = _coerce_rolling_gate_payload(
        rolling_gate_payload=rolling_gate_payload,
        rolling_breakout_gate_payload=rolling_breakout_gate_payload,
        gate_payload=gate_payload,
    )

    selected_rows: list[dict[str, Any]] = []
    ranking_tables: dict[str, list[dict[str, Any]]] = {}
    source_of_truth: dict[str, str] = {}
    excluded_sleeves: list[dict[str, Any]] = []

    for sleeve_name, sleeve_candidates in grouped.items():
        strategy_classes = {
            str(row.get("strategy_class") or "") for row in list(sleeve_candidates or [])
        }
        if "PairSpreadZScoreStrategy" in strategy_classes and not bool(
            resolved_pair_payload.get("survives")
        ):
            excluded_sleeves.append(
                {
                    "sleeve": sleeve_name,
                    "reason": "pair_spread_excluded_by_followup_guard",
                    "review_not_before": resolved_pair_payload.get("review_not_before"),
                }
            )
            continue
        if not sleeve_candidates:
            continue

        ranking_tables[sleeve_name] = _ranking_table(sleeve_candidates)
        chosen = dict(min(sleeve_candidates, key=_ranking_key))
        if str(chosen.get("strategy_class") or "") == "RollingBreakoutStrategy":
            source_of_truth[sleeve_name] = "exact_window_candidate_details_then_gate_supplement"
            chosen = _supplement_rolling_gate(chosen, resolved_gate_payload)
        else:
            source_of_truth[sleeve_name] = "exact_window_candidate_details"

        selected_rows.append(_trim_selected_row(chosen))

    selected_rows.sort(
        key=lambda row: (
            str(row.get("strategy_class") or ""),
            str(row.get("timeframe") or row.get("strategy_timeframe") or ""),
        )
    )

    equal_weight_summary = _summary_from_payload(EQUAL_WEIGHT_BASELINE_PATH)
    one_shot_summary = (
        _summary_from_payload(ONE_SHOT_BASELINE_PATH) if include_one_shot_baseline else {}
    )
    equal_weight_baseline_support = _equal_weight_baseline_support(EQUAL_WEIGHT_BASELINE_PATH)
    one_shot_baseline_support = (
        _one_shot_baseline_support(ONE_SHOT_BASELINE_PATH) if include_one_shot_baseline else {}
    )

    notes = [
        "Exact-window sleeve freeze uses the deterministic train/val-only formula; OOS-derived helper fields are excluded from selection.",
        "RollingBreakout is frozen from exact-window candidate-detail rows first, then optionally supplemented by the gate artifact for stream/metadata only.",
        "PairSpread 4h is excluded unless the follow-up guard explicitly survives.",
        "Equal-weight baseline default handling is rebuild/normalize from component streams before 3-way comparison.",
    ]
    if include_one_shot_baseline:
        notes.append(
            "One-shot optimized baseline defaults to reports/portfolio_optimization_latest.json."
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "portfolio_exact_window_freeze",
        "schema_version": "1.0",
        "selection_basis": "validation_only",
        "comparison_scope": [
            "equal_weight_diagnostic",
            "one_shot_optimized",
            "exact_window_frozen",
        ],
        "candidates": selected_rows,
        "selected_team": selected_rows,
        "frozen_rows": selected_rows,
        "optimizer_bundle": {
            "artifact_kind": "portfolio_exact_window_freeze_optimizer_bundle",
            "selection_basis": "validation_only",
            "candidates": selected_rows,
            "selected_team": selected_rows,
            "source_artifact_kind": "portfolio_exact_window_freeze",
        },
        "ranking_tables": ranking_tables,
        "excluded_sleeves": excluded_sleeves,
        "pair_survives": bool(resolved_pair_payload.get("survives")),
        "pair_payload": {
            "survives": bool(resolved_pair_payload.get("survives")),
            "coverage_guard": dict(resolved_pair_payload.get("coverage_guard") or {}),
            "review_not_before": resolved_pair_payload.get("review_not_before")
            or PAIR_REVIEW_NOT_BEFORE,
        },
        "selection": {
            "selection_basis": "validation_only",
            "formula": "(3*val.sharpe)+(2*val.deflated_sharpe)+(25*val.return)-(2.5*val.pbo)-(0.25*val.turnover)",
            "secondary_rule": "train.deflated_sharpe + 0.5*train.sharpe",
            "denylisted_fields": sorted(DENYLIST_FIELDS),
            "source_of_truth": source_of_truth,
            "count": len(selected_rows),
        },
        "split_windows": split_windows(),
        "memory_policy": memory_policy_payload(),
        "baseline_paths": {
            "equal_weight_diagnostic": str(EQUAL_WEIGHT_BASELINE_PATH.resolve()),
            "one_shot_optimized": str(ONE_SHOT_BASELINE_PATH.resolve()),
        },
        "baseline_summaries": {
            "equal_weight_diagnostic": equal_weight_summary,
            "one_shot_optimized": one_shot_summary,
        },
        "baseline_support": {
            "equal_weight_diagnostic": equal_weight_baseline_support,
            "one_shot_optimized": one_shot_baseline_support,
        },
        "source_manifest": source_manifest,
        "notes": notes,
    }
    return payload


def write_portfolio_exact_window_freeze(
    *,
    report_root: Path | str = REPORT_ROOT,
    output_dir: Path | str | None = None,
    run_name: str = "portfolio_exact_window_freeze",
    **kwargs: Any,
) -> dict[str, Any]:
    payload = build_portfolio_exact_window_freeze(
        report_root=report_root,
        output_dir=output_dir,
        **kwargs,
    )
    manifest_payload = _build_manifest_payload(payload)
    followup_root = Path(output_dir or report_root) / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    json_path = followup_root / f"{run_name}_latest.json"
    md_path = followup_root / f"{run_name}_latest.md"
    manifest_json_path = followup_root / f"{run_name}_manifest_latest.json"
    manifest_md_path = followup_root / f"{run_name}_manifest_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_json_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# portfolio exact-window freeze",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- frozen_count: `{len(list(payload.get('candidates') or []))}`",
        f"- pair_survives: `{payload.get('pair_survives')}`",
        f"- equal_weight_baseline: `{payload['baseline_paths']['equal_weight_diagnostic']}`",
        f"- one_shot_baseline: `{payload['baseline_paths']['one_shot_optimized']}`",
        "",
        "## frozen sleeves",
    ]
    for row in list(payload.get("candidates") or []):
        val = dict(row.get("val") or {})
        lines.append(
            f"- `{row.get('name')}` | strategy={row.get('strategy_class')} | tf={row.get('timeframe') or row.get('strategy_timeframe')} | "
            f"val_sharpe={_safe_float(val.get('sharpe'), 0.0):.3f} | "
            f"val_return={_safe_float(val.get('return'), 0.0):.4%} | "
            f"val_pbo={_safe_float(val.get('pbo'), 1.0):.3f}"
        )
    lines.extend(["", "## notes"])
    for note in list(payload.get("notes") or []):
        lines.append(f"- {note}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_lines = [
        "# portfolio exact-window freeze manifest",
        "",
        f"- generated_at: `{manifest_payload.get('generated_at')}`",
        f"- selection_basis: `{manifest_payload.get('selection_basis')}`",
        f"- oos_start: `{dict(manifest_payload.get('split_windows') or {}).get('oos_start')}`",
        f"- heavy_lock_path: `{dict(manifest_payload.get('memory_policy') or {}).get('heavy_lock_path')}`",
        "",
        "## sources",
    ]
    for sleeve, source in sorted(dict(manifest_payload.get("source_manifest") or {}).items()):
        manifest_lines.append(f"- {sleeve}: `{source}`")
    pair = dict(manifest_payload.get("pairspread_exclusion") or {})
    manifest_lines.extend(
        [
            "",
            "## baselines",
            f"- equal_weight: `{dict((manifest_payload.get('baseline_support') or {}).get('equal_weight_diagnostic') or {}).get('normalization_method')}`",
            f"- one_shot_optimized: `{dict((manifest_payload.get('baseline_support') or {}).get('one_shot_optimized') or {}).get('normalization_method')}`",
            "",
            "## exclusions",
            f"- PairSpread 4h excluded: `{pair.get('excluded')}` | review_not_before=`{pair.get('review_not_before')}`",
        ]
    )
    manifest_md_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
        "manifest_payload": manifest_payload,
        "manifest_json_path": str(manifest_json_path.resolve()),
        "manifest_md_path": str(manifest_md_path.resolve()),
    }


def main() -> int:
    result = write_portfolio_exact_window_freeze()
    print(result["json_path"])
    print(result["md_path"])
    print(result["manifest_json_path"])
    print(result["manifest_md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
