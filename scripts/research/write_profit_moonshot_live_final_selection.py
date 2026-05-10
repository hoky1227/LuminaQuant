#!/usr/bin/env python3
"""Write the profit-moonshot live final-selection comparison artifact.

The writer is intentionally artifact-only: it normalizes existing refresh,
candidate-tuning, liquidation-aware replay, and optional legacy-hybrid benchmark
JSONs into one final decision surface. It does not run research/backtests.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

EIGHT_GIB_BYTES = 8 * 1024 * 1024 * 1024
CURRENT_BASE_LEVERAGE = 2.3427334297703024
CURRENT_BASE_SLEEVE_COUNT = 4
MAX_ACCEPTABLE_OOS_MDD = 0.25
MIN_OOS_SHARPE = 2.0
MIN_OOS_SORTINO = 3.0
MIN_OOS_SMART_SORTINO = 3.0
MIN_OOS_CALMAR = 1.0
DEFAULT_REQUIRED_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT"
DEFAULT_OUTPUT_DIR = Path("var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision")
_TIME_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

METRICS_EXPLANATION = {
    "total_return": "Split ending equity return before compounding; higher is better after gates.",
    "monthlyized_return": "Monthlyized return estimate used for train/validation stability context.",
    "annualized_return": "Annualized/CAGR-style return where available.",
    "max_drawdown": "Largest peak-to-trough equity loss in the split; lower is safer.",
    "return_mdd": "Total return divided by max drawdown; rewards return per drawdown unit.",
    "sharpe": "Return per total volatility; higher is better.",
    "sortino": "Return per downside volatility; higher is better for asymmetric downside risk.",
    "smart_sortino": "Sortino adjusted by strategy-specific tail/quality penalties when available.",
    "calmar": "Annualized return divided by max drawdown; live safety ratio.",
    "volatility": "Annualized or split-normalized total volatility from source artifact.",
    "downside_volatility": "Downside-only volatility where available.",
    "positive_period_ratio": "Fraction of positive periods or proxy win-rate where available.",
    "fills": "Executed fill count; helps detect starved or overactive sleeves.",
    "round_trips": "Completed trade count when available.",
    "liquidation_count": "Number of liquidation events in the split.",
    "minimum_margin_buffer": "Minimum equity minus conservative margin requirement; must stay positive.",
    "minimum_margin_ratio": "Minimum equity divided by margin requirement; higher is safer.",
    "maximum_liquidation_event_drawdown": "Worst instantaneous drawdown at liquidation event.",
    "maximum_liquidation_equity_loss_fraction": "Worst equity fraction lost at liquidation event.",
    "account_wipeout": "Whether an event wiped the whole account; any true value blocks promotion.",
    "memory_max_rss": "Maximum resident set size evidence; must stay below 8 GiB.",
}


@dataclass(frozen=True, slots=True)
class SourceArtifacts:
    refresh_json: str = ""
    candidate_portfolio_json: str = ""
    liquidation_json: str = ""
    legacy_hybrid_json: str = ""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path | str | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _safe_optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _canonical_symbol(value: Any) -> str:
    token = str(value or "").strip().upper().replace("-", "/")
    if not token:
        return ""
    if "/" in token:
        left, right = token.split("/", 1)
        return f"{left}{right}"
    return token


def _parse_required_symbols(value: str | Iterable[str] | None) -> list[str]:
    if isinstance(value, str):
        tokens = [item.strip() for item in value.split(",") if item.strip()]
    else:
        tokens = [str(item).strip() for item in list(value or []) if str(item).strip()]
    out: list[str] = []
    for token in tokens:
        canonical = _canonical_symbol(token)
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def _parse_utc(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _complete_date_from_timestamp(timestamp: datetime) -> date:
    threshold = timestamp.replace(hour=23, minute=59, second=0, microsecond=0)
    if timestamp >= threshold:
        return timestamp.date()
    return timestamp.date() - timedelta(days=1)


def derive_data_cutoff(
    refresh_payload: Mapping[str, Any],
    *,
    required_symbols: Iterable[str] | None = None,
) -> dict[str, Any]:
    required = _parse_required_symbols(required_symbols or DEFAULT_REQUIRED_SYMBOLS)
    rows_by_symbol: dict[str, Mapping[str, Any]] = {}
    for row in _as_list(refresh_payload.get("ohlcv_results")):
        if not isinstance(row, Mapping):
            continue
        symbol = _canonical_symbol(row.get("symbol"))
        if symbol:
            rows_by_symbol[symbol] = row

    timestamps: list[datetime] = []
    missing: list[str] = []
    source_rows: list[dict[str, Any]] = []
    for symbol in required:
        row = rows_by_symbol.get(symbol)
        if row is None:
            missing.append(symbol)
            continue
        ts = _parse_utc(row.get("after_ohlcv_max_utc"))
        if ts is None:
            missing.append(symbol)
            continue
        timestamps.append(ts)
        source_rows.append(
            {
                "symbol": symbol,
                "after_ohlcv_max_utc": ts.isoformat().replace("+00:00", "Z"),
            }
        )

    if not timestamps:
        fallback = _parse_utc(refresh_payload.get("collection_cutoff_utc")) or datetime.now(UTC)
        latest_date = _complete_date_from_timestamp(fallback)
        status = "fallback_collection_cutoff"
        min_ts = fallback
    else:
        min_ts = min(timestamps)
        latest_date = _complete_date_from_timestamp(min_ts)
        status = "derived_from_refresh" if not missing else "derived_with_missing_symbols"

    return {
        "latest_complete_oos_end_date": latest_date.isoformat(),
        "minimum_after_ohlcv_max_utc": min_ts.isoformat().replace("+00:00", "Z"),
        "required_symbols": required,
        "missing_symbols": missing,
        "source_rows": source_rows,
        "status": status,
    }


def train_val_stability_score_from_components(
    components: Mapping[str, Any],
    *,
    base_leverage: float = CURRENT_BASE_LEVERAGE,
    base_sleeve_count: int = CURRENT_BASE_SLEEVE_COUNT,
) -> float:
    """Frozen train/validation-only score; locked-OOS is intentionally absent."""
    return (
        35.0 * min(_safe_float(components.get("train_monthlyized_return")), 0.06)
        + 45.0 * min(_safe_float(components.get("validation_monthlyized_return")), 0.12)
        + 0.40 * _safe_float(components.get("train_sharpe"))
        + 0.60 * _safe_float(components.get("validation_sharpe"))
        + 0.35 * _safe_float(components.get("train_sortino"))
        + 0.55 * _safe_float(components.get("validation_sortino"))
        + 0.20 * min(_safe_float(components.get("train_calmar")), 20.0)
        + 0.30 * min(_safe_float(components.get("validation_calmar")), 60.0)
        - 35.0 * _safe_float(components.get("train_max_drawdown"))
        - 45.0 * _safe_float(components.get("validation_max_drawdown"))
        - 0.15 * max(0.0, _safe_float(components.get("leverage")) - float(base_leverage))
        - 0.25 * max(0.0, _safe_float(components.get("sleeve_count")) - float(base_sleeve_count))
    )


def _monthlyized_from_cagr(cagr: Any) -> float | None:
    parsed = _safe_optional_float(cagr)
    if parsed is None or parsed <= -1.0:
        return None
    return float((1.0 + parsed) ** (1.0 / 12.0) - 1.0)


def _return_mdd(total_return: float | None, max_drawdown: float | None) -> float | None:
    if total_return is None or max_drawdown is None or max_drawdown <= 0.0:
        return None
    return float(total_return / max_drawdown)


def _split_aliases(split_name: str) -> list[str]:
    if split_name == "validation":
        return ["validation", "val"]
    return [split_name]


def _raw_split(raw: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    splits = _as_dict(raw.get("splits"))
    for key in _split_aliases(split_name):
        if isinstance(splits.get(key), Mapping):
            return dict(splits[key])
    return {}


def _return_quality(raw: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(raw.get("return_quality"))


def _quality_key(split_name: str, metric: str) -> list[str]:
    if split_name == "train":
        return [f"train_{metric}"]
    if split_name == "validation":
        return [f"validation_{metric}", f"val_{metric}"]
    return [f"locked_oos_{metric}", f"oos_{metric}"]


def _metric_from_quality(raw: Mapping[str, Any], split_name: str, metric: str) -> float | None:
    quality = _return_quality(raw)
    for key in _quality_key(split_name, metric):
        value = _safe_optional_float(quality.get(key))
        if value is not None:
            return value
    return None


def _normalize_split(raw: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    source = _raw_split(raw, split_name)
    metrics = _as_dict(source.get("metrics"))
    total_return = _safe_optional_float(metrics.get("total_return", metrics.get("raw_total_return")))
    if total_return is None:
        total_return = _metric_from_quality(raw, split_name, "total_return")
    max_drawdown = _safe_optional_float(metrics.get("max_drawdown", metrics.get("raw_max_drawdown")))
    if max_drawdown is None:
        max_drawdown = _metric_from_quality(raw, split_name, "max_drawdown")
    monthlyized = _metric_from_quality(raw, split_name, "monthlyized_return")
    if monthlyized is None:
        monthlyized = _monthlyized_from_cagr(metrics.get("cagr"))
    sharpe = _safe_optional_float(metrics.get("sharpe"))
    if sharpe is None:
        sharpe = _metric_from_quality(raw, split_name, "sharpe")
    sortino = _safe_optional_float(metrics.get("sortino"))
    if sortino is None:
        sortino = _metric_from_quality(raw, split_name, "sortino")
    smart_sortino = _safe_optional_float(metrics.get("smart_sortino"))
    if smart_sortino is None:
        smart_sortino = _metric_from_quality(raw, split_name, "smart_sortino")
    calmar = _safe_optional_float(metrics.get("calmar"))
    if calmar is None:
        calmar = _metric_from_quality(raw, split_name, "calmar")
    return {
        "total_return": total_return,
        "monthlyized_return": monthlyized,
        "annualized_return": _safe_optional_float(metrics.get("cagr")),
        "max_drawdown": max_drawdown,
        "return_mdd": _return_mdd(total_return, max_drawdown),
        "sharpe": sharpe,
        "sortino": sortino,
        "smart_sortino": smart_sortino,
        "calmar": calmar,
        "volatility": _safe_optional_float(metrics.get("volatility")),
        "downside_volatility": _safe_optional_float(metrics.get("downside_volatility")),
        "positive_period_ratio": _safe_optional_float(
            metrics.get("positive_period_ratio", metrics.get("win_rate"))
        ),
        "fills": int(_safe_float(source.get("fills"), 0.0)),
        "round_trips": int(_safe_float(source.get("round_trips"), 0.0)),
        "liquidation_count": int(_safe_float(source.get("liquidation_count"), 0.0)),
        "liquidation_event_count": int(
            _safe_float(source.get("liquidation_event_count_total", source.get("liquidation_event_count")), 0.0)
        ),
        "minimum_margin_buffer": _safe_optional_float(source.get("minimum_margin_buffer")),
        "minimum_margin_ratio": _safe_optional_float(source.get("minimum_margin_ratio")),
        "maximum_liquidation_event_drawdown": _safe_float(
            source.get("maximum_liquidation_event_drawdown"), 0.0
        ),
        "maximum_liquidation_equity_loss_fraction": _safe_float(
            source.get("maximum_liquidation_equity_loss_fraction"), 0.0
        ),
        "account_wipeout": any(_truthy(event.get("account_wipeout")) for event in _as_list(source.get("liquidation_events")) if isinstance(event, Mapping)),
    }


def _candidate_leverage(raw: Mapping[str, Any]) -> float:
    return _safe_float(raw.get("leverage"), CURRENT_BASE_LEVERAGE)


def _candidate_sleeve_count(raw: Mapping[str, Any]) -> int:
    sleeves = raw.get("sleeves")
    if isinstance(sleeves, list | tuple):
        return len(sleeves)
    return int(_safe_float(raw.get("sleeve_count"), CURRENT_BASE_SLEEVE_COUNT))


def _components_from_splits(raw: Mapping[str, Any], splits: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    existing = _as_dict(raw.get("train_val_stability"))
    if existing:
        return {key: _safe_float(value) for key, value in existing.items()}
    train = _as_dict(splits.get("train"))
    validation = _as_dict(splits.get("validation"))
    return {
        "train_monthlyized_return": _safe_float(train.get("monthlyized_return")),
        "validation_monthlyized_return": _safe_float(validation.get("monthlyized_return")),
        "train_sharpe": _safe_float(train.get("sharpe")),
        "validation_sharpe": _safe_float(validation.get("sharpe")),
        "train_sortino": _safe_float(train.get("sortino")),
        "validation_sortino": _safe_float(validation.get("sortino")),
        "train_calmar": _safe_float(train.get("calmar")),
        "validation_calmar": _safe_float(validation.get("calmar")),
        "train_max_drawdown": _safe_float(train.get("max_drawdown")),
        "validation_max_drawdown": _safe_float(validation.get("max_drawdown")),
        "leverage": _candidate_leverage(raw),
        "sleeve_count": float(_candidate_sleeve_count(raw)),
    }


def _selection_policy(raw: Mapping[str, Any], *, benchmark_only: bool = False) -> dict[str, Any]:
    policy = _as_dict(raw.get("selection_policy"))
    locked = str(policy.get("locked_oos") or "report_only_gate_only")
    uses_oos = bool(policy.get("uses_locked_oos_for_selection", False))
    locked_policy = _as_dict(raw.get("locked_oos_policy"))
    if locked_policy:
        uses_oos = bool(locked_policy.get("uses_locked_oos_for_selection", uses_oos))
        if _truthy(locked_policy.get("oos_is_gate_only")) and _truthy(locked_policy.get("oos_is_report_only")):
            locked = "report_only_gate_only"
    return {
        "selection_inputs": list(policy.get("selection_inputs") or ["train", "validation"]),
        "locked_oos": locked,
        "uses_locked_oos_for_selection": uses_oos,
        "benchmark_only": bool(benchmark_only),
    }


def _liquidation_summary(splits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    split_values = [_as_dict(splits.get(name)) for name in ("train", "validation", "oos")]
    total_liquidations = sum(int(_safe_float(split.get("liquidation_count"), 0.0)) for split in split_values)
    buffers = [split.get("minimum_margin_buffer") for split in split_values if split.get("minimum_margin_buffer") is not None]
    ratios = [split.get("minimum_margin_ratio") for split in split_values if split.get("minimum_margin_ratio") is not None]
    max_drawdown = max(_safe_float(split.get("maximum_liquidation_event_drawdown"), 0.0) for split in split_values)
    max_loss = max(_safe_float(split.get("maximum_liquidation_equity_loss_fraction"), 0.0) for split in split_values)
    account_wipeout = any(bool(split.get("account_wipeout")) for split in split_values)
    return {
        "liquidation_count": total_liquidations,
        "minimum_margin_buffer": min(buffers) if buffers else None,
        "minimum_margin_ratio": min(ratios) if ratios else None,
        "maximum_liquidation_event_drawdown": max_drawdown,
        "maximum_liquidation_equity_loss_fraction": max_loss,
        "account_wipeout": account_wipeout,
        "evidence_available": bool(buffers or ratios or total_liquidations),
    }


def _current_base_oos(row: Mapping[str, Any] | None) -> dict[str, float]:
    if not row:
        return {"total_return": 0.0, "max_drawdown": 1.0, "return_mdd": 0.0}
    oos = _as_dict(_as_dict(row.get("splits")).get("oos"))
    return {
        "total_return": _safe_float(oos.get("total_return")),
        "max_drawdown": _safe_float(oos.get("max_drawdown"), 1.0),
        "return_mdd": _safe_float(oos.get("return_mdd")),
    }


def _decision_gates(
    *,
    kind: str,
    candidate_derived: bool,
    benchmark_only: bool,
    diagnostic_not_promoted: bool,
    selection_policy: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, Any]],
    liquidation: Mapping[str, Any],
    current_base_oos: Mapping[str, Any],
) -> dict[str, Any]:
    oos = _as_dict(splits.get("oos"))
    oos_return = _safe_float(oos.get("total_return"))
    oos_mdd = _safe_float(oos.get("max_drawdown"), 1.0)
    oos_return_mdd = _safe_float(oos.get("return_mdd"))
    base_return = _safe_float(current_base_oos.get("total_return"))
    base_return_mdd = _safe_float(current_base_oos.get("return_mdd"))
    margin_buffer = liquidation.get("minimum_margin_buffer")
    liquidation_count = int(_safe_float(liquidation.get("liquidation_count"), 0.0))
    selection_firewall = not bool(selection_policy.get("uses_locked_oos_for_selection"))
    locked_oos_ok = str(selection_policy.get("locked_oos") or "") == "report_only_gate_only"
    evidence_available = bool(liquidation.get("evidence_available")) or kind in {"current_base", "direct_candidate"}
    margin_buffer_positive = margin_buffer is None or _safe_float(margin_buffer) > 0.0
    no_account_wipeout = not bool(liquidation.get("account_wipeout"))
    tiny_liq_ok = (
        liquidation_count <= 1
        and _safe_float(liquidation.get("maximum_liquidation_event_drawdown")) <= 0.005
        and _safe_float(liquidation.get("maximum_liquidation_equity_loss_fraction")) <= 0.005
    )
    liquidation_gate = evidence_available and margin_buffer_positive and no_account_wipeout and tiny_liq_ok
    performance_gate = (
        oos_mdd <= MAX_ACCEPTABLE_OOS_MDD
        and oos_return > base_return
        and oos_return_mdd > base_return_mdd
        and _safe_float(oos.get("sharpe")) >= MIN_OOS_SHARPE
        and _safe_float(oos.get("sortino")) >= MIN_OOS_SORTINO
        and _safe_float(oos.get("smart_sortino")) >= MIN_OOS_SMART_SORTINO
        and _safe_float(oos.get("calmar")) >= MIN_OOS_CALMAR
    )
    eligible = bool(candidate_derived) and not bool(benchmark_only) and kind != "current_base"
    deployable = (
        eligible
        and not bool(diagnostic_not_promoted)
        and selection_firewall
        and locked_oos_ok
        and liquidation_gate
        and performance_gate
    )
    return {
        "eligible_for_candidate_live_promotion": eligible,
        "deployable_candidate": deployable,
        "selection_firewall": selection_firewall,
        "locked_oos_report_gate_only": locked_oos_ok,
        "liquidation_gate": liquidation_gate,
        "liquidation_evidence_available": evidence_available,
        "margin_buffer_positive": margin_buffer_positive,
        "no_account_wipeout": no_account_wipeout,
        "tiny_liquidation_tolerance_ok": tiny_liq_ok,
        "oos_mdd_within_25pct_budget": oos_mdd <= MAX_ACCEPTABLE_OOS_MDD,
        "oos_return_beats_current_base": oos_return > base_return,
        "oos_return_mdd_beats_current_base": oos_return_mdd > base_return_mdd,
        "oos_sharpe_ok": _safe_float(oos.get("sharpe")) >= MIN_OOS_SHARPE,
        "oos_sortino_ok": _safe_float(oos.get("sortino")) >= MIN_OOS_SORTINO,
        "oos_smart_sortino_ok": _safe_float(oos.get("smart_sortino")) >= MIN_OOS_SMART_SORTINO,
        "oos_calmar_ok": _safe_float(oos.get("calmar")) >= MIN_OOS_CALMAR,
        "diagnostic_not_promoted": bool(diagnostic_not_promoted),
    }


def _comparison_row(
    *,
    name: str,
    kind: str,
    raw: Mapping[str, Any],
    source_artifact: str,
    candidate_derived: bool,
    benchmark_only: bool,
    current_base_oos: Mapping[str, Any],
    diagnostic_not_promoted: bool = False,
) -> dict[str, Any]:
    splits = {split: _normalize_split(raw, split) for split in ("train", "validation", "oos")}
    components = _components_from_splits(raw, splits)
    score = _safe_optional_float(raw.get("train_val_stability_score"))
    formula_score = train_val_stability_score_from_components(components)
    if score is None or not math.isfinite(score):
        score = formula_score
    policy = _selection_policy(raw, benchmark_only=benchmark_only)
    liquidation = _liquidation_summary(splits)
    gates = _decision_gates(
        kind=kind,
        candidate_derived=candidate_derived,
        benchmark_only=benchmark_only,
        diagnostic_not_promoted=diagnostic_not_promoted,
        selection_policy=policy,
        splits=splits,
        liquidation=liquidation,
        current_base_oos=current_base_oos,
    )
    return {
        "name": name,
        "kind": kind,
        "source_artifact": source_artifact,
        "candidate_derived": bool(candidate_derived),
        "benchmark_only": bool(benchmark_only),
        "leverage": _candidate_leverage(raw),
        "sleeve_count": _candidate_sleeve_count(raw),
        "sleeves": list(raw.get("sleeves") or []),
        "selection_policy": policy,
        "train_val_stability": components,
        "train_val_stability_score": score,
        "train_val_stability_formula_score": formula_score,
        "train_val_stability_formula": "frozen_weighted_train_validation_score_v1",
        "splits": splits,
        "liquidation": liquidation,
        "decision_gates": gates,
        "rejection_reasons": _rejection_reasons(gates),
    }


def _rejection_reasons(gates: Mapping[str, Any]) -> list[str]:
    if bool(gates.get("deployable_candidate")):
        return []
    reasons = []
    for key, value in gates.items():
        if key in {"deployable_candidate", "diagnostic_not_promoted"}:
            continue
        if value is False:
            reasons.append(key)
    if gates.get("diagnostic_not_promoted"):
        reasons.append("diagnostic_not_promoted")
    return reasons or ["not_candidate_live_promotion_row"]


def _add_unique(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row.get("kind"), row.get("name"))
    if any((existing.get("kind"), existing.get("name")) == key for existing in rows):
        return
    rows.append(row)


def _liquidation_rows(liquidation_payload: Mapping[str, Any], source_artifact: str) -> list[tuple[str, str, Mapping[str, Any]]]:
    keys = [
        ("current_base", "current_base_reference_result"),
        ("direct_candidate", "promoted_candidate"),
        ("direct_candidate", "best_deployable_train_validation_retune"),
        ("direct_candidate", "selected_by_train_validation"),
        ("direct_candidate", "selected_by_train_validation_retune"),
        ("direct_candidate", "highest_zero_liquidation_integer"),
        ("direct_candidate", "forced_5x"),
    ]
    out: list[tuple[str, str, Mapping[str, Any]]] = []
    for kind, key in keys:
        raw = liquidation_payload.get(key)
        if isinstance(raw, Mapping) and raw:
            out.append((kind, key, raw))
    retune_results = [item for item in _as_list(liquidation_payload.get("retune_results")) if isinstance(item, Mapping)]
    retune_results.sort(key=lambda item: _safe_float(item.get("train_val_score")), reverse=True)
    for index, raw in enumerate(retune_results[:20], start=1):
        out.append(("direct_candidate", f"retune_results_top_{index:02d}", raw))
    return out


def _candidate_portfolio_rows(candidate_payload: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any]]]:
    keys = [
        ("candidate_portfolio", "selected_by_train_val_stability"),
        ("candidate_portfolio", "best_success_candidate"),
        ("candidate_portfolio", "selected_best_candidate"),
        ("candidate_portfolio", "selected_by_validation"),
        ("candidate_portfolio", "diagnostic_best_oos"),
    ]
    out: list[tuple[str, str, Mapping[str, Any]]] = []
    for kind, key in keys:
        raw = candidate_payload.get(key)
        if isinstance(raw, Mapping) and raw:
            out.append((kind, key, raw))
    for index, raw in enumerate(_as_list(candidate_payload.get("diagnostic_quarantine"))[:10], start=1):
        if isinstance(raw, Mapping) and raw:
            out.append(("candidate_portfolio", f"diagnostic_quarantine_{index:02d}", raw))
    return out


def _legacy_hybrid_raw(legacy_hybrid_payload: Mapping[str, Any]) -> dict[str, Any]:
    if not legacy_hybrid_payload:
        return {}
    scenario = _as_dict(_as_dict(legacy_hybrid_payload.get("scenarios")).get("refreshed_latest_tail"))
    split_metrics = _as_dict(scenario.get("split_metrics"))
    oos_metrics = _as_dict(split_metrics.get("oos")) or _as_dict(legacy_hybrid_payload.get("oos"))
    if not oos_metrics:
        return {}
    return {
        "name": str(legacy_hybrid_payload.get("name") or "legacy_hybrid_benchmark"),
        "splits": {
            "oos": {"metrics": oos_metrics},
            "train": {"metrics": _as_dict(split_metrics.get("train"))},
            "validation": {"metrics": _as_dict(split_metrics.get("validation", split_metrics.get("val")))},
        },
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
        },
    }


def _cash_row(current_base_oos: Mapping[str, Any]) -> dict[str, Any]:
    raw = {
        "name": "cash",
        "splits": {
            "train": {"metrics": _metrics_zero()},
            "validation": {"metrics": _metrics_zero()},
            "oos": {"metrics": _metrics_zero()},
        },
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
        },
        "leverage": 1.0,
        "sleeve_count": 0,
    }
    return _comparison_row(
        name="cash",
        kind="cash",
        raw=raw,
        source_artifact="synthetic_cash",
        candidate_derived=False,
        benchmark_only=True,
        current_base_oos=current_base_oos,
    )


def _metrics_zero() -> dict[str, float]:
    return {
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "smart_sortino": 0.0,
        "calmar": 0.0,
        "volatility": 0.0,
    }


def _artifact_oos_end_dates(payload: Mapping[str, Any], prefix: str = "") -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if str(key).lower() in {"oos_end_date", "oos_end", "oos_end_inclusive"}:
            token = str(value or "").strip()[:10]
            if token:
                out.append({"path": path, "date": token})
        elif isinstance(value, Mapping):
            out.extend(_artifact_oos_end_dates(value, path))
    return out


def _cutoff_gate(
    *,
    latest_complete_oos_end_date: str,
    artifacts: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    stale: list[dict[str, str]] = []
    observed: list[dict[str, str]] = []
    latest = date.fromisoformat(latest_complete_oos_end_date)
    for artifact_name, payload in artifacts:
        for item in _artifact_oos_end_dates(payload):
            observed_item = {"artifact": artifact_name, **item}
            observed.append(observed_item)
            try:
                parsed = date.fromisoformat(item["date"])
            except ValueError:
                continue
            if parsed < latest:
                stale.append(observed_item)
    return {
        "artifact_cutoff_gate_passed": not stale,
        "observed_artifact_oos_end_dates": observed,
        "stale_artifact_oos_end_dates": stale,
    }


def _time_log_entry(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    match = _TIME_RSS_RE.search(text)
    peak = int(match.group(1)) * 1024 if match else 0
    return {
        "kind": "time_log",
        "path": str(path),
        "peak_rss_bytes": peak,
        "peak_rss_mib": round(peak / (1024 * 1024), 3),
        "under_8gib": peak > 0 and peak < EIGHT_GIB_BYTES,
    }


def _artifact_memory_entry(index: int, artifact: Mapping[str, Any]) -> dict[str, Any] | None:
    summary = _as_dict(artifact.get("memory_summary"))
    if not summary:
        peak_mib = _safe_optional_float(artifact.get("peak_rss_mib"))
        if peak_mib is None:
            return None
        peak = int(peak_mib * 1024 * 1024)
    else:
        peak = int(_safe_float(summary.get("peak_rss_bytes"), _safe_float(summary.get("peak_rss_mib")) * 1024 * 1024))
    return {
        "kind": "artifact_memory_summary",
        "artifact_index": index,
        "peak_rss_bytes": peak,
        "peak_rss_mib": round(peak / (1024 * 1024), 3),
        "under_8gib": 0 < peak < EIGHT_GIB_BYTES,
    }


def build_memory_ledger(
    *,
    artifacts: Iterable[Mapping[str, Any]],
    time_logs: Iterable[Path | str],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for index, artifact in enumerate(artifacts):
        entry = _artifact_memory_entry(index, artifact)
        if entry is not None:
            entries.append(entry)
    for path in time_logs:
        entries.append(_time_log_entry(Path(path)))
    return {
        "limit_bytes": EIGHT_GIB_BYTES,
        "limit_gib": 8,
        "entries": entries,
        "under_8gib": bool(entries) and all(bool(entry.get("under_8gib")) for entry in entries),
    }


def build_final_selection_payload(
    *,
    refresh_payload: Mapping[str, Any],
    candidate_portfolio_payload: Mapping[str, Any],
    liquidation_payload: Mapping[str, Any],
    legacy_hybrid_payload: Mapping[str, Any] | None = None,
    source_artifacts: Mapping[str, str] | SourceArtifacts | None = None,
    time_logs: Iterable[Path | str] | None = None,
    required_symbols: Iterable[str] | None = None,
) -> dict[str, Any]:
    if isinstance(source_artifacts, SourceArtifacts):
        sources = asdict(source_artifacts)
    else:
        sources = dict(source_artifacts or {})
    data_cutoff = derive_data_cutoff(refresh_payload, required_symbols=required_symbols)
    cutoff_gate = _cutoff_gate(
        latest_complete_oos_end_date=str(data_cutoff["latest_complete_oos_end_date"]),
        artifacts=(
            ("candidate_portfolio", candidate_portfolio_payload),
            ("liquidation", liquidation_payload),
            ("legacy_hybrid", legacy_hybrid_payload or {}),
        ),
    )
    data_cutoff.update(cutoff_gate)

    artifacts_for_memory = [refresh_payload, candidate_portfolio_payload, liquidation_payload]
    if legacy_hybrid_payload:
        artifacts_for_memory.append(legacy_hybrid_payload)
    memory_ledger = build_memory_ledger(artifacts=artifacts_for_memory, time_logs=list(time_logs or []))

    rows: list[dict[str, Any]] = []
    current_base_raw = _as_dict(liquidation_payload.get("current_base_reference_result"))
    current_base_placeholder: dict[str, Any] | None = None
    if current_base_raw:
        current_base_placeholder = _comparison_row(
            name=str(current_base_raw.get("name") or "current_base"),
            kind="current_base",
            raw=current_base_raw,
            source_artifact=sources.get("liquidation_json", ""),
            candidate_derived=False,
            benchmark_only=True,
            current_base_oos={"total_return": 0.0, "return_mdd": 0.0, "max_drawdown": 1.0},
        )
    base_oos = _current_base_oos(current_base_placeholder)
    if current_base_raw:
        current_base_row = _comparison_row(
            name=str(current_base_raw.get("name") or "current_base"),
            kind="current_base",
            raw=current_base_raw,
            source_artifact=sources.get("liquidation_json", ""),
            candidate_derived=False,
            benchmark_only=True,
            current_base_oos=base_oos,
        )
        _add_unique(rows, current_base_row)

    for kind, source_key, raw in _liquidation_rows(liquidation_payload, sources.get("liquidation_json", "")):
        if source_key == "current_base_reference_result":
            continue
        name = str(raw.get("name") or raw.get("candidate_name") or source_key)
        _add_unique(
            rows,
            _comparison_row(
                name=name,
                kind=kind,
                raw=raw,
                source_artifact=sources.get("liquidation_json", ""),
                candidate_derived=True,
                benchmark_only=False,
                current_base_oos=base_oos,
                diagnostic_not_promoted=bool(raw.get("diagnostic_not_promoted")),
            ),
        )

    for kind, source_key, raw in _candidate_portfolio_rows(candidate_portfolio_payload):
        name = str(raw.get("name") or source_key)
        _add_unique(
            rows,
            _comparison_row(
                name=name,
                kind=kind,
                raw=raw,
                source_artifact=sources.get("candidate_portfolio_json", ""),
                candidate_derived=True,
                benchmark_only=False,
                current_base_oos=base_oos,
                diagnostic_not_promoted=bool(raw.get("diagnostic_not_promoted")),
            ),
        )

    legacy_raw = _legacy_hybrid_raw(legacy_hybrid_payload or {})
    if legacy_raw:
        _add_unique(
            rows,
            _comparison_row(
                name=str(legacy_raw.get("name") or "legacy_hybrid_benchmark"),
                kind="legacy_hybrid_benchmark",
                raw=legacy_raw,
                source_artifact=sources.get("legacy_hybrid_json", ""),
                candidate_derived=False,
                benchmark_only=True,
                current_base_oos=base_oos,
            ),
        )
    _add_unique(rows, _cash_row(base_oos))

    if not data_cutoff["artifact_cutoff_gate_passed"]:
        for row in rows:
            row["decision_gates"]["artifact_cutoff_current"] = False
            row["decision_gates"]["deployable_candidate"] = False
            row["rejection_reasons"] = sorted(set(row["rejection_reasons"] + ["artifact_cutoff_stale"]))

    if not memory_ledger["under_8gib"]:
        for row in rows:
            row["decision_gates"]["memory_under_8gib"] = False
            row["decision_gates"]["deployable_candidate"] = False
            row["rejection_reasons"] = sorted(set(row["rejection_reasons"] + ["memory_over_8gib_or_missing"]))

    contenders = [row for row in rows if bool(row["decision_gates"].get("deployable_candidate"))]
    contenders.sort(
        key=lambda row: (
            _safe_float(row.get("train_val_stability_score")),
            -_safe_float(_as_dict(row.get("liquidation")).get("liquidation_count")),
            str(row.get("name") or ""),
        ),
        reverse=True,
    )
    winner = contenders[0] if contenders else None
    if not data_cutoff["artifact_cutoff_gate_passed"]:
        status = "failed_stale_artifact_cutoff"
    elif not memory_ledger["under_8gib"]:
        status = "failed_memory_gate"
    elif winner:
        status = "promote_candidate"
    else:
        status = "no_live_promotion"

    return {
        "artifact_kind": "profit_moonshot_live_final_selection",
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "recommendation": "promote" if winner else "no_live_promotion",
        "winner": _winner_summary(winner) if winner else None,
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
            "ranking_formula": "frozen_weighted_train_validation_score_v1",
            "benchmark_only_rows_ineligible": True,
        },
        "data_cutoff": data_cutoff,
        "source_artifacts": sources,
        "rows": rows,
        "rejected_alternatives": _rejected_alternatives(rows, winner),
        "metrics_explanation": METRICS_EXPLANATION,
        "memory_ledger": memory_ledger,
        "verification": {
            "required_tests": [
                "tests/test_profit_moonshot_live_final_selection.py -q",
                "tests/test_profit_moonshot_liquidation_aware_validation.py -q",
                "full pytest",
                "ruff check .",
                "compileall",
                "git diff --check",
                "GitHub Actions ci/private-ci green",
            ],
            "finalized": False,
        },
    }


def _winner_summary(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "name": row.get("name"),
        "kind": row.get("kind"),
        "source_artifact": row.get("source_artifact"),
        "train_val_stability_score": row.get("train_val_stability_score"),
        "oos_total_return": _as_dict(_as_dict(row.get("splits")).get("oos")).get("total_return"),
        "oos_max_drawdown": _as_dict(_as_dict(row.get("splits")).get("oos")).get("max_drawdown"),
        "oos_return_mdd": _as_dict(_as_dict(row.get("splits")).get("oos")).get("return_mdd"),
        "liquidation_count": _as_dict(row.get("liquidation")).get("liquidation_count"),
        "minimum_margin_buffer": _as_dict(row.get("liquidation")).get("minimum_margin_buffer"),
    }


def _rejected_alternatives(rows: Iterable[Mapping[str, Any]], winner: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    winner_key = (winner or {}).get("kind"), (winner or {}).get("name")
    out: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("kind"), row.get("name")
        if key == winner_key:
            continue
        out.append(
            {
                "name": row.get("name"),
                "kind": row.get("kind"),
                "candidate_derived": row.get("candidate_derived"),
                "benchmark_only": row.get("benchmark_only"),
                "reasons": list(row.get("rejection_reasons") or []),
            }
        )
    return out


def build_markdown(payload: Mapping[str, Any]) -> str:
    winner = _as_dict(payload.get("winner"))
    lines = [
        "# Profit moonshot live final selection",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        f"- status: `{payload.get('status')}`",
        f"- recommendation: `{payload.get('recommendation')}`",
        f"- latest_complete_oos_end_date: `{_as_dict(payload.get('data_cutoff')).get('latest_complete_oos_end_date')}`",
        f"- winner: `{winner.get('name') if winner else None}`",
        "",
        "## Comparison rows",
        "",
        "| Kind | Name | Candidate-derived | Benchmark-only | TV score | OOS return | OOS MDD | OOS return/MDD | Liq | Min buffer | Deployable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(payload.get("rows") or []):
        if not isinstance(row, Mapping):
            continue
        oos = _as_dict(_as_dict(row.get("splits")).get("oos"))
        liq = _as_dict(row.get("liquidation"))
        gates = _as_dict(row.get("decision_gates"))
        lines.append(
            "| "
            f"`{row.get('kind')}` | `{row.get('name')}` | `{bool(row.get('candidate_derived'))}` | "
            f"`{bool(row.get('benchmark_only'))}` | {_fmt_float(row.get('train_val_stability_score'))} | "
            f"{_fmt_pct(oos.get('total_return'))} | {_fmt_pct(oos.get('max_drawdown'))} | "
            f"{_fmt_float(oos.get('return_mdd'))} | {liq.get('liquidation_count')} | "
            f"{_fmt_float(liq.get('minimum_margin_buffer'))} | `{bool(gates.get('deployable_candidate'))}` |"
        )
    lines.extend(["", "## Metric explanations", ""])
    for key, description in METRICS_EXPLANATION.items():
        lines.append(f"- **{key}**: {description}")
    lines.extend(["", "## Memory ledger", ""])
    memory = _as_dict(payload.get("memory_ledger"))
    lines.append(f"- under_8gib: `{memory.get('under_8gib')}`")
    for entry in list(memory.get("entries") or []):
        if isinstance(entry, Mapping):
            lines.append(
                f"- `{entry.get('kind')}` `{entry.get('path', entry.get('artifact_index'))}`: "
                f"max_rss_mib=`{entry.get('peak_rss_mib')}`, under_8gib=`{entry.get('under_8gib')}`"
            )
    return "\n".join(lines) + "\n"


def _fmt_float(value: Any) -> str:
    parsed = _safe_optional_float(value)
    return "" if parsed is None else f"{parsed:.6f}"


def _fmt_pct(value: Any) -> str:
    parsed = _safe_optional_float(value)
    return "" if parsed is None else f"{parsed * 100:.4f}%"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-json", required=True)
    parser.add_argument("--candidate-portfolio-json", required=True)
    parser.add_argument("--liquidation-json", required=True)
    parser.add_argument("--legacy-hybrid-json", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--required-symbols", default=DEFAULT_REQUIRED_SYMBOLS)
    parser.add_argument("--time-log", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    refresh_path = Path(args.refresh_json)
    candidate_path = Path(args.candidate_portfolio_json)
    liquidation_path = Path(args.liquidation_json)
    hybrid_path = Path(args.legacy_hybrid_json) if str(args.legacy_hybrid_json).strip() else None
    payload = build_final_selection_payload(
        refresh_payload=_read_json(refresh_path),
        candidate_portfolio_payload=_read_json(candidate_path),
        liquidation_payload=_read_json(liquidation_path),
        legacy_hybrid_payload=_read_json(hybrid_path),
        source_artifacts={
            "refresh_json": str(refresh_path),
            "candidate_portfolio_json": str(candidate_path),
            "liquidation_json": str(liquidation_path),
            "legacy_hybrid_json": str(hybrid_path or ""),
        },
        time_logs=[Path(path) for path in args.time_log],
        required_symbols=_parse_required_symbols(str(args.required_symbols)),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "profit_moonshot_live_final_selection_latest.json"
    md_path = output_dir / "profit_moonshot_live_final_selection_latest.md"
    _write_json(json_path, payload)
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path), "status": payload["status"]}))
    return 2 if str(payload.get("status") or "").startswith("failed_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
