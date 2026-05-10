#!/usr/bin/env python3
"""Build a candidate-derived hybrid strategy for the profit-moonshot sleeve set.

This runner is deliberately separate from the legacy hybrid-online benchmark:
it takes the freshly tuned/liquidation-aware profit-moonshot candidate rows,
reconstructs their sleeve return streams from the same refreshed market data,
tunes a deterministic online hybrid allocator on train/validation only, and
reports locked-OOS as report-only/gate-only evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.portfolio_split_contract import (  # noqa: E402
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

TUNER_PATH = REPO_ROOT / "scripts/research/tune_profit_moonshot_fresh_portfolio.py"
FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
HYBRID_PATH = REPO_ROOT / "scripts/research/run_hybrid_online_portfolio.py"

DEFAULT_ROOT = REPO_ROOT / "var/reports/profit_moonshot_20260501/live_final_selection_20260510"
DEFAULT_CANDIDATE_PORTFOLIO_JSON = (
    DEFAULT_ROOT / "candidate_portfolio/fresh_portfolio_tuning_latest.json"
)
DEFAULT_LIQUIDATION_JSON = (
    DEFAULT_ROOT / "liquidation_validation/liquidation_aware_current_base_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "candidate_hybrid"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market_parquet"
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT"
RUN_NAME = "profit_moonshot_candidate_hybrid"
STARTING_EQUITY = 10_000.0


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path | str | None) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _safe_optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return float(parsed) if math.isfinite(parsed) else None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _slug(value: str, *, max_len: int = 120) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    while "__" in out:
        out = out.replace("__", "_")
    return (out.strip("_") or "candidate_hybrid")[:max_len]


def _display_split_name(name: str) -> str:
    return "validation" if str(name) == "val" else str(name)


def _split_metrics(payload: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    splits = dict(payload.get("splits") or {})
    key = "val" if split_name == "validation" and "val" in splits else split_name
    split = dict(splits.get(key) or {})
    metrics = dict(split.get("metrics") or split)
    return metrics


def _candidate_train_val_score(tuner: Any, payload: Mapping[str, Any]) -> float:
    existing = _safe_optional_float(payload.get("train_val_stability_score"))
    if existing is not None:
        return existing
    train = _split_metrics(payload, "train")
    val = _split_metrics(payload, "validation")
    components = {
        "train_monthlyized_return": _monthlyized(tuner, train),
        "validation_monthlyized_return": _monthlyized(tuner, val),
        "train_sharpe": _safe_float(train.get("sharpe")),
        "validation_sharpe": _safe_float(val.get("sharpe")),
        "train_sortino": _safe_float(train.get("sortino")),
        "validation_sortino": _safe_float(val.get("sortino")),
        "train_calmar": _safe_float(train.get("calmar")),
        "validation_calmar": _safe_float(val.get("calmar")),
        "train_max_drawdown": _safe_float(train.get("max_drawdown"), 1.0),
        "validation_max_drawdown": _safe_float(val.get("max_drawdown"), 1.0),
        "leverage": _safe_float(payload.get("leverage"), 1.0),
        "sleeve_count": float(len(list(payload.get("sleeves") or [])) or 1),
    }
    return float(tuner._train_val_stability_score_from_components(components))


def _monthlyized(tuner: Any, metrics: Mapping[str, Any]) -> float:
    return float(tuner._monthlyized_return(dict(metrics)))


def _smart_sortino(tuner: Any, metrics: Mapping[str, Any]) -> float:
    return float(tuner._smart_sortino(dict(metrics)))


def _with_extra_metrics(tuner: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    total_return = _safe_float(out.get("total_return", out.get("return")))
    max_drawdown = _safe_float(out.get("max_drawdown", out.get("mdd")), 0.0)
    out["total_return"] = total_return
    out["max_drawdown"] = max_drawdown
    out["return_mdd"] = total_return / max(1e-9, max_drawdown)
    out["monthlyized_return"] = _monthlyized(tuner, out)
    out["smart_sortino"] = _smart_sortino(tuner, out)
    return out


def _result_score(tuner: Any, result: Mapping[str, Any], *, sleeve_count: int) -> float:
    splits = dict(result.get("split_metrics") or {})
    train = _with_extra_metrics(tuner, dict(splits.get("train") or {}))
    val = _with_extra_metrics(tuner, dict(splits.get("val") or splits.get("validation") or {}))
    components = {
        "train_monthlyized_return": _safe_float(train.get("monthlyized_return")),
        "validation_monthlyized_return": _safe_float(val.get("monthlyized_return")),
        "train_sharpe": _safe_float(train.get("sharpe")),
        "validation_sharpe": _safe_float(val.get("sharpe")),
        "train_sortino": _safe_float(train.get("sortino")),
        "validation_sortino": _safe_float(val.get("sortino")),
        "train_calmar": _safe_float(train.get("calmar")),
        "validation_calmar": _safe_float(val.get("calmar")),
        "train_max_drawdown": _safe_float(train.get("max_drawdown"), 1.0),
        "validation_max_drawdown": _safe_float(val.get("max_drawdown"), 1.0),
        "leverage": 1.0,
        "sleeve_count": float(sleeve_count),
    }
    return float(tuner._train_val_stability_score_from_components(components))


def _iter_candidate_rows(
    *,
    candidate_payload: Mapping[str, Any],
    liquidation_payload: Mapping[str, Any],
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(kind: str, source_key: str, raw: Any, source_artifact: str) -> None:
        if not isinstance(raw, Mapping):
            return
        sleeves = [str(item) for item in list(raw.get("sleeves") or []) if str(item)]
        if not sleeves:
            return
        item = dict(raw)
        item["_candidate_hybrid_source_kind"] = kind
        item["_candidate_hybrid_source_key"] = source_key
        item["_candidate_hybrid_source_artifact"] = source_artifact
        rows.append(item)

    for key in (
        "promoted_candidate",
        "best_deployable_train_validation_retune",
        "selected_by_train_validation_retune",
        "forced_5x",
        "highest_zero_liquidation_integer",
    ):
        add("liquidation", key, liquidation_payload.get(key), "liquidation_json")
    for idx, raw in enumerate(list(liquidation_payload.get("retune_results") or [])[: max_rows]):
        add("liquidation", f"retune_results_{idx:02d}", raw, "liquidation_json")

    for key in (
        "selected_by_train_val_stability",
        "best_success_candidate",
        "selected_best_candidate",
        "selected_by_validation",
        "diagnostic_best_oos",
    ):
        add("candidate_portfolio", key, candidate_payload.get(key), "candidate_portfolio_json")
    for idx, raw in enumerate(list(candidate_payload.get("diagnostic_quarantine") or [])[: max_rows]):
        add("candidate_portfolio", f"diagnostic_quarantine_{idx:02d}", raw, "candidate_portfolio_json")

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], str, float]] = set()
    for item in rows:
        sleeves = tuple(str(name) for name in list(item.get("sleeves") or []))
        key = (sleeves, str(item.get("mode") or "train_val_monthly_return_budget"), round(_safe_float(item.get("leverage"), 1.0), 8))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[: max(1, int(max_rows))]


def _split_indices(fresh: Any, arrays: Mapping[str, Any], split: Any) -> np.ndarray:
    timestamps = arrays["timestamp"]
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = int(datetime.combine(split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()) - 1
    return np.flatnonzero((timestamps >= start_ts) & (timestamps <= end_ts))


def _timestamp_for_idx(arrays: Mapping[str, Any], idx: int) -> str:
    raw = int(arrays["timestamp"][idx])
    return datetime.fromtimestamp(raw, tz=UTC).isoformat().replace("+00:00", "Z")


def _returns_stream_from_equity(
    *,
    equity: Iterable[float],
    indices: Iterable[int],
    arrays: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stream: list[dict[str, Any]] = []
    previous = STARTING_EQUITY
    for value, idx in zip(list(equity), list(indices), strict=False):
        current = _safe_float(value, previous)
        ret = current / previous - 1.0 if previous > 1e-12 else 0.0
        stream.append({"t": _timestamp_for_idx(arrays, int(idx)), "v": float(ret)})
        previous = max(1e-12, current)
    return stream


def _group_equity(
    *,
    tuner: Any,
    row: Mapping[str, Any],
    curves: list[list[float]],
) -> list[float]:
    mode = str(row.get("mode") or "train_val_monthly_return_budget")
    weights = list(row.get("weights") or [])
    parsed_weights = [_safe_float(item) for item in weights] if weights else None
    if mode in {"validation_return_risk_weight", "validation_drawdown_budget", "cluster_capped_validation_weight"} and not parsed_weights:
        mode = "equal_weight"
    return tuner._combine_equity(
        curves,
        mode=mode,
        weights=parsed_weights,
        leverage=_safe_float(row.get("leverage"), 1.0),
    )


def _build_candidate_hybrid_rows(
    *,
    fresh: Any,
    tuner: Any,
    arrays: Mapping[str, Any],
    splits: list[Any],
    specs_by_name: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    split_curves: Mapping[str, Mapping[str, list[float]]],
    split_payloads: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    active_rows: list[dict[str, Any]] = []
    split_index_by_name = {_display_split_name(split.name): _split_indices(fresh, arrays, split) for split in splits}
    for source_idx, source_row in enumerate(candidate_rows, start=1):
        sleeves = [str(name) for name in list(source_row.get("sleeves") or []) if str(name) in specs_by_name]
        if not sleeves:
            continue
        streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
        split_metrics: dict[str, dict[str, Any]] = {}
        fills_by_split: dict[str, int] = {}
        trips_by_split: dict[str, int] = {}
        for split in splits:
            split_name = _display_split_name(split.name)
            raw_name = "val" if split_name == "validation" else split_name
            equity = _group_equity(
                tuner=tuner,
                row=source_row,
                curves=[split_curves[name][raw_name] for name in sleeves],
            )
            streams[raw_name] = _returns_stream_from_equity(
                equity=equity,
                indices=split_index_by_name[split_name],
                arrays=arrays,
            )
            metrics = fresh._metrics_from_equity_totals(
                equity,
                periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
            )
            split_metrics[raw_name] = _with_extra_metrics(tuner, metrics)
            fills_by_split[raw_name] = int(
                sum(int((split_payloads[name][raw_name]).get("fills") or 0) for name in sleeves)
            )
            trips_by_split[raw_name] = int(
                sum(int((split_payloads[name][raw_name]).get("round_trips") or 0) for name in sleeves)
            )
        candidate_id = f"candidate_hybrid_input_{source_idx:02d}_{_slug(str(source_row.get('name') or source_idx), max_len=80)}"
        active_rows.append(
            {
                "candidate_id": candidate_id,
                "name": candidate_id,
                "strategy_class": "ProfitMoonshotCandidatePortfolio",
                "strategy_timeframe": "1h_to_1d_compounded",
                "family": "profit_moonshot_candidate",
                "symbols": list(arrays.get("symbols") or []),
                "return_streams": streams,
                "metadata": {
                    "source_name": str(source_row.get("name") or ""),
                    "source_kind": str(source_row.get("_candidate_hybrid_source_kind") or ""),
                    "source_key": str(source_row.get("_candidate_hybrid_source_key") or ""),
                    "source_artifact": str(source_row.get("_candidate_hybrid_source_artifact") or ""),
                    "mode": str(source_row.get("mode") or "train_val_monthly_return_budget"),
                    "leverage": _safe_float(source_row.get("leverage"), 1.0),
                    "sleeves": sleeves,
                    "source_train_val_score": _candidate_train_val_score(tuner, source_row),
                    "max_weight_cap": 0.65,
                },
                "train": {**split_metrics["train"], "fills": fills_by_split["train"], "round_trips": trips_by_split["train"]},
                "val": {**split_metrics["val"], "fills": fills_by_split["val"], "round_trips": trips_by_split["val"]},
                "oos": {**split_metrics["oos"], "fills": fills_by_split["oos"], "round_trips": trips_by_split["oos"]},
            }
        )
    return active_rows


def _cash_row_from_active(hybrid: Any, active_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_days = sorted(set().union(*(hybrid._merged_daily_map(row["return_streams"]).keys() for row in active_rows)))
    return {
        "candidate_id": "risk_off_cash",
        "name": "risk_off_cash",
        "strategy_class": "CashSleeve",
        "strategy_timeframe": "1d",
        "family": "cash",
        "symbols": [],
        "return_streams": hybrid._DYN._portfolio_return_streams_from_daily(all_days, [0.0] * len(all_days)),
        "metadata": {"source_payload_path": ""},
        "train": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        "val": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        "oos": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
    }


def _config_grid(hybrid: Any) -> list[Any]:
    base = asdict(hybrid.HybridOnlineConfig())
    configs = []
    for variant in ("fixed_default", "disagreement_switching"):
        for lookback in (5, 13, 21):
            for min_positive in (0.0, 0.10):
                for temp in (0.90,):
                    for default_boost in (0.0, 0.12, 0.25):
                        cfg = {
                            **base,
                            "variant": variant,
                            "warmup_days": max(lookback, 7),
                            "lookback_days": lookback,
                            "min_positive_score": min_positive,
                            "score_temperature": temp,
                            "default_boost": default_boost,
                            "diversified_weight_cap": 0.75,
                            "pair_weight_cap": 0.20,
                            "use_current_health_priors": False,
                        }
                        configs.append(hybrid.HybridOnlineConfig(**cfg))
    return configs


def _row_splits_from_result(tuner: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    split_metrics = dict(result.get("split_metrics") or {})
    out: dict[str, Any] = {}
    for source_name, target_name in (("train", "train"), ("val", "validation"), ("oos", "oos")):
        out[target_name] = {
            "metrics": _with_extra_metrics(tuner, dict(split_metrics.get(source_name) or {})),
            "fills": 0,
            "round_trips": 0,
            "liquidation_count": 0,
            "liquidation_event_count_total": 0,
            "liquidation_events": [],
        }
    return out


def _candidate_hybrid_record(
    *,
    tuner: Any,
    result: Mapping[str, Any],
    score: float,
    rank: int,
    config: Any,
    default_name: str,
    selected: bool,
) -> dict[str, Any]:
    splits = _row_splits_from_result(tuner, result)
    return {
        "name": f"candidate_hybrid_online_rank_{rank:02d}_{_slug(default_name, max_len=60)}",
        "mode": "candidate_online_hybrid",
        "leverage": 1.0,
        "sleeves": sorted(
            {
                sleeve
                for allocation in list(result.get("allocations") or [])
                for sleeve in dict(allocation.get("weights") or {})
            }
        ),
        "sleeve_count": len(
            {
                sleeve
                for allocation in list(result.get("allocations") or [])
                for sleeve in dict(allocation.get("weights") or {})
            }
        ),
        "splits": splits,
        "train_val_stability_score": float(score),
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
            "selection_basis": "train_validation_only_candidate_hybrid_grid",
        },
        "hybrid_config": asdict(config),
        "default_candidate": default_name,
        "selected_by_train_validation": bool(selected),
        "locked_oos_policy": {
            "oos_is_report_only": True,
            "oos_is_gate_only": True,
            "uses_locked_oos_for_selection": False,
        },
        "liquidation_evidence_note": (
            "candidate-hybrid allocator uses reconstructed candidate portfolio return streams; "
            "no dynamic-weight liquidation replay is claimed, so final live writer must keep "
            "this row non-promotable until a dedicated margin replay exists."
        ),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    fresh = _load_module(FRESH_PATH, "profit_moonshot_fresh_replay_for_candidate_hybrid")
    tuner = _load_module(TUNER_PATH, "profit_moonshot_fresh_tuner_for_candidate_hybrid")
    hybrid = _load_module(HYBRID_PATH, "profit_moonshot_legacy_hybrid_module_for_candidate_hybrid")

    candidate_payload = _load_json(args.candidate_portfolio_json)
    liquidation_payload = _load_json(args.liquidation_json)
    oos_end_token = str(args.oos_end_date or candidate_payload.get("oos_end_date") or liquidation_payload.get("oos_end_date") or "")
    if not oos_end_token:
        raise RuntimeError("oos_end_date is required when source artifacts do not carry one")
    oos_end = datetime.fromisoformat(oos_end_token[:10]).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
    )
    arrays = fresh._build_arrays(panel, symbols)
    specs_by_name = {spec.name: spec for spec in fresh._candidate_specs(arrays, symbols)}

    candidate_rows = _iter_candidate_rows(
        candidate_payload=candidate_payload,
        liquidation_payload=liquidation_payload,
        max_rows=int(args.max_candidate_rows),
    )
    required_sleeves = sorted({name for row in candidate_rows for name in list(row.get("sleeves") or [])})
    missing_sleeves = [name for name in required_sleeves if name not in specs_by_name]
    available_sleeves = [name for name in required_sleeves if name in specs_by_name]
    split_curves: dict[str, dict[str, list[float]]] = {}
    split_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for name in available_sleeves:
        split_curves[name] = {}
        split_payloads[name] = {}
        for split in splits:
            result = fresh._run_split(
                spec=specs_by_name[name],
                arrays=arrays,
                split=split,
                include_equity=True,
            )
            raw_name = str(split.name)
            split_curves[name][raw_name] = list(result.get("equity_history") or [])
            split_payloads[name][raw_name] = dict(result)

    active = _build_candidate_hybrid_rows(
        fresh=fresh,
        tuner=tuner,
        arrays=arrays,
        splits=splits,
        specs_by_name=specs_by_name,
        candidate_rows=candidate_rows,
        split_curves=split_curves,
        split_payloads=split_payloads,
    )
    if not active:
        raise RuntimeError("no candidate-hybrid active rows could be reconstructed")
    active.sort(
        key=lambda row: (
            _safe_float(dict(row.get("metadata") or {}).get("source_train_val_score")),
            _safe_float(dict(row.get("val") or {}).get("total_return")),
        ),
        reverse=True,
    )
    active = active[: max(1, int(args.max_active_inputs))]
    rows = [_cash_row_from_active(hybrid, active), *active]
    split_config = hybrid.HybridSplitConfig(oos_end=oos_end.isoformat())

    default_candidates = [str(row.get("candidate_id")) for row in active[: max(1, int(args.default_candidate_count))]]
    tuning_results: list[dict[str, Any]] = []
    for default_name in default_candidates:
        for config in _config_grid(hybrid):
            result = hybrid.run_hybrid_online_allocator(
                rows,
                config=config,
                refreshed_health_metrics=None,
                split_config=split_config,
                default_name=default_name,
            )
            score = _result_score(tuner, result, sleeve_count=len(active))
            split_metrics = dict(result.get("split_metrics") or {})
            train = dict(split_metrics.get("train") or {})
            val = dict(split_metrics.get("val") or {})
            train_val_gate = (
                _safe_float(train.get("total_return")) > 0.0
                and _safe_float(val.get("total_return")) > 0.0
                and _safe_float(train.get("max_drawdown"), 1.0) <= 0.25
                and _safe_float(val.get("max_drawdown"), 1.0) <= 0.25
            )
            tuning_results.append(
                {
                    "score": float(score),
                    "train_val_gate": bool(train_val_gate),
                    "default_name": default_name,
                    "config": asdict(config),
                    "result": {
                        "split_metrics": {
                            name: _with_extra_metrics(tuner, dict(metrics))
                            for name, metrics in split_metrics.items()
                        },
                        "final_allocation": dict(result.get("final_allocation") or {}),
                        "resolved_warmup_days": int(result.get("resolved_warmup_days") or 0),
                    },
                }
            )
    selection_pool = [item for item in tuning_results if bool(item["train_val_gate"])] or tuning_results
    selection_pool.sort(
        key=lambda item: (
            _safe_float(item.get("score")),
            _safe_float(dict(dict(item.get("result") or {}).get("split_metrics") or {}).get("val", {}).get("total_return")),
        ),
        reverse=True,
    )
    best_tuning = selection_pool[0]
    best_config = hybrid.HybridOnlineConfig(**dict(best_tuning["config"]))
    best_result = hybrid.run_hybrid_online_allocator(
        rows,
        config=best_config,
        refreshed_health_metrics=None,
        split_config=split_config,
        default_name=str(best_tuning["default_name"]),
    )
    ranked = sorted(
        tuning_results,
        key=lambda item: (
            bool(item.get("train_val_gate")),
            _safe_float(item.get("score")),
        ),
        reverse=True,
    )
    selected_record = _candidate_hybrid_record(
        tuner=tuner,
        result=best_result,
        score=_result_score(tuner, best_result, sleeve_count=len(active)),
        rank=1,
        config=best_config,
        default_name=str(best_tuning["default_name"]),
        selected=True,
    )
    top_records = [
        _candidate_hybrid_record(
            tuner=tuner,
            result=item["result"],
            score=_safe_float(item.get("score")),
            rank=idx,
            config=hybrid.HybridOnlineConfig(**dict(item["config"])),
            default_name=str(item["default_name"]),
            selected=idx == 1,
        )
        for idx, item in enumerate(ranked[: int(args.report_top_n)], start=1)
    ]
    source_sleeve_metrics = {
        str(row.get("candidate_id")): {
            "source_name": dict(row.get("metadata") or {}).get("source_name"),
            "source_kind": dict(row.get("metadata") or {}).get("source_kind"),
            "mode": dict(row.get("metadata") or {}).get("mode"),
            "leverage": dict(row.get("metadata") or {}).get("leverage"),
            "sleeves": dict(row.get("metadata") or {}).get("sleeves"),
            "train": dict(row.get("train") or {}),
            "validation": dict(row.get("val") or {}),
            "oos": dict(row.get("oos") or {}),
        }
        for row in active
    }
    final_allocation = dict(best_result.get("final_allocation") or {})
    payload = {
        "artifact_kind": "profit_moonshot_candidate_hybrid",
        "generated_at_utc": _utc_now_iso(),
        "oos_end_date": oos_end.isoformat(),
        "selection_basis": "candidate_derived_online_hybrid_train_validation_only",
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only",
            "uses_locked_oos_for_selection": False,
            "tuning_objective": "frozen_weighted_train_validation_score_v1",
        },
        "split_windows": split_config.as_payload(),
        "candidate_basis": {
            "candidate_portfolio_json": str(args.candidate_portfolio_json),
            "liquidation_json": str(args.liquidation_json),
            "source_candidate_rows": len(candidate_rows),
            "available_sleeve_count": len(available_sleeves),
            "missing_sleeves": missing_sleeves,
            "active_hybrid_input_count": len(active),
            "default_candidates": default_candidates,
        },
        "data_metadata": data_metadata,
        "source_sleeve_metrics": source_sleeve_metrics,
        "selected_by_train_validation": selected_record,
        "best_candidate_hybrid": selected_record,
        "tuning_results": top_records,
        "full_tuning_result_count": len(tuning_results),
        "final_allocation": final_allocation,
        "peak_rss_mib": _rss_mib(),
        "memory_policy": memory_policy_payload(budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES),
    }
    return payload


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):+.4%}"


def _fmt_num(value: Any) -> str:
    parsed = _safe_optional_float(value)
    return "" if parsed is None else f"{parsed:.4f}"


def _markdown(payload: Mapping[str, Any]) -> str:
    selected = dict(payload.get("selected_by_train_validation") or {})
    splits = dict(selected.get("splits") or {})
    oos = dict(dict(splits.get("oos") or {}).get("metrics") or {})
    train = dict(dict(splits.get("train") or {}).get("metrics") or {})
    val = dict(dict(splits.get("validation") or {}).get("metrics") or {})
    lines = [
        "# Profit moonshot candidate-derived hybrid",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        f"- oos_end_date: `{payload.get('oos_end_date')}`",
        "- selection basis: train/validation-only candidate hybrid tuning",
        "- locked-OOS: report-only / gate-only",
        "- promotion note: no dynamic-weight liquidation replay is claimed, so this is comparison evidence unless a dedicated margin replay is added.",
        "",
        "## Selected candidate hybrid",
        "",
        f"- name: `{selected.get('name')}`",
        f"- TV score: `{_fmt_num(selected.get('train_val_stability_score'))}`",
        f"- train return/MDD: `{_fmt_pct(train.get('total_return'))}` / `{_fmt_pct(train.get('max_drawdown'))}`",
        f"- validation return/MDD: `{_fmt_pct(val.get('total_return'))}` / `{_fmt_pct(val.get('max_drawdown'))}`",
        f"- OOS return/MDD: `{_fmt_pct(oos.get('total_return'))}` / `{_fmt_pct(oos.get('max_drawdown'))}`",
        f"- OOS return/MDD ratio: `{_fmt_num(oos.get('return_mdd'))}`",
        f"- OOS Sharpe/Sortino/smart Sortino/Calmar: `{_fmt_num(oos.get('sharpe'))}` / `{_fmt_num(oos.get('sortino'))}` / `{_fmt_num(oos.get('smart_sortino'))}` / `{_fmt_num(oos.get('calmar'))}`",
        "",
        "## Final allocation",
        "",
        f"- date: `{dict(payload.get('final_allocation') or {}).get('date')}`",
        f"- cash_weight: `{_fmt_pct(dict(payload.get('final_allocation') or {}).get('cash_weight'))}`",
    ]
    for name, weight in sorted(
        dict(dict(payload.get("final_allocation") or {}).get("weights") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"- `{name}`: `{_fmt_pct(weight)}`")
    lines.extend(
        [
            "",
            "## Top tuned candidate-hybrid rows",
            "",
            "| rank | name | TV score | train | val | OOS | OOS MDD | OOS R/MDD | Sharpe | Sortino | SmartSort | Calmar |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, item in enumerate(list(payload.get("tuning_results") or []), start=1):
        item_splits = dict(item.get("splits") or {})
        item_train = dict(dict(item_splits.get("train") or {}).get("metrics") or {})
        item_val = dict(dict(item_splits.get("validation") or {}).get("metrics") or {})
        item_oos = dict(dict(item_splits.get("oos") or {}).get("metrics") or {})
        lines.append(
            f"| {idx} | `{item.get('name')}` | {_fmt_num(item.get('train_val_stability_score'))} | "
            f"{_fmt_pct(item_train.get('total_return'))} | {_fmt_pct(item_val.get('total_return'))} | "
            f"{_fmt_pct(item_oos.get('total_return'))} | {_fmt_pct(item_oos.get('max_drawdown'))} | "
            f"{_fmt_num(item_oos.get('return_mdd'))} | {_fmt_num(item_oos.get('sharpe'))} | "
            f"{_fmt_num(item_oos.get('sortino'))} | {_fmt_num(item_oos.get('smart_sortino'))} | "
            f"{_fmt_num(item_oos.get('calmar'))} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-portfolio-json", default=str(DEFAULT_CANDIDATE_PORTFOLIO_JSON))
    parser.add_argument("--liquidation-json", default=str(DEFAULT_LIQUIDATION_JSON))
    parser.add_argument("--market-root", default=str(DEFAULT_MARKET_ROOT))
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--oos-end-date", default="")
    parser.add_argument("--max-candidate-rows", type=int, default=22)
    parser.add_argument("--max-active-inputs", type=int, default=12)
    parser.add_argument("--default-candidate-count", type=int, default=3)
    parser.add_argument("--report-top-n", type=int, default=12)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name=RUN_NAME,
        output_dir=output_dir,
        input_path=args.candidate_portfolio_json,
        metadata={
            "script": Path(__file__).name,
            "max_candidate_rows": int(args.max_candidate_rows),
            "max_active_inputs": int(args.max_active_inputs),
            "locked_oos_policy": "report_only_gate_only",
        },
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    finalized = False
    try:
        memory_guard.checkpoint(
            "start",
            {
                "candidate_portfolio_json": str(args.candidate_portfolio_json),
                "liquidation_json": str(args.liquidation_json),
            },
        )
        payload = build_payload(args)
        memory_summary = memory_guard.finalize(
            status="completed",
            context={
                "selected": str(dict(payload.get("selected_by_train_validation") or {}).get("name") or ""),
                "full_tuning_result_count": int(payload.get("full_tuning_result_count") or 0),
            },
        )
        finalized = True
        memory_summary["summary_path"] = str(memory_guard.summary_path)
        payload["rss_log_path"] = str(memory_guard.rss_log_path)
        payload["memory_summary_path"] = str(memory_guard.summary_path)
        payload["memory_summary"] = memory_summary
        payload["peak_rss_mib"] = max(
            _safe_float(payload.get("peak_rss_mib")),
            _safe_float(memory_summary.get("peak_rss_bytes")) / (1024.0 * 1024.0),
        )
        json_path = output_dir / "candidate_hybrid_latest.json"
        md_path = output_dir / "candidate_hybrid_latest.md"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        timestamped_json = output_dir / f"candidate_hybrid_{timestamp}.json"
        timestamped_md = output_dir / f"candidate_hybrid_{timestamp}.md"
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        json_path.write_text(text, encoding="utf-8")
        timestamped_json.write_text(text, encoding="utf-8")
        markdown = _markdown(payload)
        md_path.write_text(markdown, encoding="utf-8")
        timestamped_md.write_text(markdown, encoding="utf-8")
    except Exception as exc:
        if not finalized:
            memory_guard.finalize(status="failed", error=str(exc), context={"script": Path(__file__).name})
        raise
    finally:
        memory_guard.release()
    print(
        json.dumps(
            {
                "json": str(output_dir / "candidate_hybrid_latest.json"),
                "markdown": str(output_dir / "candidate_hybrid_latest.md"),
                "peak_rss_mib": payload["peak_rss_mib"],
                "status": "completed",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
