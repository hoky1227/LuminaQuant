"""Curated low-memory tuning pass for the hybrid online portfolio governor."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "run_hybrid_online_portfolio",
    ROOT / "run_hybrid_online_portfolio.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load run_hybrid_online_portfolio")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

OUTPUT_DIR = _MOD.GROUP_ROOT / "portfolio_hybrid_online_tuning_current"


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(_MOD._safe_float(value, default))


def _candidate_configs(*, base: _MOD.HybridOnlineConfig | None = None) -> list[tuple[str, _MOD.HybridOnlineConfig]]:
    base = base or _MOD.HybridOnlineConfig()

    def _cfg(**overrides: Any) -> _MOD.HybridOnlineConfig:
        return _MOD.HybridOnlineConfig(**({**asdict(base), **overrides}))

    variants = [
        ("baseline", base),
        (
            "conservative_cash_bias",
            _cfg(
                lookback_days=28,
                default_boost=0.25,
                sticky_default_bonus=0.20,
                switch_margin=0.15,
                score_temperature=1.10,
                min_positive_score=0.15,
                pair_weight_cap=0.20,
                mixed_health_floor=0.45,
                negative_health_floor=0.10,
            ),
        ),
        (
            "responsive_pair",
            _cfg(
                lookback_days=14,
                default_boost=0.30,
                sticky_default_bonus=0.05,
                switch_margin=0.05,
                score_temperature=0.80,
                min_positive_score=0.00,
                pair_weight_cap=0.30,
                mixed_health_floor=0.60,
                negative_health_floor=0.20,
            ),
        ),
        (
            "balanced_sticky",
            _cfg(
                lookback_days=21,
                default_boost=0.40,
                sticky_default_bonus=0.20,
                switch_margin=0.15,
                score_temperature=1.00,
                min_positive_score=0.05,
                pair_weight_cap=0.25,
                mixed_health_floor=0.55,
                negative_health_floor=0.15,
            ),
        ),
        (
            "strict_health",
            _cfg(
                lookback_days=21,
                default_boost=0.35,
                sticky_default_bonus=0.15,
                switch_margin=0.10,
                score_temperature=1.00,
                min_positive_score=0.20,
                pair_weight_cap=0.20,
                mixed_health_floor=0.40,
                negative_health_floor=0.05,
            ),
        ),
        (
            "soft_health",
            _cfg(
                lookback_days=21,
                default_boost=0.30,
                sticky_default_bonus=0.10,
                switch_margin=0.08,
                score_temperature=0.90,
                min_positive_score=0.05,
                pair_weight_cap=0.30,
                mixed_health_floor=0.70,
                negative_health_floor=0.25,
            ),
        ),
        (
            "long_memory",
            _cfg(
                lookback_days=35,
                default_boost=0.30,
                sticky_default_bonus=0.20,
                switch_margin=0.20,
                score_temperature=1.10,
                min_positive_score=0.10,
                pair_weight_cap=0.20,
                mixed_health_floor=0.50,
                negative_health_floor=0.10,
            ),
        ),
        (
            "high_conviction_switch",
            _cfg(
                lookback_days=21,
                default_boost=0.45,
                sticky_default_bonus=0.05,
                switch_margin=0.20,
                score_temperature=0.85,
                min_positive_score=0.12,
                pair_weight_cap=0.25,
                mixed_health_floor=0.55,
                negative_health_floor=0.15,
            ),
        ),
        (
            "pair_penalty_strict",
            _cfg(
                lookback_days=21,
                default_boost=0.35,
                sticky_default_bonus=0.15,
                switch_margin=0.10,
                score_temperature=1.00,
                min_positive_score=0.08,
                pair_weight_cap=0.20,
                pair_pbo_penalty_scale=3.5,
                pair_sparsity_penalty_scale=3.0,
                mixed_health_floor=0.50,
                negative_health_floor=0.15,
            ),
        ),
        (
            "pair_penalty_relaxed",
            _cfg(
                lookback_days=21,
                default_boost=0.35,
                sticky_default_bonus=0.15,
                switch_margin=0.10,
                score_temperature=1.00,
                min_positive_score=0.05,
                pair_weight_cap=0.30,
                pair_pbo_penalty_scale=1.5,
                pair_sparsity_penalty_scale=1.0,
                mixed_health_floor=0.55,
                negative_health_floor=0.20,
            ),
        ),
    ]
    return variants


def _objective(payload: dict[str, Any]) -> float:
    refreshed = dict(payload["scenarios"]["refreshed_latest_tail"]["split_metrics"])
    historical = dict(payload["scenarios"]["historical_saved_baseline"]["split_metrics"])
    readiness = dict(payload.get("readiness") or {})
    ref_train = dict(refreshed.get("train") or {})
    ref_val = dict(refreshed.get("val") or {})
    ref_oos = dict(refreshed.get("oos") or {})
    hist_oos = dict(historical.get("oos") or {})
    score = 0.0
    score += 240.0 * _safe_float(ref_oos.get("total_return", ref_oos.get("return")), 0.0)
    score += 10.0 * _safe_float(ref_oos.get("sharpe"), 0.0)
    score -= 120.0 * _safe_float(ref_oos.get("max_drawdown", ref_oos.get("mdd")), 0.0)
    score += 60.0 * _safe_float(ref_val.get("total_return", ref_val.get("return")), 0.0)
    score += 8.0 * _safe_float(ref_val.get("sharpe"), 0.0)
    score += 20.0 * _safe_float(ref_train.get("total_return", ref_train.get("return")), 0.0)
    score += 3.0 * _safe_float(ref_train.get("sharpe"), 0.0)
    score += 20.0 * _safe_float(hist_oos.get("total_return", hist_oos.get("return")), 0.0)
    score += 2.0 * _safe_float(hist_oos.get("sharpe"), 0.0)
    if not readiness.get("beats_cash_refreshed"):
        score -= 1000.0
    if not readiness.get("pair_cap_respected"):
        score -= 500.0
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _MOD.add_split_config_arguments(parser)
    parser.add_argument("--warmup-ratio", type=float, default=_MOD.HybridOnlineConfig().warmup_ratio)
    parser.add_argument("--warmup-days", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_config = _MOD.split_config_from_args(args)
    base_config = _MOD.HybridOnlineConfig(
        warmup_ratio=float(args.warmup_ratio),
        warmup_days=None if args.warmup_days is None else int(args.warmup_days),
    )

    historical_active = _MOD._historical_active_rows(split_config=split_config)
    refreshed_active, refreshed_benchmarks = _MOD._refreshed_rows(split_config=split_config)
    refreshed_health_metrics = {row["name"]: dict(row.get("oos") or {}) for row in refreshed_active + refreshed_benchmarks}

    leaderboard: list[dict[str, Any]] = []
    for name, config in _candidate_configs(base=base_config):
        historical_config = _MOD.HybridOnlineConfig(**({**asdict(config), "use_current_health_priors": False}))
        historical_result = _MOD.run_hybrid_online_allocator(
            historical_active,
            config=historical_config,
            refreshed_health_metrics=None,
            split_config=split_config,
        )
        refreshed_result = _MOD.run_hybrid_online_allocator(
            refreshed_active,
            config=config,
            refreshed_health_metrics=refreshed_health_metrics,
            split_config=split_config,
        )
        refreshed_rows = _MOD._comparison_rows(
            hybrid_result=refreshed_result,
            benchmarks=refreshed_benchmarks,
            active_rows=refreshed_active,
        )
        refreshed_by_name = {row["name"]: row for row in refreshed_rows}
        pair_alloc_weights = [
            _safe_float((alloc.get("weights") or {}).get("pair_tactical_mode"), 0.0)
            for alloc in list(refreshed_result.get("allocations") or [])
        ]
        payload = {
            "config_name": name,
            "config": asdict(config),
            "scenarios": {
                "historical_saved_baseline": {"split_metrics": historical_result["split_metrics"]},
                "refreshed_latest_tail": {"split_metrics": refreshed_result["split_metrics"]},
            },
            "readiness": {
                "beats_cash_refreshed": bool(
                    _safe_float(refreshed_by_name["hybrid_online_portfolio"].get("total_return", refreshed_by_name["hybrid_online_portfolio"].get("return")), 0.0)
                    > _safe_float(refreshed_by_name["risk_off_cash"].get("total_return", refreshed_by_name["risk_off_cash"].get("return")), 0.0)
                ),
                "beats_pair_tactical_refreshed": bool(
                    _safe_float(refreshed_by_name["hybrid_online_portfolio"].get("total_return", refreshed_by_name["hybrid_online_portfolio"].get("return")), 0.0)
                    > _safe_float(refreshed_by_name["pair_tactical_mode"].get("total_return", refreshed_by_name["pair_tactical_mode"].get("return")), 0.0)
                ),
                "pair_cap_respected": bool(max(pair_alloc_weights or [0.0]) <= config.pair_weight_cap + 1e-9),
            },
        }
        payload["objective"] = _objective(payload)
        leaderboard.append(payload)

    leaderboard.sort(key=lambda row: float(row["objective"]), reverse=True)
    best = leaderboard[0]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"hybrid_online_tuning_{stamp}.json"
    latest_json = output_dir / "hybrid_online_tuning_latest.json"
    md_path = output_dir / f"hybrid_online_tuning_{stamp}.md"
    payload = {
        "artifact_kind": "hybrid_online_portfolio_tuning",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "split_windows": split_config.as_payload(),
        "warmup_ratio": float(base_config.warmup_ratio),
        "fixed_warmup_days": (
            None
            if base_config.warmup_days is None
            else int(base_config.warmup_days)
        ),
        "resolved_warmup_days": _MOD.resolve_warmup_days(config=base_config, split_config=split_config),
        "leaderboard": leaderboard,
        "best": best,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    json_path.write_text(text, encoding="utf-8")
    latest_json.write_text(text, encoding="utf-8")
    lines = [
        "# Hybrid online portfolio tuning",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- evaluated_configs: `{len(leaderboard)}`",
        "",
        "## Best config",
        "",
        "```json",
        json.dumps(best, indent=2, sort_keys=True),
        "```",
        "",
        "## Leaderboard",
        "",
        "| Config | Objective | Refresh OOS | Refresh Sharpe | Refresh Val | Refresh Train | Beats cash | Beats pair |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in leaderboard:
        ref = row["scenarios"]["refreshed_latest_tail"]["split_metrics"]
        lines.append(
            f"| `{row['config_name']}` | {float(row['objective']):.4f} | {_safe_float(ref['oos'].get('total_return', ref['oos'].get('return')),0.0):+.4%} | {_safe_float(ref['oos'].get('sharpe'),0.0):.4f} | {_safe_float(ref['val'].get('total_return', ref['val'].get('return')),0.0):+.4%} | {_safe_float(ref['train'].get('total_return', ref['train'].get('return')),0.0):+.4%} | `{row['readiness']['beats_cash_refreshed']}` | `{row['readiness']['beats_pair_tactical_refreshed']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(latest_json.resolve()))
    print(str(md_path.resolve()))


if __name__ == "__main__":
    main()
