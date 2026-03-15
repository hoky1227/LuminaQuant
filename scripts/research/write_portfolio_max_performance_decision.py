"""Write the canonical max-performance promotion decision artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_ROOT = ROOT / "var" / "reports" / "exact_window_backtests" / "followup_status"
DEFAULT_INCUMBENT_BUNDLE = FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.json"
DEFAULT_INCUMBENT_PORTFOLIO = (
    FOLLOWUP_ROOT / "portfolio_one_shot_current_opt" / "portfolio_optimization_latest.json"
)
DEFAULT_TUNED_COMPARISON = FOLLOWUP_ROOT / "portfolio_comparison_latest.json"
DEFAULT_DYNAMIC_COMPARISON = FOLLOWUP_ROOT / "portfolio_dynamic_comparison_latest.json"
DEFAULT_OVERLAY_COMPARISON = FOLLOWUP_ROOT / "portfolio_overlay_comparison_latest.json"
DEFAULT_BACKBONE_TRIPLET = FOLLOWUP_ROOT / "portfolio_backbone_triplet_search_latest.json"
DEFAULT_ANCHORED_COMPARISON = FOLLOWUP_ROOT / "portfolio_four_sleeve_comparison_latest.json"
DEFAULT_OUTPUT_JSON = FOLLOWUP_ROOT / "portfolio_max_performance_decision_latest.json"
DEFAULT_OUTPUT_MD = FOLLOWUP_ROOT / "portfolio_max_performance_decision_latest.md"
SPLIT_CONTRACT_PATH = ROOT / "src" / "lumina_quant" / "portfolio_split_contract.py"

_DEFAULT_SPLIT_CONTRACT = {
    "train_start": "2025-01-01T00:00:00Z",
    "train_end_exclusive": "2026-01-01T00:00:00Z",
    "val_start": "2026-01-01T00:00:00Z",
    "val_end_exclusive": "2026-02-01T00:00:00Z",
    "oos_start": "2026-02-01T00:00:00Z",
}
_REQUIRED_SPLIT_KEYS = tuple(_DEFAULT_SPLIT_CONTRACT.keys())
PROMOTION_FORMULA = (
    "(1.0 * oos_sharpe) + (0.35 * oos_sortino) + (0.10 * oos_calmar) + "
    "(10.0 * oos_total_return) - (4.0 * oos_max_drawdown) - (0.75 * oos_volatility)"
)
PROMOTION_THRESHOLDS = {
    "promotion_score_delta_min": 0.10,
    "promotion_score_delta_with_drawdown_relief_min": 0.25,
    "require_positive_total_return_delta": True,
    "drawdown_relief_required": True,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _maybe_split_payload(module: Any) -> dict[str, Any] | None:
    for attr_name in (
        "split_contract_payload",
        "build_split_contract_payload",
        "get_portfolio_split_contract",
    ):
        attr = getattr(module, attr_name, None)
        if callable(attr):
            candidate = attr()
            if isinstance(candidate, dict):
                return dict(candidate)
    for attr_name in ("SPLIT_CONTRACT", "PORTFOLIO_SPLIT_CONTRACT"):
        candidate = getattr(module, attr_name, None)
        if isinstance(candidate, dict):
            return dict(candidate)
    if all(hasattr(module, key.upper()) for key in _REQUIRED_SPLIT_KEYS):
        return {key: getattr(module, key.upper()) for key in _REQUIRED_SPLIT_KEYS}
    return None


def _resolve_split_contract() -> dict[str, Any]:
    payload = {**_DEFAULT_SPLIT_CONTRACT, "source": "local_default_constants"}
    if not SPLIT_CONTRACT_PATH.exists():
        return payload

    spec = importlib.util.spec_from_file_location("portfolio_split_contract", SPLIT_CONTRACT_PATH)
    if spec is None or spec.loader is None:
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload
    candidate = _maybe_split_payload(module)
    if not isinstance(candidate, dict):
        payload["source"] = str(SPLIT_CONTRACT_PATH.resolve())
        return payload

    resolved = dict(payload)
    for key in _REQUIRED_SPLIT_KEYS:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = value.strip()
    resolved["source"] = str(SPLIT_CONTRACT_PATH.resolve())
    return resolved


def _extract_split_metrics(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    for key in ("portfolio_metrics", "split_metrics", "metrics"):
        metric_block = payload.get(key)
        if not isinstance(metric_block, dict):
            continue
        for split in ("train", "val", "oos"):
            split_payload = metric_block.get(split)
            if isinstance(split_payload, dict):
                extracted[split] = dict(split_payload)
    for split in ("train", "val", "oos"):
        if split not in extracted and isinstance(payload.get(split), dict):
            extracted[split] = dict(payload.get(split) or {})
    return extracted


def _extract_weights(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("weights", "final_allocation"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _promotion_score(metrics: dict[str, Any]) -> float:
    return (
        (1.0 * _safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * _safe_float(metrics.get("sortino"), 0.0))
        + (0.10 * _safe_float(metrics.get("calmar"), 0.0))
        + (10.0 * _safe_float(metrics.get("total_return", metrics.get("return")), 0.0))
        - (4.0 * _safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0))
        - (0.75 * _safe_float(metrics.get("volatility"), 0.0))
    )


def _artifact_entry(
    *,
    candidate_key: str,
    label: str,
    artifact_path: Path,
    payload: dict[str, Any],
    source_artifact_kind: str,
    selection_basis: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    split_metrics = _extract_split_metrics(payload)
    return {
        "candidate_key": candidate_key,
        "label": label,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_kind": str(payload.get("artifact_kind") or source_artifact_kind),
        "source_artifact_kind": source_artifact_kind,
        "selection_basis": selection_basis,
        "train": dict(split_metrics.get("train") or {}),
        "val": dict(split_metrics.get("val") or {}),
        "oos": dict(split_metrics.get("oos") or {}),
        "weights": _extract_weights(payload),
        "notes": list(notes or []),
    }


def _normalized_notes(value: Any) -> list[str]:
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _challenger_reason(entry: dict[str, Any]) -> str:
    if entry.get("promotable"):
        if _safe_float(entry.get("oos_total_return_delta"), 0.0) > 0.0:
            return "Promotable because promotion score and OOS total return both improved over the incumbent."
        return "Promotable because promotion score improved materially while OOS max drawdown declined."

    reasons: list[str] = []
    if (
        _safe_float(entry.get("promotion_score_delta"), 0.0)
        <= PROMOTION_THRESHOLDS["promotion_score_delta_min"]
    ):
        reasons.append("promotion score improvement did not clear the base threshold")
    if _safe_float(entry.get("oos_total_return_delta"), 0.0) <= 0.0:
        reasons.append("OOS total return did not improve")
    if _safe_float(entry.get("oos_max_drawdown_delta"), 0.0) >= 0.0:
        reasons.append("OOS max drawdown did not improve")
    if not reasons:
        reasons.append("incumbent tie policy kept the current one-shot leader")
    return "; ".join(reasons)


def _decorate_vs_incumbent(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not entries:
        raise RuntimeError("no candidate entries available for decision bundle")

    incumbent = dict(entries[0])
    incumbent_oos = dict(incumbent.get("oos") or {})
    incumbent_score = _promotion_score(incumbent_oos)
    incumbent["promotion_score"] = incumbent_score
    incumbent["promotion_score_delta"] = 0.0
    incumbent["oos_total_return_delta"] = 0.0
    incumbent["oos_max_drawdown_delta"] = 0.0
    incumbent["oos_sharpe_delta"] = 0.0
    incumbent["promotable"] = False
    incumbent["decision_reason"] = "Current one-shot incumbent baseline."

    decorated = [incumbent]
    for raw in entries[1:]:
        entry = dict(raw)
        oos = dict(entry.get("oos") or {})
        score = _promotion_score(oos)
        score_delta = score - incumbent_score
        total_return_delta = _safe_float(
            oos.get("total_return", oos.get("return")), 0.0
        ) - _safe_float(
            incumbent_oos.get("total_return", incumbent_oos.get("return")),
            0.0,
        )
        max_drawdown_delta = _safe_float(
            oos.get("max_drawdown", oos.get("mdd")), 0.0
        ) - _safe_float(
            incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")),
            0.0,
        )
        sharpe_delta = _safe_float(oos.get("sharpe"), 0.0) - _safe_float(
            incumbent_oos.get("sharpe"), 0.0
        )
        promotable = (
            score_delta > PROMOTION_THRESHOLDS["promotion_score_delta_min"]
            and total_return_delta > 0.0
        ) or (
            score_delta > PROMOTION_THRESHOLDS["promotion_score_delta_with_drawdown_relief_min"]
            and total_return_delta > 0.0
            and max_drawdown_delta < 0.0
        )
        entry["promotion_score"] = score
        entry["promotion_score_delta"] = score_delta
        entry["oos_total_return_delta"] = total_return_delta
        entry["oos_max_drawdown_delta"] = max_drawdown_delta
        entry["oos_sharpe_delta"] = sharpe_delta
        entry["promotable"] = promotable
        entry["decision_reason"] = _challenger_reason(entry)
        decorated.append(entry)
    return decorated, incumbent


def build_portfolio_max_performance_decision(
    *,
    incumbent_bundle_path: Path | str = DEFAULT_INCUMBENT_BUNDLE,
    incumbent_portfolio_path: Path | str = DEFAULT_INCUMBENT_PORTFOLIO,
    tuned_comparison_path: Path | str = DEFAULT_TUNED_COMPARISON,
    dynamic_comparison_path: Path | str = DEFAULT_DYNAMIC_COMPARISON,
    overlay_comparison_path: Path | str = DEFAULT_OVERLAY_COMPARISON,
    backbone_triplet_path: Path | str = DEFAULT_BACKBONE_TRIPLET,
    anchored_comparison_path: Path | str = DEFAULT_ANCHORED_COMPARISON,
    anchored_tuned_comparison_path: Path | str | None = None,
    portfolio_four_sleeve_comparison_path: Path | str | None = None,
    four_sleeve_comparison_path: Path | str | None = None,
) -> dict[str, Any]:
    incumbent_bundle = _load_json(Path(incumbent_bundle_path))
    incumbent_portfolio = _load_json(Path(incumbent_portfolio_path))
    split_contract = _resolve_split_contract()

    entries = [
        _artifact_entry(
            candidate_key="current_one_shot_incumbent",
            label="Current one-shot incumbent",
            artifact_path=Path(incumbent_portfolio_path),
            payload=incumbent_portfolio,
            source_artifact_kind="portfolio_one_shot_current_opt",
            selection_basis=str(
                incumbent_bundle.get("selection_basis") or "incumbent_saved_one_shot_weights"
            ),
            notes=[
                "Baseline incumbent for the max-performance loop.",
                f"Restricted sleeve bundle: {Path(incumbent_bundle_path).resolve()}",
            ],
        )
    ]
    supporting_artifacts = {
        "incumbent_bundle": str(Path(incumbent_bundle_path).resolve()),
        "incumbent_portfolio": str(Path(incumbent_portfolio_path).resolve()),
    }
    missing_artifacts: list[str] = []

    tuned_path = Path(tuned_comparison_path)
    if tuned_path.exists():
        tuned_comparison = _load_json(tuned_path)
        supporting_artifacts["portfolio_comparison"] = str(tuned_path.resolve())
        tuned_section = dict(tuned_comparison.get("exact_window_frozen_tuned") or {})
        if tuned_section:
            entries.append(
                _artifact_entry(
                    candidate_key="exact_window_frozen_tuned",
                    label="Exact-window frozen tuned challenger",
                    artifact_path=Path(tuned_section.get("path") or tuned_path),
                    payload=tuned_section,
                    source_artifact_kind="portfolio_comparison.exact_window_frozen_tuned",
                    selection_basis=str(
                        tuned_comparison.get("selection_basis") or "validation_only"
                    ),
                    notes=[
                        "Validation-ranked frozen challenger from the exact-window sleeve freeze path."
                    ],
                )
            )
    else:
        missing_artifacts.append(str(tuned_path.resolve()))

    triplet_path = Path(backbone_triplet_path)
    if triplet_path.exists():
        triplet_payload = _load_json(triplet_path)
        supporting_artifacts["portfolio_backbone_triplet_search"] = str(triplet_path.resolve())
        entries.append(
            _artifact_entry(
                candidate_key="backbone_preserving_triplet_search",
                label="Backbone-preserving triplet search challenger",
                artifact_path=triplet_path,
                payload=triplet_payload,
                source_artifact_kind="portfolio_backbone_triplet_search",
                selection_basis="validation_objective_then_locked_oos",
                notes=["Same-family backbone-preserving challenger."],
            )
        )
    else:
        missing_artifacts.append(str(triplet_path.resolve()))

    dynamic_path = Path(dynamic_comparison_path)
    if dynamic_path.exists():
        dynamic_comparison = _load_json(dynamic_path)
        supporting_artifacts["portfolio_dynamic_comparison"] = str(dynamic_path.resolve())
        dynamic_section = dict(dynamic_comparison.get("causal_dynamic_portfolio") or {})
        if dynamic_section:
            entries.append(
                _artifact_entry(
                    candidate_key="causal_dynamic_portfolio",
                    label="Causal dynamic challenger",
                    artifact_path=Path(dynamic_section.get("path") or dynamic_path),
                    payload=dynamic_section,
                    source_artifact_kind="portfolio_dynamic_comparison.causal_dynamic_portfolio",
                    selection_basis=str(
                        dynamic_comparison.get("selection_basis") or "validation_only"
                    ),
                    notes=["Dynamic allocation challenger evaluated after incumbent-first lanes."],
                )
            )
    else:
        missing_artifacts.append(str(dynamic_path.resolve()))

    overlay_path = Path(overlay_comparison_path)
    if overlay_path.exists():
        overlay_comparison = _load_json(overlay_path)
        supporting_artifacts["portfolio_overlay_comparison"] = str(overlay_path.resolve())
        overlay_section = dict(overlay_comparison.get("causal_overlay_portfolio") or {})
        if overlay_section:
            entries.append(
                _artifact_entry(
                    candidate_key="causal_overlay_portfolio",
                    label="Causal overlay challenger",
                    artifact_path=Path(overlay_section.get("path") or overlay_path),
                    payload=overlay_section,
                    source_artifact_kind="portfolio_overlay_comparison.causal_overlay_portfolio",
                    selection_basis=str(
                        overlay_comparison.get("selection_basis") or "validation_only"
                    ),
                    notes=["Light overlay challenger evaluated on the incumbent backbone."],
                )
            )
    else:
        missing_artifacts.append(str(overlay_path.resolve()))

    anchored_candidates = [
        anchored_comparison_path,
        anchored_tuned_comparison_path,
        portfolio_four_sleeve_comparison_path,
        four_sleeve_comparison_path,
    ]
    anchored_path = next(
        (Path(candidate) for candidate in anchored_candidates if candidate and Path(candidate).exists()),
        None,
    )
    if anchored_path is not None:
        anchored_comparison = _load_json(anchored_path)
        supporting_artifacts["portfolio_four_sleeve_comparison"] = str(anchored_path.resolve())
        anchored_section = dict(
            anchored_comparison.get("anchored_four_sleeve_tuned")
            or anchored_comparison.get("portfolio_four_sleeve_anchored_tuned")
            or anchored_comparison.get("portfolio_four_sleeve_tuned")
            or {}
        )
        if anchored_section:
            rolling_gate = dict(anchored_comparison.get("rolling_gate") or {})
            metadata = dict(anchored_comparison.get("challenger_metadata") or {})
            for key in ("candidate_key", "label", "source_artifact_kind", "selection_basis", "notes"):
                if key not in metadata and anchored_section.get(key) is not None:
                    metadata[key] = anchored_section.get(key)
            candidate_key = str(metadata.get("candidate_key") or "anchored_four_sleeve_tuned")
            label = str(metadata.get("label") or "Anchored four-sleeve tuned challenger")
            source_artifact_kind = str(
                metadata.get("source_artifact_kind")
                or "portfolio_four_sleeve_comparison.anchored_four_sleeve_tuned"
            )
            selection_basis = str(
                metadata.get("selection_basis")
                or anchored_comparison.get("selection_basis")
                or "incumbent_anchor_rolling_gate"
            )
            notes = _normalized_notes(metadata.get("notes"))
            if not notes:
                notes = ["Incumbent-aware four-sleeve challenger with RollingBreakout gate."]
            if rolling_gate:
                notes.append(
                    "Rolling gate: "
                    f"selection_basis={rolling_gate.get('selection_basis')} "
                    f"survives_train_val={rolling_gate.get('survives_train_val')}"
                )
            entries.append(
                _artifact_entry(
                    candidate_key=candidate_key,
                    label=label,
                    artifact_path=Path(anchored_section.get("path") or anchored_path),
                    payload=anchored_section,
                    source_artifact_kind=source_artifact_kind,
                    selection_basis=selection_basis,
                    notes=notes,
                )
            )
    else:
        missing_artifacts.append(str(Path(anchored_comparison_path).resolve()))

    decorated, incumbent = _decorate_vs_incumbent(entries)
    challengers = [dict(entry) for entry in decorated[1:]]
    promotable = [entry for entry in challengers if bool(entry.get("promotable"))]
    if promotable:
        winner = max(
            promotable,
            key=lambda entry: (
                _safe_float(entry.get("promotion_score_delta"), 0.0),
                _safe_float(entry.get("oos_total_return_delta"), 0.0),
                -_safe_float(entry.get("oos_max_drawdown_delta"), 0.0),
            ),
        )
        winner_status = "promoted_challenger"
        winner_reason = str(winner.get("decision_reason") or "")
        recommended_action = (
            "Promote the challenger, then run cleanup and regression hardening on the new winner."
        )
    else:
        winner = incumbent
        winner_status = "retained_incumbent"
        winner_reason = "No challenger cleared the locked-OOS promotion rule; keep the current one-shot incumbent."
        recommended_action = "Retain the incumbent and continue with cleanup, evidence collation, and reproducibility hardening."

    challenger_rankings = [
        {
            "candidate_key": entry.get("candidate_key"),
            "label": entry.get("label"),
            "promotion_score": entry.get("promotion_score"),
            "promotion_score_delta": entry.get("promotion_score_delta"),
            "promotable": entry.get("promotable"),
        }
        for entry in sorted(
            challengers,
            key=lambda item: _safe_float(item.get("promotion_score_delta"), 0.0),
            reverse=True,
        )
    ]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "portfolio_max_performance_decision",
        "schema_version": "1.0",
        "selection_basis": "locked_oos_promotion_score",
        "split_contract": split_contract,
        "promotion_formula": PROMOTION_FORMULA,
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "comparison_scope": [entry.get("candidate_key") for entry in decorated],
        "supporting_artifacts": supporting_artifacts,
        "missing_artifacts": missing_artifacts,
        "incumbent_bundle_path": str(Path(incumbent_bundle_path).resolve()),
        "incumbent_components": [
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "timeframe": row.get("timeframe") or row.get("strategy_timeframe"),
                "weight": _safe_float(
                    row.get("portfolio_weight", row.get("_portfolio_weight")), 0.0
                ),
            }
            for row in list(incumbent_bundle.get("candidates") or [])
            if isinstance(row, dict)
        ],
        "candidates": decorated,
        "challenger_rankings": challenger_rankings,
        "winner": {
            "candidate_key": winner.get("candidate_key"),
            "label": winner.get("label"),
            "status": winner_status,
            "promotion_score": winner.get("promotion_score"),
            "promotion_score_delta": winner.get("promotion_score_delta"),
            "reason": winner_reason,
        },
        "recommended_action": recommended_action,
        "notes": [
            "Promotion uses locked-OOS metrics only after each challenger artifact is frozen.",
            f"Locked OOS starts at {split_contract['oos_start']} and remains excluded from tuning decisions.",
            "If thresholds are not clearly exceeded, incumbent tie policy keeps the current one-shot leader.",
        ],
    }
    return payload


def write_portfolio_max_performance_decision(
    *,
    incumbent_bundle_path: Path | str = DEFAULT_INCUMBENT_BUNDLE,
    incumbent_portfolio_path: Path | str = DEFAULT_INCUMBENT_PORTFOLIO,
    tuned_comparison_path: Path | str = DEFAULT_TUNED_COMPARISON,
    dynamic_comparison_path: Path | str = DEFAULT_DYNAMIC_COMPARISON,
    overlay_comparison_path: Path | str = DEFAULT_OVERLAY_COMPARISON,
    backbone_triplet_path: Path | str = DEFAULT_BACKBONE_TRIPLET,
    anchored_comparison_path: Path | str = DEFAULT_ANCHORED_COMPARISON,
    anchored_tuned_comparison_path: Path | str | None = None,
    portfolio_four_sleeve_comparison_path: Path | str | None = None,
    four_sleeve_comparison_path: Path | str | None = None,
    output_json_path: Path | str = DEFAULT_OUTPUT_JSON,
    output_md_path: Path | str = DEFAULT_OUTPUT_MD,
) -> dict[str, Any]:
    payload = build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle_path,
        incumbent_portfolio_path=incumbent_portfolio_path,
        tuned_comparison_path=tuned_comparison_path,
        dynamic_comparison_path=dynamic_comparison_path,
        overlay_comparison_path=overlay_comparison_path,
        backbone_triplet_path=backbone_triplet_path,
        anchored_comparison_path=anchored_comparison_path,
        anchored_tuned_comparison_path=anchored_tuned_comparison_path,
        portfolio_four_sleeve_comparison_path=portfolio_four_sleeve_comparison_path,
        four_sleeve_comparison_path=four_sleeve_comparison_path,
    )
    json_path = Path(output_json_path)
    md_path = Path(output_md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    winner = dict(payload.get("winner") or {})
    lines = [
        "# portfolio max-performance decision",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- winner: `{winner.get('label')}` ({winner.get('status')})",
        f"- winner_reason: {winner.get('reason')}",
        f"- oos_start: `{payload['split_contract']['oos_start']}`",
        "",
        "## candidate summary",
    ]
    for entry in list(payload.get("candidates") or []):
        oos = dict(entry.get("oos") or {})
        lines.append(
            f"- `{entry.get('label')}` | score={_safe_float(entry.get('promotion_score'), 0.0):.4f} | "
            f"delta={_safe_float(entry.get('promotion_score_delta'), 0.0):.4f} | "
            f"oos_return={_safe_float(oos.get('total_return', oos.get('return')), 0.0):.4%} | "
            f"oos_sharpe={_safe_float(oos.get('sharpe'), 0.0):.3f} | "
            f"oos_max_dd={_safe_float(oos.get('max_drawdown', oos.get('mdd')), 0.0):.4%} | "
            f"promotable={entry.get('promotable')}"
        )
        lines.append(f"  - {entry.get('decision_reason')}")
    lines.extend(["", "## notes"])
    for note in list(payload.get("notes") or []):
        lines.append(f"- {note}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the portfolio max-performance decision artifact."
    )
    parser.add_argument("--incumbent-bundle", default=str(DEFAULT_INCUMBENT_BUNDLE))
    parser.add_argument("--incumbent-portfolio", default=str(DEFAULT_INCUMBENT_PORTFOLIO))
    parser.add_argument("--tuned-comparison", default=str(DEFAULT_TUNED_COMPARISON))
    parser.add_argument("--dynamic-comparison", default=str(DEFAULT_DYNAMIC_COMPARISON))
    parser.add_argument("--overlay-comparison", default=str(DEFAULT_OVERLAY_COMPARISON))
    parser.add_argument("--backbone-triplet", default=str(DEFAULT_BACKBONE_TRIPLET))
    parser.add_argument("--anchored-comparison", default=str(DEFAULT_ANCHORED_COMPARISON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = write_portfolio_max_performance_decision(
        incumbent_bundle_path=Path(args.incumbent_bundle).resolve(),
        incumbent_portfolio_path=Path(args.incumbent_portfolio).resolve(),
        tuned_comparison_path=Path(args.tuned_comparison).resolve(),
        dynamic_comparison_path=Path(args.dynamic_comparison).resolve(),
        overlay_comparison_path=Path(args.overlay_comparison).resolve(),
        backbone_triplet_path=Path(args.backbone_triplet).resolve(),
        anchored_comparison_path=Path(args.anchored_comparison).resolve(),
        output_json_path=Path(args.output_json).resolve(),
        output_md_path=Path(args.output_md).resolve(),
    )
    print(result["json_path"])
    print(result["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
