from __future__ import annotations

import inspect
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "write_portfolio_max_performance_decision.py"
SPEC = importlib.util.spec_from_file_location(
    "write_portfolio_max_performance_decision", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load write_portfolio_max_performance_decision module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _call_build_decision(**kwargs):
    params = inspect.signature(MODULE.build_portfolio_max_performance_decision).parameters
    return MODULE.build_portfolio_max_performance_decision(
        **{name: value for name, value in kwargs.items() if name in params}
    )


def _bundle_payload() -> dict[str, object]:
    return {
        "artifact_kind": "portfolio_one_shot_incumbent_bundle",
        "selection_basis": "incumbent_saved_one_shot_weights",
        "candidates": [
            {
                "candidate_id": "inc-a",
                "name": "inc-a",
                "strategy_class": "StubStrategy",
                "strategy_timeframe": "1h",
                "portfolio_weight": 0.5,
            },
            {
                "candidate_id": "inc-b",
                "name": "inc-b",
                "strategy_class": "StubStrategy",
                "strategy_timeframe": "1h",
                "portfolio_weight": 0.5,
            },
        ],
    }


def _portfolio_payload(
    *,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
) -> dict[str, object]:
    return {
        "artifact_kind": "portfolio_optimization",
        "portfolio_metrics": {
            "oos": {
                "total_return": total_return,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
            },
            "val": {
                "total_return": 0.03,
                "sharpe": 1.4,
                "sortino": 1.8,
                "calmar": 3.5,
                "max_drawdown": 0.05,
                "volatility": 0.10,
            },
        },
        "weights": [
            {"candidate_id": "inc-a", "weight": 0.5},
            {"candidate_id": "inc-b", "weight": 0.5},
        ],
    }


def _comparison_section(
    *,
    path: Path,
    total_return: float,
    sharpe: float,
    sortino: float,
    calmar: float,
    max_drawdown: float,
    volatility: float,
) -> dict[str, object]:
    return {
        "path": str(path),
        "val": {"total_return": 0.02, "sharpe": 1.0},
        "oos": {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
        },
        "weights": [{"candidate_id": "c1", "weight": 1.0}],
    }


def _anchored_candidate_entries(payload: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for entry in list(payload.get("candidates") or []):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("candidate_key") or "").lower()
        label = str(entry.get("label") or "").lower()
        if any(token in key or token in label for token in ("anchored", "four_sleeve", "four-sleeve")):
            out.append(entry)
    return out


def test_build_portfolio_max_performance_decision_retains_incumbent_when_no_challenger_clears_threshold(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.70,
                sortino=2.20,
                calmar=4.50,
                max_drawdown=0.06,
                volatility=0.12,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "exact_window_frozen_tuned": _comparison_section(
                    path=tmp_path / "tuned.json",
                    total_return=0.049,
                    sharpe=1.69,
                    sortino=2.10,
                    calmar=4.20,
                    max_drawdown=0.065,
                    volatility=0.13,
                ),
            }
        ),
        encoding="utf-8",
    )
    dynamic.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_dynamic_portfolio": _comparison_section(
                    path=tmp_path / "dynamic.json",
                    total_return=0.048,
                    sharpe=1.55,
                    sortino=2.00,
                    calmar=4.10,
                    max_drawdown=0.07,
                    volatility=0.14,
                ),
            }
        ),
        encoding="utf-8",
    )
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.051,
                    sharpe=1.69,
                    sortino=2.18,
                    calmar=4.45,
                    max_drawdown=0.061,
                    volatility=0.121,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_backbone_triplet_search",
                "val": {"total_return": 0.03, "sharpe": 1.1},
                "oos": {
                    "total_return": 0.0505,
                    "sharpe": 1.71,
                    "sortino": 2.21,
                    "calmar": 4.46,
                    "max_drawdown": 0.0605,
                    "volatility": 0.121,
                },
                "weights": [{"candidate_id": "triplet", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
    )

    assert payload["winner"]["candidate_key"] == "current_one_shot_incumbent"
    assert payload["winner"]["status"] == "retained_incumbent"
    challengers = {entry["candidate_key"]: entry for entry in payload["candidates"][1:]}
    assert challengers["causal_overlay_portfolio"]["promotable"] is False
    assert "No challenger cleared" in payload["winner"]["reason"]


def test_build_portfolio_max_performance_decision_promotes_clear_improver(tmp_path: Path) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "exact_window_frozen_tuned": _comparison_section(
                    path=tmp_path / "tuned.json",
                    total_return=0.045,
                    sharpe=1.3,
                    sortino=1.8,
                    calmar=3.8,
                    max_drawdown=0.08,
                    volatility=0.16,
                ),
            }
        ),
        encoding="utf-8",
    )
    dynamic.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_dynamic_portfolio": _comparison_section(
                    path=tmp_path / "dynamic.json",
                    total_return=0.048,
                    sharpe=1.4,
                    sortino=1.9,
                    calmar=3.9,
                    max_drawdown=0.075,
                    volatility=0.15,
                ),
            }
        ),
        encoding="utf-8",
    )
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.072,
                    sharpe=1.95,
                    sortino=2.80,
                    calmar=5.20,
                    max_drawdown=0.055,
                    volatility=0.12,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_backbone_triplet_search",
                "val": {"total_return": 0.03, "sharpe": 1.0},
                "oos": {
                    "total_return": 0.049,
                    "sharpe": 1.45,
                    "sortino": 2.0,
                    "calmar": 4.1,
                    "max_drawdown": 0.069,
                    "volatility": 0.139,
                },
                "weights": [{"candidate_id": "triplet", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
    )

    assert payload["winner"]["candidate_key"] == "causal_overlay_portfolio"
    assert payload["winner"]["status"] == "promoted_challenger"
    overlay_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "causal_overlay_portfolio"
    )
    assert overlay_entry["promotable"] is True
    assert overlay_entry["promotion_score_delta"] > 0.10
    assert overlay_entry["oos_total_return_delta"] > 0.0


def test_build_portfolio_max_performance_decision_requires_positive_return_even_with_drawdown_relief(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.049,
                    sharpe=2.10,
                    sortino=3.10,
                    calmar=5.50,
                    max_drawdown=0.05,
                    volatility=0.11,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8")

    payload = MODULE.build_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
    )

    overlay_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "causal_overlay_portfolio"
    )
    assert overlay_entry["promotion_score_delta"] > 0.25
    assert overlay_entry["oos_max_drawdown_delta"] < 0.0
    assert overlay_entry["oos_total_return_delta"] < 0.0
    assert overlay_entry["promotable"] is False


def test_build_portfolio_max_performance_decision_includes_anchored_four_sleeve_candidate(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    anchored = tmp_path / "portfolio_four_sleeve_comparison_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.60,
                sortino=2.10,
                calmar=4.20,
                max_drawdown=0.065,
                volatility=0.13,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8")

    anchored_section = _comparison_section(
        path=tmp_path / "anchored_tuned.json",
        total_return=0.049,
        sharpe=1.55,
        sortino=2.00,
        calmar=4.00,
        max_drawdown=0.067,
        volatility=0.132,
    )
    anchored.write_text(
        json.dumps(
            {
                "selection_basis": "incumbent_anchor_rolling_gate",
                "rolling_gate": {
                    "selection_basis": "train_val_only",
                    "survives_train_val": True,
                    "survives": True,
                },
                "portfolio_four_sleeve_tuned": anchored_section,
                "portfolio_four_sleeve_anchored_tuned": anchored_section,
                "anchored_four_sleeve_tuned": anchored_section,
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
        anchored_comparison_path=anchored,
        anchored_tuned_comparison_path=anchored,
        portfolio_four_sleeve_comparison_path=anchored,
        four_sleeve_comparison_path=anchored,
    )

    anchored_entries = _anchored_candidate_entries(payload)
    assert anchored_entries, "expected anchored four-sleeve candidate to be included"
    assert payload["winner"]["candidate_key"] == "current_one_shot_incumbent"
    assert anchored_entries[0]["promotable"] is False


def test_build_portfolio_max_performance_decision_allows_custom_anchored_metadata(
    tmp_path: Path,
) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    anchored = tmp_path / "portfolio_four_sleeve_comparison_latest.json"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    dynamic.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    overlay.write_text(json.dumps({"selection_basis": "validation_only"}), encoding="utf-8")
    triplet.write_text(json.dumps({"artifact_kind": "portfolio_backbone_triplet_search"}), encoding="utf-8")

    anchored_section = _comparison_section(
        path=tmp_path / "tradecount_probe.json",
        total_return=0.08,
        sharpe=2.30,
        sortino=3.20,
        calmar=6.10,
        max_drawdown=0.04,
        volatility=0.10,
    )
    anchored_section["notes"] = ["section-level override should also be accepted"]
    anchored.write_text(
        json.dumps(
            {
                "selection_basis": "ignored_by_custom_metadata",
                "challenger_metadata": {
                    "candidate_key": "autonomous_tradecount_pair_topcap_probe",
                    "label": "Autonomous tradecount pair+topcap probe",
                    "source_artifact_kind": "autonomous_probe.tradecount_pair_topcap",
                    "selection_basis": "autonomous_tradecount_screen_pair_topcap",
                    "notes": ["Custom autonomous probe challenger."],
                },
                "anchored_four_sleeve_tuned": anchored_section,
            }
        ),
        encoding="utf-8",
    )

    payload = _call_build_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
        anchored_comparison_path=anchored,
        anchored_tuned_comparison_path=anchored,
        portfolio_four_sleeve_comparison_path=anchored,
        four_sleeve_comparison_path=anchored,
    )

    probe_entry = next(
        entry
        for entry in payload["candidates"]
        if entry["candidate_key"] == "autonomous_tradecount_pair_topcap_probe"
    )
    assert probe_entry["label"] == "Autonomous tradecount pair+topcap probe"
    assert probe_entry["source_artifact_kind"] == "autonomous_probe.tradecount_pair_topcap"
    assert probe_entry["selection_basis"] == "autonomous_tradecount_screen_pair_topcap"
    assert "Custom autonomous probe challenger." in probe_entry["notes"]
    assert payload["winner"]["candidate_key"] == "autonomous_tradecount_pair_topcap_probe"
    assert payload["winner"]["status"] == "promoted_challenger"


def test_write_portfolio_max_performance_decision_writes_latest_files(tmp_path: Path) -> None:
    incumbent_bundle = tmp_path / "incumbent_bundle.json"
    incumbent_portfolio = tmp_path / "incumbent_portfolio.json"
    tuned = tmp_path / "portfolio_comparison_latest.json"
    dynamic = tmp_path / "portfolio_dynamic_comparison_latest.json"
    overlay = tmp_path / "portfolio_overlay_comparison_latest.json"
    triplet = tmp_path / "portfolio_backbone_triplet_search_latest.json"
    output_json = tmp_path / "portfolio_max_performance_decision_latest.json"
    output_md = tmp_path / "portfolio_max_performance_decision_latest.md"

    incumbent_bundle.write_text(json.dumps(_bundle_payload()), encoding="utf-8")
    incumbent_portfolio.write_text(
        json.dumps(
            _portfolio_payload(
                total_return=0.05,
                sharpe=1.50,
                sortino=2.00,
                calmar=4.00,
                max_drawdown=0.07,
                volatility=0.14,
            )
        ),
        encoding="utf-8",
    )
    tuned.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "exact_window_frozen_tuned": _comparison_section(
                    path=tmp_path / "tuned.json",
                    total_return=0.045,
                    sharpe=1.3,
                    sortino=1.8,
                    calmar=3.8,
                    max_drawdown=0.08,
                    volatility=0.16,
                ),
            }
        ),
        encoding="utf-8",
    )
    dynamic.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_dynamic_portfolio": _comparison_section(
                    path=tmp_path / "dynamic.json",
                    total_return=0.048,
                    sharpe=1.4,
                    sortino=1.9,
                    calmar=3.9,
                    max_drawdown=0.075,
                    volatility=0.15,
                ),
            }
        ),
        encoding="utf-8",
    )
    overlay.write_text(
        json.dumps(
            {
                "selection_basis": "validation_only",
                "causal_overlay_portfolio": _comparison_section(
                    path=tmp_path / "overlay.json",
                    total_return=0.072,
                    sharpe=1.95,
                    sortino=2.80,
                    calmar=5.20,
                    max_drawdown=0.055,
                    volatility=0.12,
                ),
            }
        ),
        encoding="utf-8",
    )
    triplet.write_text(
        json.dumps(
            {
                "artifact_kind": "portfolio_backbone_triplet_search",
                "val": {"total_return": 0.03, "sharpe": 1.0},
                "oos": {
                    "total_return": 0.049,
                    "sharpe": 1.45,
                    "sortino": 2.0,
                    "calmar": 4.1,
                    "max_drawdown": 0.069,
                    "volatility": 0.139,
                },
                "weights": [{"candidate_id": "triplet", "weight": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.write_portfolio_max_performance_decision(
        incumbent_bundle_path=incumbent_bundle,
        incumbent_portfolio_path=incumbent_portfolio,
        tuned_comparison_path=tuned,
        dynamic_comparison_path=dynamic,
        overlay_comparison_path=overlay,
        backbone_triplet_path=triplet,
        output_json_path=output_json,
        output_md_path=output_md,
    )

    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["winner"]["candidate_key"] == "causal_overlay_portfolio"
    assert "portfolio max-performance decision" in output_md.read_text(encoding="utf-8")
