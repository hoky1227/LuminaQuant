import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "write_portfolio_live_readiness_decision.py"
)
SPEC = importlib.util.spec_from_file_location("write_portfolio_live_readiness_decision", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_live_readiness_decision_prefers_current_switch_mode(tmp_path: Path) -> None:
    switch_path = tmp_path / "switch.json"
    switch_path.write_text(
        json.dumps(
            {
                "recommended_mode": {"mode": "hybrid_guarded_mode", "allocation": {"hybrid_online_portfolio": 1.0}},
                "rationale": ["older", "Promote the current hybrid mode."],
                "current_market_state": {"favored_group": "mixed", "pair_liquidity_state": "normal"},
            }
        ),
        encoding="utf-8",
    )
    max_perf_path = tmp_path / "max_perf.json"
    max_perf_path.write_text(json.dumps({"winner": {"status": "retained_incumbent"}}), encoding="utf-8")

    payload = MODULE.build_live_readiness_decision(
        switch_path=switch_path,
        max_perf_path=max_perf_path,
    )

    assert payload["decision"] == "selected_live_mode"
    assert payload["selected_mode"] == "hybrid_guarded_mode"
    assert payload["candidate_key"] == "hybrid_guarded_mode"
    assert payload["selection_basis"] == "current_operating_switch"


def test_build_live_readiness_decision_falls_back_to_max_perf_when_switch_missing(tmp_path: Path) -> None:
    max_perf_path = tmp_path / "max_perf.json"
    max_perf_path.write_text(
        json.dumps(
            {
                "selection_basis": "locked_oos_robustness_gates",
                "winner": {
                    "status": "promoted_challenger",
                    "candidate_key": "challenger_55_45",
                    "label": "55/45 challenger",
                    "reason": "Promote the challenger.",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_live_readiness_decision(
        switch_path=tmp_path / "missing_switch.json",
        max_perf_path=max_perf_path,
    )

    assert payload["decision"] == "promote_candidate"
    assert payload["candidate_key"] == "challenger_55_45"
    assert payload["selection_basis"] == "locked_oos_robustness_gates"
