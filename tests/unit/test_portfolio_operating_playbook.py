from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "write_portfolio_operating_playbook.py"
)
SPEC = importlib.util.spec_from_file_location("write_portfolio_operating_playbook", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_playbook_includes_hybrid_guarded_mode() -> None:
    payload = MODULE.build_playbook(
        base_plan={
            "deployment_modes": {
                "core_mode": {"allocation": {"soft_three_way_regime": 1.0}},
                "balanced_overlay_mode": {"allocation": {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}},
            }
        },
        switch_validation={
            "refreshed_metrics": {
                "risk_off_cash": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
                "strategy1_balanced_overlay_80_20": {"total_return": -0.01, "sharpe": -1.0, "max_drawdown": 0.02},
                "switch_strategy_core_soft100": {"total_return": -0.02, "sharpe": -2.0, "max_drawdown": 0.03},
                "aggressive_three_way": {"total_return": -0.03, "sharpe": -3.0, "max_drawdown": 0.04},
            },
            "comparison_switch_vs_strategy1": {"oos_return_delta": 0.01, "oos_sharpe_delta": 1.0, "oos_max_drawdown_delta": -0.01},
        },
        switch_recommendation={
            "current_market_state": {"favored_group": "incumbent", "pair_liquidity_state": "normal"},
            "recommended_mode": {"mode": "hybrid_guarded_mode", "allocation": {"hybrid_online_portfolio": 1.0}},
        },
        bearish_scan={"ranked_by_oos_return_then_sharpe": []},
        hybrid_payload={
            "scenarios": {"refreshed_latest_tail": {"split_metrics": {"oos": {"total_return": 0.02, "sharpe": 2.0, "max_drawdown": 0.01}}}},
            "readiness": {"recommended_stage": "guarded_candidate", "beats_cash_refreshed": True},
        },
    )
    hybrid = payload["deployment_modes"]["hybrid_guarded_mode"]
    assert hybrid["allocation"] == {"hybrid_online_portfolio": 1.0}
    assert hybrid["metrics"]["total_return"] == 0.02
    assert payload["recommended_mode"]["mode"] == "hybrid_guarded_mode"
