from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_causal_overlay_portfolio.py"
SPEC = importlib.util.spec_from_file_location("run_causal_overlay_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load run_causal_overlay_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_backbone_weights_reads_candidate_ids(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.json"
    path.write_text(
        '{"weights":[{"candidate_id":"a","weight":0.35},{"candidate_id":"b","weight":0.65}]}',
        encoding="utf-8",
    )
    weights = MODULE._load_backbone_weights(path)
    assert weights == {"a": 0.35, "b": 0.65}


def test_performance_scale_zeroes_failed_sleeves() -> None:
    scale = MODULE._performance_scale(
        trailing_sharpe=-1.0,
        trailing_return=-0.01,
        trailing_drawdown=0.20,
        min_trailing_sharpe=0.0,
        min_trailing_return=0.0,
        max_trailing_drawdown=0.15,
        overlay_strength=1.0,
    )
    assert scale == 0.0


def test_positive_corr_penalty_nonnegative() -> None:
    history = {
        "a": MODULE.np.asarray([0.01, 0.02, 0.03]),
        "b": MODULE.np.asarray([0.01, 0.02, 0.03]),
        "c": MODULE.np.asarray([-0.01, -0.02, -0.03]),
    }
    penalty = MODULE._positive_corr_penalty(history, ["a", "b", "c"], "a")
    assert penalty >= 0.0


def test_write_overlay_comparison_adds_scope(tmp_path: Path, monkeypatch) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "comparison_scope": ["equal_weight_diagnostic", "current_one_shot_optimized"],
                "current_one_shot_optimized": {"oos": {"total_return": 0.05, "sharpe": 1.5}},
                "deltas": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "COMPARISON_INPUT", comparison)
    monkeypatch.setattr(MODULE, "FOLLOWUP_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "DEFAULT_OUTPUT_DIR", tmp_path)
    overlay_payload = {
        "split_metrics": {"val": {}, "oos": {"total_return": 0.04, "sharpe": 1.2}},
        "final_allocation": [],
        "best_params": {},
    }
    result = MODULE.write_overlay_comparison(overlay_payload)
    written = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert "causal_overlay_portfolio" in written["comparison_scope"]
