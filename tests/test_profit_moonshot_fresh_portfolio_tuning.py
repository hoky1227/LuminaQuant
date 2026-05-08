from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "tune_profit_moonshot_fresh_portfolio.py"
SPEC = importlib.util.spec_from_file_location("tune_profit_moonshot_fresh_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load tune_profit_moonshot_fresh_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_pool_can_reserve_family_balanced_slots() -> None:
    rows = [
        {
            "name": "calendar_a",
            "family": "calendar_rotation",
            "train_total_return": "0.02",
            "val_total_return": "0.05",
            "val_sharpe": "2.0",
            "val_max_drawdown": "0.001",
            "val_round_trips": "5",
        },
        {
            "name": "calendar_b",
            "family": "calendar_rotation",
            "train_total_return": "0.02",
            "val_total_return": "0.04",
            "val_sharpe": "2.0",
            "val_max_drawdown": "0.001",
            "val_round_trips": "5",
        },
        {
            "name": "new_family_a",
            "family": "adaptive_trend_fade",
            "train_total_return": "0.01",
            "val_total_return": "0.005",
            "val_sharpe": "1.0",
            "val_max_drawdown": "0.001",
            "val_round_trips": "5",
        },
    ]

    unbalanced = MODULE._candidate_pool(rows, top_n=2)
    balanced = MODULE._candidate_pool(rows, top_n=3, family_quota=1)

    assert [row["name"] for row in unbalanced] == ["calendar_a", "calendar_b"]
    assert {row["family"] for row in balanced[:2]} == {"calendar_rotation", "adaptive_trend_fade"}


class _FakeMemoryGuard:
    def __init__(self, output_dir: Path) -> None:
        memory_dir = output_dir / "_memory_guard"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self.rss_log_path = memory_dir / "profit_moonshot_fresh_portfolio_tuning_rss_latest.jsonl"
        self.summary_path = memory_dir / "profit_moonshot_fresh_portfolio_tuning_memory_latest.json"
        self.checkpoints: list[tuple[str, dict[str, Any] | None]] = []
        self.finalize_calls: list[dict[str, Any]] = []
        self.released = False

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        self.checkpoints.append((event, context))

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "artifact_kind": "portfolio_followup_memory_summary",
            "run_name": MODULE.RUN_NAME,
            "status": status,
            "error": error,
            "context": dict(context or {}),
            "rss_log_path": str(self.rss_log_path),
            "summary_path": str(self.summary_path),
            "peak_rss_bytes": 123456,
            "memory_policy": MODULE.memory_policy_payload(
                budget_bytes=MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
            ),
        }
        self.finalize_calls.append(payload)
        self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def release(self) -> None:
        self.released = True


def _minimal_payload(tmp_path: Path) -> dict[str, Any]:
    split_payload = {
        "metrics": {
            "total_return": 0.01,
            "max_drawdown": 0.001,
            "sharpe": 1.2,
            "sortino": 1.5,
            "volatility": 0.02,
        },
        "round_trips": 7,
        "fills": 14,
        "final_equity": 10_100.0,
    }
    selected = {
        "name": "fresh_portfolio_equal_weight_stub_a__stub_b",
        "mode": "equal_weight",
        "sleeves": ["stub_a", "stub_b"],
        "sleeve_count": 2,
        "splits": {"train": split_payload, "val": split_payload, "oos": split_payload},
        "gates": {
            "train_positive": True,
            "val_positive": True,
            "oos_return_beats_incumbent": True,
            "oos_mdd_beats_shadow": True,
            "oos_sharpe_gt_1": True,
            "oos_trades_not_starved": True,
        },
        "success_candidate": True,
        "validation_score": 1.0,
    }
    return {
        "artifact_kind": "profit_moonshot_fresh_portfolio_tuning",
        "generated_at_utc": "2026-05-07T00:00:00Z",
        "candidate_csv": str(tmp_path / "candidates.csv"),
        "candidate_sleeve_count": 2,
        "portfolio_spec_count": 1,
        "success_candidate_count": 1,
        "selected_by_validation": selected,
        "diagnostic_best_oos": selected,
        "data_metadata": {"source": "unit"},
        "peak_rss_mib": 1.0,
    }


def test_tuning_main_wraps_artifacts_with_memory_guard_and_lockbox_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "fresh_overhaul"
    candidate_csv = tmp_path / "candidates.csv"
    candidate_csv.write_text("name,train_total_return,val_total_return,val_round_trips\n", encoding="utf-8")
    captured: dict[str, Any] = {}
    guards: list[_FakeMemoryGuard] = []

    def _fake_acquire(**kwargs: Any) -> _FakeMemoryGuard:
        captured.update(kwargs)
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(
        MODULE,
        "build_payload",
        lambda args: (
            _minimal_payload(tmp_path),
            [
                {
                    "name": "fresh_portfolio_equal_weight_stub_a__stub_b",
                    "mode": "equal_weight",
                    "sleeve_count": 2,
                    "sleeves": "stub_a,stub_b",
                    "validation_score": 1.0,
                    "success_candidate": True,
                    "failed_gates": "",
                    "train_total_return": 0.01,
                    "train_max_drawdown": 0.001,
                    "train_sharpe": 1.2,
                    "train_sortino": 1.5,
                    "train_volatility": 0.02,
                    "train_round_trips": 7,
                    "val_total_return": 0.01,
                    "val_max_drawdown": 0.001,
                    "val_sharpe": 1.2,
                    "val_sortino": 1.5,
                    "val_volatility": 0.02,
                    "val_round_trips": 7,
                    "oos_total_return": 0.01,
                    "oos_max_drawdown": 0.001,
                    "oos_sharpe": 1.2,
                    "oos_sortino": 1.5,
                    "oos_volatility": 0.02,
                    "oos_round_trips": 7,
                }
            ],
        ),
    )

    assert (
        MODULE.main(
            [
                "--candidate-csv",
                str(candidate_csv),
                "--output-dir",
                str(output_dir),
                "--top-n",
                "2",
                "--max-sleeves",
                "2",
            ]
        )
        == 0
    )

    expected_budget = MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    assert captured["run_name"] == MODULE.RUN_NAME
    assert captured["input_path"] == str(candidate_csv)
    assert captured["budget_bytes"] == expected_budget
    assert captured["metadata"]["locked_oos_label"] == "locked_oos_report_only"
    assert guards[0].checkpoints[0][0] == "start"
    assert guards[0].finalize_calls[0]["status"] == "completed"
    assert guards[0].released is True

    payload = json.loads((output_dir / "fresh_portfolio_tuning_latest.json").read_text(encoding="utf-8"))
    assert payload["lockbox_policy"]["selection_label"] == "train_val_validation_only"
    assert payload["lockbox_policy"]["locked_oos_label"] == "locked_oos_report_only"
    assert payload["memory_policy"]["explicit_budget_bytes"] == expected_budget
    assert payload["memory_summary"]["memory_policy"]["explicit_budget_bytes"] == expected_budget
    assert payload["rss_log_path"].endswith("_rss_latest.jsonl")
    assert payload["memory_summary_path"].endswith("_memory_latest.json")

    rendered = (output_dir / "fresh_portfolio_tuning_latest.md").read_text(encoding="utf-8")
    assert "Locked-OOS label: `locked_oos_report_only`" in rendered
    assert "locked OOS" in rendered


def test_tuning_main_finalizes_failed_guard_when_payload_build_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "fresh_overhaul"
    candidate_csv = tmp_path / "candidates.csv"
    candidate_csv.write_text("name\n", encoding="utf-8")
    guards: list[_FakeMemoryGuard] = []

    def _fake_acquire(**kwargs: Any) -> _FakeMemoryGuard:
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    def _boom(args: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raise RuntimeError("candidate build failed")

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(MODULE, "build_payload", _boom)

    with pytest.raises(RuntimeError, match="candidate build failed"):
        MODULE.main(
            [
                "--candidate-csv",
                str(candidate_csv),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert guards[0].finalize_calls[0]["status"] == "failed"
    assert guards[0].finalize_calls[0]["error"] == "candidate build failed"
    assert guards[0].released is True
