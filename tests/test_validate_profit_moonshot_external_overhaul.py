from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_profit_moonshot_external_overhaul.py"
SPEC = importlib.util.spec_from_file_location("validate_profit_moonshot_external_overhaul", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_profit_moonshot_external_overhaul module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _summary_payload() -> dict[str, Any]:
    return {
        "decision": "promoted_candidate_found",
        "promoted_candidate": {
            "mode": "profit_moonshot_external_candidate",
            "primary_split": "val",
            "promotion_eligible": True,
            "total_return": 0.01,
            "sharpe": 1.1,
            "sortino": 1.2,
            "trades": 5,
            "liquidations": 0,
            "blockers": [],
        },
        "ranked_candidates": [],
    }


def _replay_payload() -> dict[str, Any]:
    return {
        "artifact_kind": "profit_moonshot_fresh_start_overhaul_replay",
        "success_candidate_count": 1,
        "replay_survivor_count": 1,
        "peak_rss_mib": 512.0,
    }


def _tuning_payload() -> dict[str, Any]:
    return {
        "artifact_kind": "profit_moonshot_fresh_portfolio_tuning",
        "success_candidate_count": 1,
        "peak_rss_mib": 768.0,
        "lockbox_policy": {
            "selection_label": "train_val_validation_only",
            "locked_oos_label": "locked_oos_report_only",
            "oos_is_report_only": True,
        },
        "memory_summary": {
            "peak_rss_bytes": 1024 * 1024 * 1024,
            "memory_policy": {"explicit_budget_bytes": 8 * 1024 * 1024 * 1024},
        },
    }


def _pass_evidence() -> dict[str, Any]:
    return {
        "status": "PASS",
        "checks": {
            "pytest": {"status": "PASS", "command": "uv run --extra dev pytest -q"},
            "ruff": {"status": "PASS", "command": "uv run --extra dev ruff check"},
        },
    }


def test_external_overhaul_validator_passes_with_gate_rss_tests_ci_and_lockbox(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = _write_json(tmp_path / "summary.json", _summary_payload())
    replay_path = _write_json(tmp_path / "external_overhaul" / "fresh_start_overhaul_replay_latest.json", _replay_payload())
    tuning_path = _write_json(tmp_path / "external_overhaul" / "fresh_portfolio_tuning_latest.json", _tuning_payload())
    tests_path = _write_json(tmp_path / "spec" / "tests_evidence.json", _pass_evidence())
    ci_path = _write_json(tmp_path / "spec" / "ci_evidence.json", _pass_evidence())
    result_path = tmp_path / "spec" / "result.json"
    external_dir = tmp_path / "external_overhaul"

    monkeypatch.setattr(
        MODULE,
        "continuation_validator",
        SimpleNamespace(
            validate=lambda summary_path: {
                "passed": True,
                "candidate_mode": "profit_moonshot_external_candidate",
                "candidate_primary_split": "val",
                "promoted_by_summary": "profit_moonshot_external_candidate",
            }
        ),
    )

    result = MODULE.validate(
        summary_path=summary_path,
        replay_path=replay_path,
        tuning_path=tuning_path,
        tests_evidence_path=tests_path,
        ci_evidence_path=ci_path,
        result_path=result_path,
        external_overhaul_dir=external_dir,
    )

    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["checks"]["rss"]["passed"] is True
    assert result["checks"]["lockbox"]["label"] == "locked_oos_report_only"
    assert result["checks"]["pass_gate"]["summary"]["candidate_primary_split"] == "val"
    assert json.loads(result_path.read_text(encoding="utf-8"))["passed"] is True
    progress = json.loads((external_dir / "validator_progress_latest.json").read_text(encoding="utf-8"))
    assert progress["passed"] is True
    assert progress["summary_pass_gate"] is True
    assert progress["next_required_evidence"] == []
    assert progress["checks"]["ci"]["passed"] is True


def test_external_overhaul_validator_fails_until_ci_evidence_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = _write_json(tmp_path / "summary.json", _summary_payload())
    replay_path = _write_json(tmp_path / "external_overhaul" / "fresh_start_overhaul_replay_latest.json", _replay_payload())
    tuning_path = _write_json(tmp_path / "external_overhaul" / "fresh_portfolio_tuning_latest.json", _tuning_payload())
    tests_path = _write_json(tmp_path / "spec" / "tests_evidence.json", _pass_evidence())
    ci_path = _write_json(tmp_path / "spec" / "ci_evidence.json", {"status": "PENDING"})
    result_path = tmp_path / "spec" / "result.json"
    external_dir = tmp_path / "external_overhaul"

    monkeypatch.setattr(
        MODULE,
        "continuation_validator",
        SimpleNamespace(validate=lambda summary_path: {"passed": True, "candidate_mode": "candidate"}),
    )

    result = MODULE.validate(
        summary_path=summary_path,
        replay_path=replay_path,
        tuning_path=tuning_path,
        tests_evidence_path=tests_path,
        ci_evidence_path=ci_path,
        result_path=result_path,
        external_overhaul_dir=external_dir,
    )

    assert result["passed"] is False
    assert result["status"] == "running"
    assert result["checks"]["tests"]["passed"] is True
    assert result["checks"]["ci"]["passed"] is False
    assert result["checks"]["ci"]["issues"] == ["ci_evidence_status_not_pass"]
    assert json.loads(result_path.read_text(encoding="utf-8"))["passed"] is False
    progress = json.loads((external_dir / "validator_progress_latest.json").read_text(encoding="utf-8"))
    assert progress["next_required_evidence"] == ["ci"]
    assert progress["checks"]["ci"]["issues"] == ["ci_evidence_status_not_pass"]


def test_external_overhaul_validator_cli_runs_as_direct_script(tmp_path: Path) -> None:
    summary_path = _write_json(tmp_path / "summary.json", _summary_payload())
    replay_path = _write_json(tmp_path / "external_overhaul" / "fresh_start_overhaul_replay_latest.json", _replay_payload())
    tuning_path = _write_json(tmp_path / "external_overhaul" / "fresh_portfolio_tuning_latest.json", _tuning_payload())
    tests_path = _write_json(tmp_path / "spec" / "tests_evidence.json", _pass_evidence())
    ci_path = _write_json(tmp_path / "spec" / "ci_evidence.json", _pass_evidence())
    result_path = tmp_path / "spec" / "result.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--summary-path",
            str(summary_path),
            "--replay-path",
            str(replay_path),
            "--tuning-path",
            str(tuning_path),
            "--tests-evidence",
            str(tests_path),
            "--ci-evidence",
            str(ci_path),
            "--result-path",
            str(result_path),
            "--external-overhaul-dir",
            str(tmp_path / "external_overhaul"),
            "--report-only",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(result_path.read_text(encoding="utf-8"))["passed"] is True
    assert "\"passed\": true" in completed.stdout
