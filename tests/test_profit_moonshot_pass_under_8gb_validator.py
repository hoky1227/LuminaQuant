from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_profit_moonshot_pass_under_8gb.py"
SPEC = importlib.util.spec_from_file_location("validate_profit_moonshot_pass_under_8gb", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _passing_candidate_payload() -> dict:
    return {
        "research_success_candidate": True,
        "metrics": {
            "train_monthlyized_return": 0.025,
            "validation_monthlyized_return": 0.026,
            "locked_oos_monthlyized_return": 0.024,
            "locked_oos_total_return": 0.05,
            "locked_oos_max_drawdown": 0.12,
            "locked_oos_sharpe": 2.5,
            "locked_oos_sortino": 3.8,
            "locked_oos_smart_sortino": 3.2,
            "locked_oos_calmar": 1.6,
        },
    }


def test_validator_fails_running_result_with_missing_evidence(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write_json(result_path, {"status": "running", "passed": False})

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    failed = {check["name"] for check in payload["checks"] if not check["passed"]}
    assert "declared_pass_status" in failed
    assert "passing_candidate_artifact_exists" in failed
    assert "candidate_return_quality_contract" in failed
    assert "rss_under_8gib_evidence" in failed
    assert "local_tests_evidence" in failed
    assert "ci_success_evidence" in failed


def test_validator_passes_with_candidate_rss_tests_ci_and_push(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    time_log = tmp_path / "time.log"
    result_path = tmp_path / "result.json"
    _write_json(candidate_path, _passing_candidate_payload())
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    time_log.write_text("Maximum resident set size (kbytes): 65536\n", encoding="utf-8")
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": True,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json", "time.log"],
            "test_evidence": [{"command": "pytest", "passed": True}],
            "ci_evidence": [{"workflow": "ci", "conclusion": "success"}],
            "git_evidence": [{"remote": "private/main", "pushed": True}],
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is True
    assert all(check["passed"] for check in payload["checks"])
    assert [item["under_8gib"] for item in payload["rss_evidence"]] == [True, True]


def test_validator_rejects_low_monthly_return_candidate_even_with_old_pass_label(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    source_artifact = tmp_path / "source.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    _write_json(
        candidate_path,
        {
            "name": "old_low_return_pass",
            "research_success_candidate": True,
            "source_artifact": "source.json",
        },
    )
    _write_json(
        source_artifact,
        {
            "best_success_candidate": {
                "name": "old_low_return_pass",
                "splits": {
                    "train": {"metrics": {"cagr": 0.055, "total_return": 0.055}},
                    "val": {"metrics": {"cagr": 0.275, "total_return": 0.04}},
                    "oos": {
                        "metrics": {
                            "cagr": 0.076,
                            "total_return": 0.0134,
                            "max_drawdown": 0.0018,
                            "sharpe": 5.4,
                            "sortino": 6.7,
                            "calmar": 42.0,
                        }
                    },
                },
            }
        },
    )
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    assert quality_check["passed"] is False
    details = json.loads(quality_check["detail"])
    assert details["train_monthlyized_return"] < validator.MIN_STABLE_MONTHLY_RETURN
    assert details["locked_oos_monthlyized_return"] < validator.MIN_STABLE_MONTHLY_RETURN


def test_validator_rejects_over_budget_rss_evidence(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    _write_json(candidate_path, _passing_candidate_payload())
    _write_json(rss_summary, {"peak_rss_bytes": validator.EIGHT_GIB_BYTES + 1})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    rss_check = next(check for check in payload["checks"] if check["name"] == "rss_under_8gib_evidence")
    assert rss_check["passed"] is False
