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


def test_validator_fails_running_result_with_missing_evidence(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write_json(result_path, {"status": "running", "passed": False})

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    failed = {check["name"] for check in payload["checks"] if not check["passed"]}
    assert "declared_pass_status" in failed
    assert "passing_candidate_artifact_exists" in failed
    assert "rss_under_8gib_evidence" in failed
    assert "local_tests_evidence" in failed
    assert "ci_success_evidence" in failed


def test_validator_passes_with_candidate_rss_tests_ci_and_push(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    time_log = tmp_path / "time.log"
    result_path = tmp_path / "result.json"
    _write_json(candidate_path, {"research_success_candidate": True})
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


def test_validator_rejects_over_budget_rss_evidence(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    _write_json(candidate_path, {"research_success_candidate": True})
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
