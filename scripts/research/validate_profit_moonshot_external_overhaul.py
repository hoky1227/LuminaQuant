"""Validate profit-moonshot external-overhaul gate readiness.

This verifier is intentionally artifact-driven and is intended for mission/launcher
use: it only returns pass when the profit-moonshot pass-gate, replay/tuning
artifacts, RSS limit, test evidence, and CI evidence are all present and passing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONTINUATION_VALIDATOR_PATH = Path(__file__).with_name("validate_profit_moonshot_continuation.py")
DEFAULT_SUMMARY_PATH = Path("var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.json")
DEFAULT_EXTERNAL_OVERHAUL_DIR = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260507/external_overhaul"
)
# Backward-compatible fallback used while this mission-specific folder is still bootstrapped.
DEFAULT_LEGACY_REPLAY_PATH = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul/fresh_start_overhaul_replay_latest.json"
)
DEFAULT_LEGACY_TUNING_PATH = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260507/fresh_overhaul/fresh_portfolio_tuning_latest.json"
)
DEFAULT_RESULT_DIR = Path(".omx/specs/autoresearch-profit-moonshot-pass-under-8gb")
DEFAULT_RESULT_PATH = DEFAULT_RESULT_DIR / "result.json"
DEFAULT_TEST_EVIDENCE_PATH = DEFAULT_RESULT_DIR / "tests_evidence.json"
DEFAULT_CI_EVIDENCE_PATH = DEFAULT_RESULT_DIR / "ci_evidence.json"
RSS_LIMIT_MIB = 8192.0
PASS_GATE_NAME = "profit_moonshot_pass_under_8gb_overhaul"


def _load_continuation_validator() -> Any:
    """Load the sibling continuation validator for direct script execution and tests."""
    spec = importlib.util.spec_from_file_location(
        "validate_profit_moonshot_continuation_for_external_overhaul",
        CONTINUATION_VALIDATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load continuation validator: {CONTINUATION_VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


continuation_validator = _load_continuation_validator()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_evidence_path(raw: str | None, fallback: Path) -> Path:
    path = Path(raw) if raw else fallback
    if path.exists():
        return path
    return fallback


def _metric_float(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("nan")
    return parsed


def _status_ok(value: str | None) -> bool:
    return str(value or "").strip().upper() == "PASS"


def _artifact_status(payload: dict[str, Any], *, required_type: str) -> tuple[bool, list[str]]:
    if not payload:
        return False, [f"missing_{required_type}_artifact"]
    if required_type == "replay":
        if int(payload.get("success_candidate_count", 0)) <= 0:
            return False, ["replay_has_no_success_candidates"]
        if payload.get("replay_survivor_count", 0) == 0:
            return False, ["replay_no_survivors"]
        return True, []
    if required_type == "tuning":
        if int(payload.get("success_candidate_count", 0)) <= 0:
            return False, ["tuning_has_no_success_candidates"]
        policy = payload.get("lockbox_policy")
        if not isinstance(policy, dict) or not str(policy.get("locked_oos_label") or "").strip():
            return False, ["tuning_missing_lockbox_policy"]
        return True, []
    return False, [f"unknown_artifact_type:{required_type}"]


def _max_peak_rss_mib(*, replay: dict[str, Any], tuning: dict[str, Any]) -> float:
    values: list[float] = []
    for payload in (replay, tuning):
        if not isinstance(payload, dict):
            continue
        values.append(_metric_float(payload.get("peak_rss_mib") or 0.0))
        memory_summary = payload.get("memory_summary")
        if isinstance(memory_summary, dict):
            values.append(_metric_float(memory_summary.get("peak_rss_bytes") or memory_summary.get("peak_memory_bytes") or 0.0) / (1024.0 * 1024.0))
    if not values:
        return float("nan")
    valid = [value for value in values if value == value]
    if not valid:
        return float("nan")
    return max(valid)


def _coerce_artifact_path(path: Path, fallback_dir: Path, legacy_path: Path) -> Path:
    if path.exists():
        return path
    return legacy_path if legacy_path.exists() else fallback_dir / path.name


def _evidence_pass(payload: Any, *, label: str) -> tuple[bool, list[str]]:
    if not payload:
        return False, [f"missing_{label}_evidence"]
    if isinstance(payload, dict):
        checks = payload.get("checks")
        if isinstance(checks, dict) and checks:
            failed = [
                key for key, check in checks.items() if not _status_ok((check or {}).get("status") if isinstance(check, dict) else check)
            ]
            return len(failed) == 0, [f"{label}_check_failed:{item}" for item in failed]
        if str(payload.get("status") or "").strip() == "":
            return False, [f"{label}_evidence_status_missing"]
        status_passed = _status_ok(str(payload.get("status") or ""))
        return status_passed, [] if status_passed else [f"{label}_evidence_status_not_pass"]
    if isinstance(payload, list):
        failed = []
        for idx, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            if not _status_ok(str(row.get("status") or row.get("result") or "")):
                failed.append(f"{label}_evidence_{idx}")
        return len(failed) == 0, [f"{item}:failed" for item in failed]
    return False, [f"unsupported_{label}_evidence_payload"]


def validate(
    *,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    replay_path: Path = DEFAULT_LEGACY_REPLAY_PATH,
    tuning_path: Path = DEFAULT_LEGACY_TUNING_PATH,
    tests_evidence_path: Path = DEFAULT_TEST_EVIDENCE_PATH,
    ci_evidence_path: Path = DEFAULT_CI_EVIDENCE_PATH,
    rss_limit_mib: float = RSS_LIMIT_MIB,
    result_path: Path = DEFAULT_RESULT_PATH,
    external_overhaul_dir: Path = DEFAULT_EXTERNAL_OVERHAUL_DIR,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    continuation = continuation_validator.validate(summary_path=summary_path)

    replay_candidate = continuation.get("candidate_mode") or continuation.get("promoted_by_summary")
    replay_ok = False
    replay_issues: list[str] = []
    tuning_ok = False
    tuning_issues: list[str] = []

    # Prefer explicit external_overhaul artifacts, but allow legacy fresh-overhaul fallback
    replay_path = _coerce_artifact_path(
        replay_path,
        fallback_dir=external_overhaul_dir,
        legacy_path=DEFAULT_LEGACY_REPLAY_PATH,
    )
    tuning_path = _coerce_artifact_path(
        tuning_path,
        fallback_dir=external_overhaul_dir,
        legacy_path=DEFAULT_LEGACY_TUNING_PATH,
    )

    replay_payload = _load_json(replay_path)
    tuning_payload = _load_json(tuning_path)
    replay_ok, replay_issues = _artifact_status(replay_payload, required_type="replay")
    tuning_ok, tuning_issues = _artifact_status(tuning_payload, required_type="tuning")

    peak_rss_mib = _max_peak_rss_mib(replay=replay_payload, tuning=tuning_payload)
    rss_ok = peak_rss_mib == peak_rss_mib and peak_rss_mib <= rss_limit_mib
    rss_issues = [] if rss_ok else [f"peak_rss_exceeds_limit:{peak_rss_mib}"]

    tests_evidence_path = _resolve_evidence_path(str(tests_evidence_path), tests_evidence_path)
    test_payload = _load_json(tests_evidence_path)
    tests_ok, test_issues = _evidence_pass(test_payload, label="tests")
    ci_evidence_path = _resolve_evidence_path(str(ci_evidence_path), ci_evidence_path)
    ci_payload = _load_json(ci_evidence_path)
    ci_ok, ci_issues = _evidence_pass(ci_payload, label="ci")

    # Lockbox label should always come from the tuning artifact policy.
    lockbox_label = ""
    if isinstance(tuning_payload.get("lockbox_policy"), dict):
        lockbox_label = str(tuning_payload["lockbox_policy"].get("locked_oos_label") or "")
    lockbox_ok = bool(lockbox_label)
    lockbox_issues = [] if lockbox_ok else ["missing_locked_oos_label"]

    pass_gate_ok = continuation.get("passed") is True and bool(replay_candidate)

    passed = bool(
        pass_gate_ok
        and replay_ok
        and tuning_ok
        and rss_ok
        and tests_ok
        and ci_ok
        and lockbox_ok
    )

    payload: dict[str, Any] = {
        "status": "passed" if passed else "running",
        "passed": passed,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mission": PASS_GATE_NAME,
        "summary_path": str(summary_path),
        "summary_pass_gate": bool(pass_gate_ok),
        "replay_path": str(replay_path),
        "tuning_path": str(tuning_path),
        "tests_evidence_path": str(tests_evidence_path),
        "ci_evidence_path": str(ci_evidence_path),
        "continuation_result": continuation,
        "artifact_rss_mib": {
            "peak_rss_mib": peak_rss_mib,
            "limit_mib": rss_limit_mib,
            "replay_peak_mib": _metric_float(replay_payload.get("peak_rss_mib") or 0.0),
            "tuning_peak_mib": _metric_float(tuning_payload.get("peak_rss_mib") or 0.0),
            "tuning_locked_oos_label": lockbox_label,
        },
        "checks": {
            "pass_gate": {
                "passed": bool(pass_gate_ok),
                "issues": [] if pass_gate_ok else ["continuation_gate_failed"],
                "summary": {
                    "candidate_mode": continuation.get("candidate_mode") or summary.get("promoted_candidate")
                    or summary.get("best_return_candidate")
                    or {},
                    "candidate_primary_split": continuation.get("candidate_primary_split"),
                    "promoted_by_summary": continuation.get("promoted_by_summary"),
                },
            },
            "replay": {
                "passed": bool(replay_ok),
                "issues": replay_issues,
                "path": str(replay_path),
            },
            "tuning": {
                "passed": bool(tuning_ok),
                "issues": tuning_issues,
                "path": str(tuning_path),
            },
            "rss": {
                "passed": bool(rss_ok),
                "issues": rss_issues,
                "peak_rss_mib": peak_rss_mib,
                "limit_mib": rss_limit_mib,
            },
            "tests": {
                "passed": bool(tests_ok),
                "issues": test_issues,
                "path": str(tests_evidence_path),
            },
            "ci": {
                "passed": bool(ci_ok),
                "issues": ci_issues,
                "path": str(ci_evidence_path),
            },
            "lockbox": {
                "passed": bool(lockbox_ok),
                "issues": lockbox_issues,
                "label": lockbox_label,
            },
        },
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    external_overhaul_dir.mkdir(parents=True, exist_ok=True)
    progress_payload = {
        "artifact_kind": "profit_moonshot_external_overhaul_verifier",
        "generated_at": payload["generated_at"],
        "passed": payload["passed"],
        "summary_pass_gate": bool(pass_gate_ok),
        "replay_path": str(replay_path),
        "tuning_path": str(tuning_path),
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (external_overhaul_dir / "validator_progress_latest.json").write_text(
        json.dumps(progress_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--replay-path", default=str(DEFAULT_LEGACY_REPLAY_PATH))
    parser.add_argument("--tuning-path", default=str(DEFAULT_LEGACY_TUNING_PATH))
    parser.add_argument("--external-overhaul-dir", default=str(DEFAULT_EXTERNAL_OVERHAUL_DIR))
    parser.add_argument("--tests-evidence", default=str(DEFAULT_TEST_EVIDENCE_PATH))
    parser.add_argument("--ci-evidence", default=str(DEFAULT_CI_EVIDENCE_PATH))
    parser.add_argument("--result-path", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--rss-limit-mib", type=float, default=RSS_LIMIT_MIB)
    parser.add_argument("--report-only", action="store_true", help="Write report and return status code only.")
    args = parser.parse_args(argv)

    result = validate(
        summary_path=Path(args.summary_path),
        replay_path=Path(args.replay_path),
        tuning_path=Path(args.tuning_path),
        tests_evidence_path=Path(args.tests_evidence),
        ci_evidence_path=Path(args.ci_evidence),
        rss_limit_mib=float(args.rss_limit_mib),
        result_path=Path(args.result_path),
        external_overhaul_dir=Path(args.external_overhaul_dir),
    )

    if args.report_only:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
