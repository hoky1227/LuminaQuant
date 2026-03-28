from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_FOLLOWUP = Path("var/reports/exact_window_backtests/followup_status")
DEFAULT_REFRESH_JSON = DEFAULT_FOLLOWUP / "final_portfolio_validation_data_refresh_latest.json"
DEFAULT_DECISION_JSON = DEFAULT_FOLLOWUP / "portfolio_live_readiness_decision_latest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_utc(value: str | None) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(UTC)


def build_preflight_payload(
    *,
    config_path: Path,
    refresh_json: Path,
    decision_json: Path,
    stale_minutes: int,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    live = dict(config.get("live") or {})
    storage = dict(config.get("storage") or {})
    refresh = _read_json(refresh_json)
    decision = _read_json(decision_json)

    cutoff = _parse_utc(refresh.get("collection_cutoff_utc"))
    now = datetime.now(UTC)
    stale_for_minutes = None if cutoff is None else (now - cutoff).total_seconds() / 60.0
    refresh_is_stale = bool(stale_for_minutes is None or stale_for_minutes > float(stale_minutes))

    paper_mode = str(live.get("mode", "")).strip().lower() == "paper"
    real_mode = str(live.get("mode", "")).strip().lower() == "real"
    testnet = bool(live.get("testnet", False))
    require_real_flag = bool(live.get("require_real_enable_flag", False))
    real_enable_env = str(os.getenv("LUMINA_ENABLE_LIVE_REAL", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    decision_keep = str(decision.get("decision", "")).strip().lower() == "keep_incumbent"
    postgres_dsn = (
        str(storage.get("postgres_dsn", "") or "").strip()
        or str(os.getenv(str(storage.get("postgres_dsn_env", "LQ_POSTGRES_DSN")) or "LQ_POSTGRES_DSN", "")).strip()
        or str(os.getenv("LQ__STORAGE__POSTGRES_DSN", "")).strip()
    )
    postgres_dsn_present = bool(postgres_dsn)

    ready_for_paper = bool(
        paper_mode
        and testnet
        and require_real_flag
        and not refresh_is_stale
        and decision_keep
        and postgres_dsn_present
    )
    ready_for_real = bool(
        real_mode
        and not testnet
        and require_real_flag
        and real_enable_env
        and not refresh_is_stale
        and decision_keep
        and postgres_dsn_present
    )

    return {
        "generated_at": now.isoformat(),
        "artifact_kind": "live_readiness_preflight",
        "config_path": str(config_path.resolve()),
        "refresh_json": str(refresh_json.resolve()),
        "decision_json": str(decision_json.resolve()),
        "checks": {
            "paper_mode": paper_mode,
            "testnet": testnet,
            "require_real_enable_flag": require_real_flag,
            "real_enable_env": real_enable_env,
            "postgres_dsn_present": postgres_dsn_present,
            "decision_keep_incumbent": decision_keep,
            "refresh_completed": str(refresh.get("status", "")).strip().lower() == "completed",
            "decision_completed": bool(decision.get("decision")),
            "refresh_is_stale": refresh_is_stale,
        },
        "latest": {
            "refresh_cutoff_utc": refresh.get("collection_cutoff_utc"),
            "decision": decision.get("decision"),
            "feature_common_tail_utc": min(
                (row.get("last_timestamp_utc") for row in list(refresh.get("feature_results") or []) if row.get("last_timestamp_utc")),
                default=None,
            ),
            "stale_for_minutes": stale_for_minutes,
        },
        "status": {
            "ready_for_paper": ready_for_paper,
            "ready_for_real": ready_for_real,
        },
        "recommended_action": (
            "paper_run_allowed"
            if ready_for_paper
            else "real_run_allowed"
            if ready_for_real
            else "block_until_preflight_gaps_closed"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check paper/live readiness preflight.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--refresh-json", default=str(DEFAULT_REFRESH_JSON))
    parser.add_argument("--decision-json", default=str(DEFAULT_DECISION_JSON))
    parser.add_argument("--stale-minutes", type=int, default=30)
    args = parser.parse_args(argv)

    payload = build_preflight_payload(
        config_path=Path(args.config),
        refresh_json=Path(args.refresh_json),
        decision_json=Path(args.decision_json),
        stale_minutes=max(1, int(args.stale_minutes)),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
