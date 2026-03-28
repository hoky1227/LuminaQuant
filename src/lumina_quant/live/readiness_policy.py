from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.configuration.loader import load_runtime_config, load_yaml_config

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_FOLLOWUP_ROOT = Path("var/reports/exact_window_backtests/followup_status")
DEFAULT_REFRESH_JSON = DEFAULT_FOLLOWUP_ROOT / "final_portfolio_validation_data_refresh_latest.json"
DEFAULT_DECISION_JSON = DEFAULT_FOLLOWUP_ROOT / "portfolio_live_readiness_decision_latest.json"
DEFAULT_PREFLIGHT_STALE_MINUTES = 30

_TRUTHY = {"1", "true", "yes", "on"}


class LiveReadinessBlockedError(RuntimeError):
    """Raised when live startup is not ready for the requested execution mode."""

    def __init__(self, *, mode: str, recommended_action: str, payload: dict[str, Any]) -> None:
        self.mode = str(mode or "").strip().lower() or "paper"
        self.recommended_action = str(recommended_action or "block_until_preflight_gaps_closed")
        self.payload = dict(payload or {})
        super().__init__(
            f"Live readiness blocked for mode={self.mode}. "
            f"recommended_action={self.recommended_action}"
        )


@dataclass(frozen=True, slots=True)
class LiveReadinessVerdict:
    generated_at: str
    config_path: str
    refresh_json: str
    decision_json: str
    checks: dict[str, Any]
    latest: dict[str, Any]
    status: dict[str, bool]
    recommended_action: str

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_kind"] = "live_readiness_preflight"
        return payload


def _parse_utc(value: str | None) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _env_truthy(name: str, env: Mapping[str, str] | None = None) -> bool:
    source = env or os.environ
    return str(source.get(name, "") or "").strip().lower() in _TRUTHY


def effective_startup_reconciliation_hard_fail(*, mode: str, configured: bool) -> bool:
    resolved_mode = str(mode or "").strip().lower()
    return bool(configured) or resolved_mode == "real"


def real_mode_explicitly_enabled(*, mode: str, require_real_enable_flag: bool, env: Mapping[str, str] | None = None) -> bool:
    resolved_mode = str(mode or "").strip().lower()
    if resolved_mode != "real":
        return True
    if not bool(require_real_enable_flag):
        return True
    return _env_truthy("LUMINA_ENABLE_LIVE_REAL", env)


def _build_live_readiness_verdict(
    *,
    config_path: Path,
    refresh_json: Path,
    decision_json: Path,
    stale_minutes: int,
    runtime_live: Mapping[str, Any],
    runtime_storage: Mapping[str, Any],
    raw_storage: Mapping[str, Any],
    refresh: Mapping[str, Any],
    decision: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> LiveReadinessVerdict:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = _parse_utc(str(refresh.get("collection_cutoff_utc") or ""))
    stale_for_minutes = None if cutoff is None else (current_time - cutoff).total_seconds() / 60.0
    refresh_is_stale = bool(stale_for_minutes is None or stale_for_minutes > float(stale_minutes))

    mode = str(runtime_live.get("mode", "") or "").strip().lower()
    paper_mode = mode == "paper"
    real_mode = mode == "real"
    testnet = bool(runtime_live.get("testnet", False))
    require_real_flag = bool(runtime_live.get("require_real_enable_flag", False))
    real_enable_env = real_mode_explicitly_enabled(
        mode=mode,
        require_real_enable_flag=require_real_flag,
        env=env,
    )
    decision_keep = str(decision.get("decision", "") or "").strip().lower() == "keep_incumbent"

    postgres_dsn_env_name = str(
        runtime_storage.get("postgres_dsn_env", "")
        or raw_storage.get("postgres_dsn_env", "")
        or "LQ_POSTGRES_DSN"
    ).strip() or "LQ_POSTGRES_DSN"
    postgres_dsn = (
        str(runtime_storage.get("postgres_dsn", "") or "").strip()
        or str(raw_storage.get("postgres_dsn", "") or "").strip()
        or str((env or os.environ).get(postgres_dsn_env_name, "") or "").strip()
        or str((env or os.environ).get("LQ__STORAGE__POSTGRES_DSN", "") or "").strip()
    )
    postgres_dsn_present = bool(postgres_dsn)

    refresh_completed = str(refresh.get("status", "") or "").strip().lower() == "completed"
    decision_completed = bool(decision.get("decision"))

    ready_for_paper = bool(
        paper_mode
        and testnet
        and require_real_flag
        and refresh_completed
        and decision_completed
        and not refresh_is_stale
        and decision_keep
        and postgres_dsn_present
    )
    ready_for_real = bool(
        real_mode
        and not testnet
        and require_real_flag
        and real_enable_env
        and refresh_completed
        and decision_completed
        and not refresh_is_stale
        and decision_keep
        and postgres_dsn_present
    )

    feature_common_tail = min(
        (
            row.get("last_timestamp_utc")
            for row in list(refresh.get("feature_results") or [])
            if row.get("last_timestamp_utc")
        ),
        default=None,
    )
    recommended_action = (
        "paper_run_allowed"
        if ready_for_paper
        else "real_run_allowed"
        if ready_for_real
        else "block_until_preflight_gaps_closed"
    )

    return LiveReadinessVerdict(
        generated_at=current_time.isoformat(),
        config_path=str(config_path.resolve()),
        refresh_json=str(refresh_json.resolve()),
        decision_json=str(decision_json.resolve()),
        checks={
            "mode": mode,
            "paper_mode": paper_mode,
            "real_mode": real_mode,
            "testnet": testnet,
            "require_real_enable_flag": require_real_flag,
            "real_enable_env": real_enable_env,
            "postgres_dsn_present": postgres_dsn_present,
            "decision_keep_incumbent": decision_keep,
            "refresh_completed": refresh_completed,
            "decision_completed": decision_completed,
            "refresh_is_stale": refresh_is_stale,
            "effective_startup_reconciliation_hard_fail": effective_startup_reconciliation_hard_fail(
                mode=mode,
                configured=bool(runtime_live.get("startup_reconciliation_hard_fail", False)),
            ),
        },
        latest={
            "refresh_cutoff_utc": refresh.get("collection_cutoff_utc"),
            "decision": decision.get("decision"),
            "feature_common_tail_utc": feature_common_tail,
            "stale_for_minutes": stale_for_minutes,
        },
        status={
            "ready_for_paper": ready_for_paper,
            "ready_for_real": ready_for_real,
        },
        recommended_action=recommended_action,
    )


def build_live_readiness_payload(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    refresh_json: Path = DEFAULT_REFRESH_JSON,
    decision_json: Path = DEFAULT_DECISION_JSON,
    stale_minutes: int = DEFAULT_PREFLIGHT_STALE_MINUTES,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    effective_env = env or os.environ
    runtime = load_runtime_config(config_path=str(config_path), env=effective_env)
    raw = load_yaml_config(config_path=str(config_path))
    refresh = _read_json(refresh_json)
    decision = _read_json(decision_json)
    verdict = _build_live_readiness_verdict(
        config_path=config_path,
        refresh_json=refresh_json,
        decision_json=decision_json,
        stale_minutes=max(1, int(stale_minutes)),
        runtime_live=asdict(runtime.live),
        runtime_storage=asdict(runtime.storage),
        raw_storage=dict(raw.get("storage") or {}),
        refresh=refresh,
        decision=decision,
        env=effective_env,
        now=now,
    )
    return verdict.as_payload()


def enforce_live_readiness_from_files(
    *,
    mode: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    refresh_json: Path = DEFAULT_REFRESH_JSON,
    decision_json: Path = DEFAULT_DECISION_JSON,
    stale_minutes: int = DEFAULT_PREFLIGHT_STALE_MINUTES,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    payload = build_live_readiness_payload(
        config_path=config_path,
        refresh_json=refresh_json,
        decision_json=decision_json,
        stale_minutes=stale_minutes,
        env=env,
        now=now,
    )
    status = dict(payload.get("status") or {})
    resolved_mode = str(mode or "").strip().lower() or "paper"
    allowed = bool(status.get("ready_for_real" if resolved_mode == "real" else "ready_for_paper"))
    if not allowed:
        raise LiveReadinessBlockedError(
            mode=resolved_mode,
            recommended_action=str(payload.get("recommended_action") or "block_until_preflight_gaps_closed"),
            payload=payload,
        )
    return payload


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DECISION_JSON",
    "DEFAULT_FOLLOWUP_ROOT",
    "DEFAULT_PREFLIGHT_STALE_MINUTES",
    "DEFAULT_REFRESH_JSON",
    "LiveReadinessBlockedError",
    "LiveReadinessVerdict",
    "build_live_readiness_payload",
    "effective_startup_reconciliation_hard_fail",
    "enforce_live_readiness_from_files",
    "real_mode_explicitly_enabled",
]
