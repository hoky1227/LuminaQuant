"""Canonical split windows and memory-guard helpers for portfolio follow-up lanes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from lumina_quant.core.memory_budget import DEFAULT_EXECUTION_MEMORY_POLICY
from lumina_quant.eval.exact_window_runtime import HeavyRunLock, RSSGuard

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "var" / "reports" / "exact_window_backtests"
FOLLOWUP_ROOT = REPORT_ROOT / "followup_status"

TRAIN_START = "2025-01-01T00:00:00Z"
TRAIN_END_EXCLUSIVE = "2026-01-01T00:00:00Z"
VAL_START = "2026-01-01T00:00:00Z"
VAL_END_EXCLUSIVE = "2026-02-01T00:00:00Z"
OOS_START = "2026-02-01T00:00:00Z"

TRAIN_START_DATE = date(2025, 1, 1)
VAL_START_DATE = date(2026, 1, 1)
OOS_START_DATE = date(2026, 2, 1)

PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE = (
    FOLLOWUP_ROOT / "portfolio_one_shot_incumbent_bundle_latest.json"
)
PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE = FOLLOWUP_ROOT / "portfolio_one_shot_current_bundle_latest.json"
PORTFOLIO_CURRENT_OPTIMIZATION = (
    FOLLOWUP_ROOT / "portfolio_one_shot_current_opt" / "portfolio_optimization_latest.json"
)

PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH = FOLLOWUP_ROOT / "portfolio_followup_heavy_run.lock"
PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES = 8 * 1024 * 1024 * 1024
MEMORY_GUARD_DIRNAME = "_memory_guard"


def split_windows() -> dict[str, str]:
    """Return the canonical portfolio research split boundaries."""
    return {
        "train_start": TRAIN_START,
        "train_end_exclusive": TRAIN_END_EXCLUSIVE,
        "val_start": VAL_START,
        "val_end_exclusive": VAL_END_EXCLUSIVE,
        "oos_start": OOS_START,
    }


def split_dates() -> dict[str, date]:
    """Return the split boundaries as day-level dates."""
    return {
        "train_start": TRAIN_START_DATE,
        "val_start": VAL_START_DATE,
        "oos_start": OOS_START_DATE,
    }


def split_for_date(day_value: date) -> str:
    """Map a date into the canonical train/val/oos bucket."""
    if day_value < VAL_START_DATE:
        return "train"
    if day_value < OOS_START_DATE:
        return "val"
    return "oos"


def split_for_day_key(day_key: str) -> str:
    """Map a YYYY-MM-DD token into the canonical train/val/oos bucket."""
    return split_for_date(date.fromisoformat(day_key))


def resolve_incumbent_bundle_path(path: str | Path | None = None) -> Path:
    """Prefer the incumbent bundle and fall back to the legacy current bundle."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    if PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE not in candidates:
        candidates.append(PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE)
    if PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE not in candidates:
        candidates.append(PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if path is not None:
        return Path(path).resolve()
    return PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE.resolve()


def memory_policy_payload(*, budget_bytes: int | None = None) -> dict[str, Any]:
    """Return the canonical memory-budget policy for portfolio follow-up lanes."""
    policy = asdict(DEFAULT_EXECUTION_MEMORY_POLICY)
    policy["total_memory_cap_bytes"] = DEFAULT_EXECUTION_MEMORY_POLICY.total_memory_cap_bytes
    policy["rss_limit_gib"] = DEFAULT_EXECUTION_MEMORY_POLICY.rss_limit_gib
    policy["heavy_lock_path"] = str(PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH.resolve())
    if budget_bytes is not None:
        policy["explicit_budget_bytes"] = int(budget_bytes)
        policy["explicit_budget_gib"] = float(int(budget_bytes) / (1024**3))
    return policy


@dataclass(slots=True)
class PortfolioMemoryGuard:
    """Wrapper around the exact-window heavy-run lock and RSS telemetry."""

    run_name: str
    label: str
    output_dir: Path
    rss_log_path: Path
    summary_path: Path
    lock: HeavyRunLock
    guard: RSSGuard

    def sample(self, *, event: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.guard.sample(event=event, context=context)

    def checkpoint(self, event: str, context: dict[str, Any] | None = None) -> None:
        self.guard.checkpoint(event, context)

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = {
            "artifact_kind": "portfolio_followup_memory_summary",
            "schema_version": "1.0",
            "run_name": self.run_name,
            "label": self.label,
            "output_dir": str(self.output_dir),
            "memory_policy": memory_policy_payload(budget_bytes=self.guard.budget_bytes),
            "context": dict(context or {}),
            **self.guard.finalize(status=status, error=error),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def release(self) -> None:
        self.lock.release()


def acquire_portfolio_memory_guard(
    *,
    run_name: str,
    output_dir: str | Path,
    input_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    budget_bytes: int | None = None,
    memory_budget_bytes: int | None = None,
    fixed_budget_bytes: int | None = None,
) -> PortfolioMemoryGuard:
    """Acquire the shared heavy-run lock and RSS logger for a portfolio lane."""
    resolved_budget_bytes = next(
        (
            int(value)
            for value in (budget_bytes, memory_budget_bytes, fixed_budget_bytes)
            if value is not None
        ),
        None,
    )
    resolved_output_dir = Path(output_dir).resolve()
    memory_dir = resolved_output_dir / MEMORY_GUARD_DIRNAME
    memory_dir.mkdir(parents=True, exist_ok=True)
    label = f"portfolio_followup::{run_name}"
    lock = HeavyRunLock.acquire(
        lock_path=PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH,
        label=label,
        metadata={
            "run_name": run_name,
            "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
            **dict(metadata or {}),
        },
    )
    rss_log_path = memory_dir / f"{run_name}_rss_latest.jsonl"
    summary_path = memory_dir / f"{run_name}_memory_latest.json"
    guard = RSSGuard(log_path=rss_log_path, label=label, budget_bytes=resolved_budget_bytes)
    return PortfolioMemoryGuard(
        run_name=run_name,
        label=label,
        output_dir=resolved_output_dir,
        rss_log_path=rss_log_path,
        summary_path=summary_path,
        lock=lock,
        guard=guard,
    )


__all__ = [
    "FOLLOWUP_ROOT",
    "MEMORY_GUARD_DIRNAME",
    "OOS_START",
    "OOS_START_DATE",
    "PORTFOLIO_CURRENT_OPTIMIZATION",
    "PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES",
    "PORTFOLIO_FOLLOWUP_HEAVY_LOCK_PATH",
    "PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE",
    "PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE",
    "REPORT_ROOT",
    "TRAIN_END_EXCLUSIVE",
    "TRAIN_START",
    "TRAIN_START_DATE",
    "VAL_END_EXCLUSIVE",
    "VAL_START",
    "VAL_START_DATE",
    "PortfolioMemoryGuard",
    "acquire_portfolio_memory_guard",
    "memory_policy_payload",
    "resolve_incumbent_bundle_path",
    "split_dates",
    "split_for_date",
    "split_for_day_key",
    "split_windows",
]
