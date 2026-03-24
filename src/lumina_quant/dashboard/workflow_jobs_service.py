"""Workflow job payload helpers for the Next dashboard migration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lumina_quant.config import BaseConfig
from lumina_quant.postgres_state import _connect_postgres


def resolve_dashboard_postgres_dsn(dsn: str | None = None) -> str:
    import os

    return str(dsn or os.getenv("LQ_POSTGRES_DSN") or getattr(BaseConfig, "POSTGRES_DSN", "") or "").strip()


def normalize_workflow_jobs_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    normalized: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        started_at = row.get("started_at")
        ended_at = row.get("ended_at")
        normalized.append(
            {
                "job_id": str(row.get("job_id") or ""),
                "workflow": str(row.get("workflow") or ""),
                "status": str(row.get("status") or ""),
                "requested_mode": str(row.get("requested_mode") or ""),
                "strategy": str(row.get("strategy") or ""),
                "run_id": str(row.get("run_id") or ""),
                "started_at": (
                    None
                    if pd.isna(started_at)
                    else pd.to_datetime(started_at, errors="coerce", utc=True).isoformat()
                ),
                "ended_at": (
                    None
                    if pd.isna(ended_at)
                    else pd.to_datetime(ended_at, errors="coerce", utc=True).isoformat()
                ),
            }
        )
    return normalized


def load_recent_workflow_jobs(connection: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                job_id,
                workflow,
                status,
                requested_mode,
                strategy,
                run_id,
                started_at,
                ended_at
            FROM workflow_jobs
            ORDER BY COALESCE(started_at, ended_at, last_updated) DESC
            LIMIT %s
            """,
            (int(max(1, limit)),),
        )
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description or ()]
    return normalize_workflow_jobs_frame(pd.DataFrame(rows, columns=columns))


def load_recent_workflow_jobs_payload(*, dsn: str | None = None, limit: int = 10) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {"jobs": [], "status": "missing_dsn"}
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {"jobs": [], "status": "missing_dsn"}

    conn = _connect_postgres(resolved_dsn)
    try:
        return {
            "jobs": load_recent_workflow_jobs(conn, limit=limit),
            "status": "ok",
        }
    finally:
        conn.close()


__all__ = [
    "load_recent_workflow_jobs",
    "load_recent_workflow_jobs_payload",
    "normalize_workflow_jobs_frame",
    "resolve_dashboard_postgres_dsn",
]
