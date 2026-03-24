"""Postgres state-store helpers shared by the dashboard app."""

from __future__ import annotations

import logging
import os
from typing import Any


class StateCursor:
    """Thin adapter that normalizes cursor parameter handling."""

    def __init__(self, cursor: Any):
        self._cursor = cursor

    def execute(self, query: Any, params: Any = None) -> StateCursor:
        self._cursor.execute(str(query), tuple(params or ()))
        return self

    def __enter__(self) -> StateCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def close(self) -> None:
        self._cursor.close()

    @property
    def description(self):
        return self._cursor.description


class StateConnection:
    """Connection adapter with sqlite-like helpers used by the dashboard."""

    def __init__(self, connection: Any):
        self._conn = connection

    def cursor(self) -> StateCursor:
        return StateCursor(self._conn.cursor())

    def execute(self, query: Any, params: Any = None) -> StateCursor:
        cursor = self.cursor()
        cursor.execute(query, params)
        return cursor

    def executescript(self, script: str) -> None:
        with self._conn.cursor() as cursor:
            for statement in str(script).split(";"):
                payload = statement.strip()
                if not payload:
                    continue
                cursor.execute(payload)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def resolve_postgres_dsn(dsn: str | None = None, *, base_config: Any = None) -> str:
    """Resolve dashboard Postgres DSN from explicit value, env, or config."""
    token = str(
        dsn
        or os.getenv("LQ_POSTGRES_DSN")
        or getattr(base_config, "POSTGRES_DSN", "")
        or ""
    ).strip()
    return token


def connect_state_store(
    dsn: str,
    *,
    resolve_postgres_dsn=resolve_postgres_dsn,
) -> StateConnection:
    """Open a normalized dashboard state-store connection."""
    resolved = resolve_postgres_dsn(dsn)
    if not resolved:
        raise RuntimeError("Postgres DSN is required.")
    from lumina_quant.postgres_state import _connect_postgres

    return StateConnection(_connect_postgres(resolved))


def execute_query(
    dsn: str,
    query: str,
    params: Any = None,
    *,
    connect_state_store=connect_state_store,
    logger: logging.Logger | None = None,
):
    """Execute a query and tolerate fetchall failures with an empty result."""
    active_logger = logger or logging.getLogger(__name__)
    conn = connect_state_store(dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params or ()))
            try:
                rows = cursor.fetchall()
            except Exception:
                active_logger.warning(
                    "Dashboard query helper fell back to an empty result set after fetchall failed."
                )
                rows = []
        conn.commit()
        return rows
    finally:
        conn.close()


def read_sql_query(
    dsn: str,
    query: str,
    params: Any = None,
    *,
    connect_state_store=connect_state_store,
):
    """Run ``pandas.read_sql_query`` against the normalized state store."""
    import pandas as pd

    conn = connect_state_store(dsn)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
