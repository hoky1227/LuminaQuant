"""Live poll data handler that consumes committed materialized windows only."""

from __future__ import annotations

from lumina_quant.live.data_materialized import CommittedWindowDataHandler


class LiveDataHandler(CommittedWindowDataHandler):
    """Polling-mode live handler backed by committed materialized parquet windows."""


__all__ = ["LiveDataHandler"]
