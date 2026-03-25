"""Helpers for dashboard navigation on migration-touched paths."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def select_dashboard_view(
    *,
    streamlit: Any,
    label: str,
    view_options: Sequence[str],
    index: int = 0,
) -> str:
    """Read the current dashboard view selection from the sidebar."""
    return str(streamlit.sidebar.radio(label, list(view_options), index=index))
