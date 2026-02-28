from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

_RUN_BACKTEST_PATH = Path(__file__).resolve().parents[1] / "run_backtest.py"
_SPEC = importlib.util.spec_from_file_location("run_backtest_year_scale_module", _RUN_BACKTEST_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load run_backtest module from {_RUN_BACKTEST_PATH}")
run_backtest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_backtest)


def test_year_scale_low_memory_profile_default(monkeypatch):
    monkeypatch.delenv("LQ_BACKTEST_LOW_MEMORY", raising=False)
    monkeypatch.delenv("LQ_BACKTEST_PERSIST_OUTPUT", raising=False)

    profile = run_backtest._resolve_execution_profile(
        low_memory=None,
        persist_output=None,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 31),
    )

    assert profile["low_memory"] is True
    assert profile["record_history"] is False
    assert profile["record_trades"] is False
    assert profile["persist_output"] is False
    assert profile["track_metrics"] is True

