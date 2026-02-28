from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_RUN_BACKTEST_PATH = Path(__file__).resolve().parents[1] / "run_backtest.py"
_SPEC = importlib.util.spec_from_file_location("run_backtest_no_equity_module", _RUN_BACKTEST_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load run_backtest module from {_RUN_BACKTEST_PATH}")
run_backtest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_backtest)


def test_no_equity_curve_materialization_in_low_memory():
    logged_rows: list[dict[str, object]] = []

    class _Audit:
        def log_equity(self, run_id, **kwargs):
            logged_rows.append({"run_id": run_id, **kwargs})

        def log_fill(self, *_args, **_kwargs):
            return None

    class _Portfolio:
        def __init__(self):
            self._equity_points = [(1735689600.0, 1000.0), (1735689900.0, 1010.0)]
            self.trades = []

        def create_equity_curve_dataframe(self):
            raise AssertionError("low-memory path must not materialize equity_curve dataframe")

    backtest = SimpleNamespace(portfolio=_Portfolio())
    counts = run_backtest._persist_backtest_audit_rows(
        _Audit(),
        "run-low-memory",
        backtest,
        low_memory=True,
    )

    assert counts["equity_rows"] == 2
    assert counts["fill_rows"] == 0
    assert len(logged_rows) == 2

