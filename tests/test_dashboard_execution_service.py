from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd


class _FakeBar:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name")


class _FakePie:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name")


class _FakeScatter:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name")


class _FakeFigure:
    def __init__(self, data: list[Any] | None = None) -> None:
        self.data = list(data or [])
        self.layout = types.SimpleNamespace(title=types.SimpleNamespace(text=None))

    def add_trace(self, trace: Any) -> None:
        self.data.append(trace)

    def update_layout(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "title":
                self.layout.title = types.SimpleNamespace(text=value)
            else:
                setattr(self.layout, key, value)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "execution_dashboard.py"


def _load_module(monkeypatch):
    fake_plotly = types.ModuleType("plotly")
    fake_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_graph_objects.Figure = _FakeFigure
    fake_graph_objects.Bar = _FakeBar
    fake_graph_objects.Pie = _FakePie
    fake_graph_objects.Scatter = _FakeScatter
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_graph_objects)

    spec = importlib.util.spec_from_file_location("dashboard_execution_service", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_build_execution_metric_rows_and_direction_table(monkeypatch) -> None:
    execution_dashboard = _load_module(monkeypatch)
    summary = {
        "buy_fills": 3,
        "sell_fills": 2,
        "avg_qty": 1.23456,
        "avg_notional": 42.678,
        "total_commission": 0.125,
        "avg_trade_return_pct": 1.5,
        "best_trade_pnl": 2.5,
        "worst_trade_pnl": -1.2,
        "win_streak_max": 4,
        "loss_streak_max": 2,
        "win_streak_avg": 1.8,
        "loss_streak_avg": 1.2,
        "holding_time_avg_sec": 90,
        "long_trades": 5,
        "long_win_rate": 0.6,
        "short_trades": 4,
        "short_win_rate": 0.25,
    }

    metric_rows = execution_dashboard.build_execution_metric_rows(
        summary,
        format_duration_seconds=lambda value: f"{int(value)} sec",
    )
    direction_table = execution_dashboard.build_direction_table(
        summary,
        safe_float=lambda value, default=0.0: default if value is None else float(value),
    )

    assert metric_rows[0] == [
        ("BUY fills", "3"),
        ("SELL fills", "2"),
        ("Avg Qty", "1.2346"),
        ("Avg Notional", "42.68"),
    ]
    assert metric_rows[2][-1] == ("Avg Holding Time", "90 sec")
    assert direction_table.to_dict(orient="records") == [
        {"Direction": "Long", "Closed Trades": 5, "Win Rate": "60.00%"},
        {"Direction": "Short", "Closed Trades": 4, "Win Rate": "25.00%"},
    ]


def test_trade_analytics_helpers_build_expected_figures(monkeypatch) -> None:
    execution_dashboard = _load_module(monkeypatch)
    trade_analytics = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-03-22T00:00:00Z",
                    "2026-03-22T00:01:00Z",
                    "2026-03-22T00:02:00Z",
                ],
                utc=True,
            ),
            "closed_qty": [0.0, 1.0, 2.0],
            "realized_pnl": [0.0, 1.5, -0.5],
            "cum_realized_pnl": [0.0, 1.5, 1.0],
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Bar=_FakeBar, Scatter=_FakeScatter)

    closed = execution_dashboard.filter_closed_trade_analytics(trade_analytics)
    trade_pnl_figure = execution_dashboard.build_trade_pnl_figure(closed, go_module=fake_go)
    cumulative_figure = execution_dashboard.build_cumulative_realized_pnl_figure(
        closed,
        go_module=fake_go,
    )
    streak_figure = execution_dashboard.build_streak_distribution_figure(
        closed,
        streak_groups=lambda outcomes: [(True, 1), (False, 1)] if outcomes else [],
        go_module=fake_go,
    )

    assert closed["closed_qty"].tolist() == [1.0, 2.0]
    assert trade_pnl_figure.layout.title.text == "Trade-by-Trade Realized PnL"
    assert [trace.name for trace in trade_pnl_figure.data] == ["Realized PnL per closing trade"]
    assert cumulative_figure.layout.title.text == "Cumulative Realized PnL"
    assert [trace.name for trace in cumulative_figure.data] == ["Cumulative Realized PnL"]
    assert streak_figure is not None
    assert streak_figure.layout.title.text == "Win/Loss Streak Distribution"
    assert [trace.name for trace in streak_figure.data] == ["Win", "Loss"]


def test_build_order_status_figure_aggregates_status_counts(monkeypatch) -> None:
    execution_dashboard = _load_module(monkeypatch)
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Pie=_FakePie)

    figure = execution_dashboard.build_order_status_figure(
        pd.DataFrame({"status": ["FILLED", "FILLED", None, "CANCELLED"]}),
        go_module=fake_go,
    )

    assert figure.layout.title.text == "Order Status Distribution"
    assert dict(
        zip(figure.data[0].kwargs["labels"], figure.data[0].kwargs["values"], strict=False)
    ) == {
        "FILLED": 2,
        "UNKNOWN": 1,
        "CANCELLED": 1,
    }
