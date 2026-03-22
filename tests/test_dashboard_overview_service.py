from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd


class _FakeHeatmap:
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
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "overview_dashboard.py"


def _load_module(monkeypatch):
    fake_plotly = types.ModuleType("plotly")
    fake_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_graph_objects.Figure = _FakeFigure
    fake_graph_objects.Heatmap = _FakeHeatmap
    fake_graph_objects.Scatter = _FakeScatter
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_graph_objects)

    spec = importlib.util.spec_from_file_location("dashboard_overview_service", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_build_equity_and_drawdown_figures(monkeypatch) -> None:
    overview_dashboard = _load_module(monkeypatch)
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-03-22T00:00:00Z", "2026-03-22T00:01:00Z", "2026-03-22T00:02:00Z"],
                utc=True,
            ),
            "total": [100.0, 105.0, 102.0],
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)

    equity_figure = overview_dashboard.build_equity_curve_figure(frame, go_module=fake_go)
    drawdown_figure = overview_dashboard.build_drawdown_figure(frame, go_module=fake_go)

    assert equity_figure.layout.title.text == "Equity Curve"
    assert [trace.name for trace in equity_figure.data] == ["Strategy Equity"]
    assert drawdown_figure.layout.title.text == "Drawdown"
    assert [trace.name for trace in drawdown_figure.data] == ["Drawdown"]


def test_build_optional_benchmark_and_funding_figures(monkeypatch) -> None:
    overview_dashboard = _load_module(monkeypatch)
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-03-22T00:00:00Z", "2026-03-22T00:01:00Z"],
                utc=True,
            ),
            "benchmark_price": [200.0, 205.0],
            "funding": [0.05, 0.02],
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)

    benchmark_figure = overview_dashboard.build_benchmark_price_figure(frame, go_module=fake_go)
    funding_figure = overview_dashboard.build_funding_figure(frame, go_module=fake_go)

    assert benchmark_figure is not None
    assert benchmark_figure.layout.title.text == "Benchmark Price (from Equity Metadata)"
    assert funding_figure is not None
    assert funding_figure.layout.title.text == "Funding (Net) Over Time"


def test_build_cumulative_return_figure_validates_length(monkeypatch) -> None:
    overview_dashboard = _load_module(monkeypatch)
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-03-22T00:00:00Z", "2026-03-22T00:01:00Z", "2026-03-22T00:02:00Z"],
                utc=True,
            )
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)

    bad_figure = overview_dashboard.build_cumulative_return_figure(
        frame,
        {"cum_return_series": [0.1]},
        go_module=fake_go,
    )
    good_figure = overview_dashboard.build_cumulative_return_figure(
        frame,
        {"cum_return_series": [0.05, 0.1]},
        go_module=fake_go,
    )

    assert bad_figure is None
    assert good_figure is not None
    assert good_figure.layout.title.text == "Cumulative Return"
    assert [trace.name for trace in good_figure.data] == ["Cumulative Return"]


def test_build_monthly_returns_heatmap_formats_cells(monkeypatch) -> None:
    overview_dashboard = _load_module(monkeypatch)
    monthly_table = pd.DataFrame(
        [[0.1, float("nan")], [-0.05, 0.0]],
        index=[2025, 2026],
        columns=["Jan", "Feb"],
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Heatmap=_FakeHeatmap)

    figure = overview_dashboard.build_monthly_returns_heatmap(
        monthly_table,
        safe_float=lambda value, default=0.0: default if pd.isna(value) else float(value),
        go_module=fake_go,
    )

    assert figure.layout.title.text == "Monthly Returns Heatmap"
    assert figure.data[0].kwargs["x"] == ["Jan", "Feb"]
    assert figure.data[0].kwargs["y"] == ["2025", "2026"]
    assert figure.data[0].kwargs["text"] == [["10.00%", ""], ["-5.00%", "0.00%"]]
