from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import Any

import pandas as pd


class _FakeBar:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeScatter:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


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
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "risk_dashboard.py"


def _load_module(monkeypatch):
    fake_plotly = types.ModuleType("plotly")
    fake_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_graph_objects.Figure = _FakeFigure
    fake_graph_objects.Bar = _FakeBar
    fake_graph_objects.Scatter = _FakeScatter
    monkeypatch.setitem(__import__("sys").modules, "plotly", fake_plotly)
    monkeypatch.setitem(__import__("sys").modules, "plotly.graph_objects", fake_graph_objects)

    spec = importlib.util.spec_from_file_location("dashboard_risk_service", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_risk_reason_figure_aggregates_reason_counts(monkeypatch) -> None:
    risk_dashboard = _load_module(monkeypatch)
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Bar=_FakeBar)
    figure = risk_dashboard.build_risk_reason_figure(
        pd.DataFrame({"reason": ["LIMIT", "LIMIT", None, "STOP"]}),
        go_module=fake_go,
    )

    assert figure.layout.title.text == "Risk Event Counts by Reason"
    assert dict(zip(figure.data[0].kwargs["x"], figure.data[0].kwargs["y"], strict=False)) == {
        "LIMIT": 2,
        "UNKNOWN": 1,
        "STOP": 1,
    }


def test_prepare_heartbeat_interval_frame_sorts_and_computes_average(monkeypatch) -> None:
    risk_dashboard = _load_module(monkeypatch)
    frame, avg = risk_dashboard.prepare_heartbeat_interval_frame(
        pd.DataFrame(
            {
                "heartbeat_time": pd.to_datetime(
                    [
                        "2026-03-22T00:00:10Z",
                        "2026-03-22T00:00:00Z",
                        "2026-03-22T00:00:25Z",
                    ]
                )
            }
        )
    )

    assert frame["heartbeat_time"].tolist() == sorted(frame["heartbeat_time"].tolist())
    assert pd.isna(frame["delta_sec"].iloc[0])
    assert frame["delta_sec"].iloc[1:].tolist() == [10.0, 15.0]
    assert avg == 12.5


def test_build_heartbeat_interval_figure_uses_prepared_frame(monkeypatch) -> None:
    risk_dashboard = _load_module(monkeypatch)
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)
    hb = pd.DataFrame(
        {
            "heartbeat_time": pd.to_datetime(["2026-03-22T00:00:00Z", "2026-03-22T00:00:05Z"]),
            "delta_sec": [None, 5.0],
        }
    )

    figure = risk_dashboard.build_heartbeat_interval_figure(hb, go_module=fake_go)
    y_values = figure.data[0].kwargs["y"].tolist()

    assert figure.layout.title.text == "Heartbeat Interval Trend"
    assert figure.data[0].kwargs["name"] == "Heartbeat interval"
    assert pd.isna(y_values[0])
    assert y_values[1:] == [5.0]


def test_build_order_state_figure_aggregates_state_counts(monkeypatch) -> None:
    risk_dashboard = _load_module(monkeypatch)
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Bar=_FakeBar)
    figure = risk_dashboard.build_order_state_figure(
        pd.DataFrame({"state": ["FILLED", "FILLED", None, "CANCELLED"]}),
        go_module=fake_go,
    )

    assert figure.layout.title.text == "Order State Event Counts"
    assert dict(zip(figure.data[0].kwargs["x"], figure.data[0].kwargs["y"], strict=False)) == {
        "FILLED": 2,
        "UNKNOWN": 1,
        "CANCELLED": 1,
    }


def test_build_strategy_process_trace_frame_combines_sources_and_limits_rows(monkeypatch) -> None:
    risk_dashboard = _load_module(monkeypatch)
    orders = pd.DataFrame(
        {
            "created_at": pd.to_datetime(["2026-03-22T00:00:01Z"]),
            "symbol": ["BTC/USDT"],
            "side": ["BUY"],
            "status": ["FILLED"],
        }
    )
    risk = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-03-22T00:00:03Z"]),
            "reason": ["STOP_HIT"],
        }
    )
    heartbeats = pd.DataFrame(
        {
            "heartbeat_time": pd.to_datetime(["2026-03-22T00:00:02Z"]),
            "status": ["OK"],
        }
    )
    order_states = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-03-22T00:00:04Z"]),
            "symbol": ["BTC/USDT"],
            "state": ["ACKED"],
            "message": ["accepted"],
        }
    )

    trace = risk_dashboard.build_strategy_process_trace_frame(
        df_orders=orders,
        df_risk=risk,
        df_hb=heartbeats,
        df_order_states=order_states,
    )

    assert trace["event_type"].tolist() == ["order_state", "risk", "heartbeat", "order"]
    assert trace.iloc[0]["event_detail"] == "accepted"
    assert trace.iloc[-1]["event_detail"] == "BUY"
