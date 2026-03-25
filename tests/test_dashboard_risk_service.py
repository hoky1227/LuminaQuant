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


class _FakeStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def dataframe(self, value: Any, **kwargs: Any) -> None:
        self.calls.append(("dataframe", value, kwargs))

    def info(self, value: str) -> None:
        self.calls.append(("info", value))

    def metric(self, label: str, value: str) -> None:
        self.calls.append(("metric", label, value))

    def plotly_chart(self, figure: Any, **kwargs: Any) -> None:
        self.calls.append(("plotly_chart", figure, kwargs))

    def subheader(self, value: str) -> None:
        self.calls.append(("subheader", value))


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


def test_render_risk_health_section_renders_figures_metrics_and_trace(monkeypatch) -> None:
    risk_dashboard = _load_module(monkeypatch)
    fake_st = _FakeStreamlit()
    trace_frame = pd.DataFrame([{"event_type": "risk"}])
    recorded_hb_inputs: list[pd.DataFrame] = []

    monkeypatch.setattr(risk_dashboard, "build_risk_reason_figure", lambda frame: ("risk", frame))
    monkeypatch.setattr(
        risk_dashboard,
        "prepare_heartbeat_interval_frame",
        lambda frame: recorded_hb_inputs.append(frame) or (pd.DataFrame({"delta_sec": [None, 5.0]}), 5.0),
    )
    monkeypatch.setattr(risk_dashboard, "build_heartbeat_interval_figure", lambda frame: ("hb", frame))
    monkeypatch.setattr(risk_dashboard, "build_order_state_figure", lambda frame: ("order-state", frame))
    monkeypatch.setattr(risk_dashboard, "build_strategy_process_trace_frame", lambda **_: trace_frame)

    risk_dashboard.render_risk_health_section(
        streamlit=fake_st,
        df_orders=pd.DataFrame([{"created_at": "2026-03-22T00:00:00Z"}]),
        df_risk=pd.DataFrame([{"reason": "STOP_HIT"}]),
        df_hb=pd.DataFrame([{"heartbeat_time": "2026-03-22T00:00:05Z"}]),
        df_order_states=pd.DataFrame([{"state": "ACKED"}]),
    )

    assert recorded_hb_inputs and list(recorded_hb_inputs[0].columns) == ["heartbeat_time"]
    assert ("metric", "Avg Heartbeat Interval (sec)", "5.00") in fake_st.calls
    assert ("subheader", "Strategy Process Trace") in fake_st.calls
    plotted = [call[1] for call in fake_st.calls if call[0] == "plotly_chart"]
    assert [item[0] for item in plotted] == ["risk", "hb", "order-state"]
    assert ("dataframe", trace_frame, {"use_container_width": True}) in fake_st.calls


def test_render_risk_health_section_reports_empty_states_without_trace(monkeypatch) -> None:
    risk_dashboard = _load_module(monkeypatch)
    fake_st = _FakeStreamlit()

    monkeypatch.setattr(risk_dashboard, "build_strategy_process_trace_frame", lambda **_: pd.DataFrame())

    risk_dashboard.render_risk_health_section(
        streamlit=fake_st,
        df_orders=pd.DataFrame(),
        df_risk=pd.DataFrame(),
        df_hb=pd.DataFrame(),
        df_order_states=pd.DataFrame(),
    )

    assert ("info", "No risk events recorded for selected run/data source.") in fake_st.calls
    assert ("info", "No heartbeats recorded for selected run/data source.") in fake_st.calls
    assert [call for call in fake_st.calls if call[0] == "plotly_chart"] == []
    assert [call for call in fake_st.calls if call[0] == "subheader"] == []
