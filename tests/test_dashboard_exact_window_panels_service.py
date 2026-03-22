from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd


class _FakeContainer:
    def __init__(self, calls: list[tuple[Any, ...]], *, name: str = "root") -> None:
        self._calls = calls
        self._name = name

    def __enter__(self) -> _FakeContainer:
        self._calls.append(("enter", self._name))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._calls.append(("exit", self._name))
        return False

    def subheader(self, value: str) -> None:
        self._calls.append(("subheader", value))

    def info(self, value: str) -> None:
        self._calls.append(("info", value))

    def plotly_chart(self, figure: Any, **_: Any) -> None:
        self._calls.append(("plotly_chart", figure))

    def markdown(self, value: str, **kwargs: Any) -> None:
        self._calls.append(("markdown", value, kwargs))

    def columns(self, spec: int | tuple[Any, ...] | list[Any]) -> list[_FakeContainer]:
        count = int(spec) if isinstance(spec, int) else len(spec)
        self._calls.append(("columns", count))
        return [_FakeContainer(self._calls, name=f"column[{idx}]") for idx in range(count)]


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "exact_window_panels.py"


def _load_module(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    calls: list[tuple[Any, ...]] = []
    container = _FakeContainer(calls)
    fake_streamlit.columns = container.columns
    fake_streamlit.subheader = container.subheader
    fake_streamlit.info = container.info
    fake_streamlit.plotly_chart = container.plotly_chart
    fake_streamlit.markdown = container.markdown
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    spec = importlib.util.spec_from_file_location("dashboard_exact_window_panels", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, container, calls


def test_render_exact_window_visual_cockpit_renders_figures_and_empty_states(monkeypatch) -> None:
    module, container, calls = _load_module(monkeypatch)
    context = types.SimpleNamespace(summary={"windows": []}, coverage_status=pd.DataFrame({"x": [1]}))
    selection = types.SimpleNamespace(summary_frame=pd.DataFrame({"a": [1]}), metric_matrix=pd.DataFrame())

    module.render_exact_window_visual_cockpit(
        context,
        selection,
        oos_scatter_figure=lambda _frame: "scatter-fig",
        metric_heatmap_figure=lambda _frame: None,
        rss_bar_figure=lambda _frame: "rss-fig",
        window_timeline_figure=lambda _summary: "window-fig",
        coverage_timeline_figure=lambda _coverage: None,
        st_module=container,
    )

    assert ("subheader", "Performance Map") in calls
    assert ("plotly_chart", "scatter-fig") in calls
    assert ("info", "No metric heatmap available.") in calls
    assert ("plotly_chart", "rss-fig") in calls
    assert ("plotly_chart", "window-fig") in calls
    assert ("info", "No coverage timeline available.") in calls


def test_render_exact_window_control_strip_uses_panel_html_and_status_chip(monkeypatch) -> None:
    module, container, calls = _load_module(monkeypatch)
    context = types.SimpleNamespace(
        bundle={"followup_status": {"metals_blocker_latest": {"reason": "missing metals", "blocked_metals": ["XAU"]}}},
        summary={"eligible_symbols": ["BTC/USDT", "ETH/USDT"]},
        decision={"max_peak_rss_mib": 512.25},
        memory_evidence={"status": "ok"},
    )
    selection = types.SimpleNamespace(
        execution_profile={
            "custom_windows": True,
            "allow_metals": False,
            "requested_timeframes": ["1m", "5m"],
            "requested_symbols": ["BTC/USDT", "XAU/USD"],
        },
        queue_rows=[{"run_id": "1"}, {"run_id": "2"}],
    )

    module.render_exact_window_control_strip(
        context,
        selection,
        panel_html=lambda title, body, chips: f"{title}|{body}|{'/'.join(chips)}",
        status_chip=lambda label, value: f"{label}:{value}",
        st_module=container,
    )

    markdown_values = [call[1] for call in calls if call[0] == "markdown"]
    assert any("Execution profile|" in value and "Requested TF:1m, 5m" in value for value in markdown_values)
    assert any("Metal / mixed-asset status|missing metals" in value for value in markdown_values)
    assert any("Memory discipline|" in value and "Decision peak:512.2 MiB" in value for value in markdown_values)
