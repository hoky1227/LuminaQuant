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

    def caption(self, value: str) -> None:
        self._calls.append(("caption", value))

    def dataframe(self, data: Any, **_: Any) -> None:
        if isinstance(data, pd.DataFrame):
            self._calls.append(("dataframe", list(data.columns), len(data)))
        else:
            self._calls.append(("dataframe", type(data).__name__, None))

    def expander(self, label: str, *, expanded: bool = False) -> _FakeContainer:
        self._calls.append(("expander", label, expanded))
        return _FakeContainer(self._calls, name=f"expander[{label}]")

    def json(self, value: Any, **_: Any) -> None:
        self._calls.append(("json", value))

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


def test_render_exact_window_timeframe_overview_renders_cards_and_frames(monkeypatch) -> None:
    module, container, calls = _load_module(monkeypatch)
    context = types.SimpleNamespace(
        timeframe_rows=[{"timeframe": "5m"}, {"timeframe": "1m"}],
        coverage_status=pd.DataFrame({"symbol": ["BTC/USDT"]}),
    )
    selection = types.SimpleNamespace(
        summary_frame=pd.DataFrame({"timeframe": ["1m"], "oos_return": [0.1]}),
        metric_matrix=pd.DataFrame({"oos_return": [0.1]}, index=["1m"]),
    )

    module.render_exact_window_timeframe_overview(
        context,
        selection,
        format_frame=lambda frame: frame,
        timeframe_card_html=lambda row: f"<card>{row['timeframe']}</card>",
        timeframe_sort_key=lambda value: (0 if value == "1m" else 1, value),
        st_module=container,
    )

    markdown_values = [call[1] for call in calls if call[0] == "markdown"]
    assert any("exact-window-section-caption" in value for value in markdown_values)
    assert any("<card>1m</card><card>5m</card>" in value for value in markdown_values)
    assert ("subheader", "All Timeframes At a Glance") in calls
    assert ("subheader", "Metric Matrix") in calls
    assert ("subheader", "Universe Coverage / Metals") in calls
    assert any(call[0] == "dataframe" and "timeframe" in call[1] for call in calls)


def test_render_exact_window_candidate_analysis_renders_leaderboards_and_triage(monkeypatch) -> None:
    module, container, calls = _load_module(monkeypatch)
    candidate_scope = pd.DataFrame(
        [
            {
                "timeframe": "1m",
                "asset_mix": "crypto",
                "strategy": "PairTrading",
                "name": "alpha",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "oos_return": 0.12,
                "oos_sharpe": 1.2,
                "oos_sortino": 1.3,
                "oos_calmar": 1.1,
                "oos_mdd": -0.05,
                "oos_pbo": 0.2,
                "val_return": 0.11,
                "val_sharpe": 1.1,
                "val_pbo": 0.25,
                "oos_trades": 10,
                "oos_turnover": 0.3,
                "oos_win_rate": 0.55,
                "oos_avg_trade": 0.01,
                "rejects": 0,
            }
        ]
    )
    context = types.SimpleNamespace(
        decision={"generated_at": "2026-03-22T00:00:00Z"},
        details_frame=pd.DataFrame({"family": ["pairs"], "asset_mix": ["crypto"]}),
        bundle={"fail_analysis": {"reason": "gate"}},
        next_iteration={"action": "rerun"},
    )
    selection = types.SimpleNamespace(candidate_scope=candidate_scope)

    module.render_exact_window_candidate_analysis(
        context,
        selection,
        candidate_pool_frame=lambda _decision: pd.DataFrame({"candidate": ["alpha"]}),
        strict_pass_frame=lambda _decision: pd.DataFrame({"name": ["alpha"]}),
        format_frame=lambda frame: frame,
        top_candidates=lambda frame, columns: frame.loc[:, columns],
        family_mix_frame=lambda _details: (
            pd.DataFrame({"family": ["pairs"], "count": [1]}),
            pd.DataFrame({"asset_mix": ["crypto"], "count": [1]}),
        ),
        fail_reason_summary=lambda _bundle: (
            pd.DataFrame({"reason": ["gate"], "count": [1]}),
            pd.DataFrame({"timeframe": ["1m"], "count": [1]}),
            pd.DataFrame({"proposal": ["rerun"]}),
        ),
        st_module=container,
    )

    assert ("subheader", "Selection / Promotion Overview") in calls
    assert ("subheader", "Candidate Leaderboards") in calls
    assert ("caption", "Top by OOS quality") in calls
    assert ("subheader", "Family / Timeframe Distribution") in calls
    assert ("subheader", "Reject Reasons") in calls
    assert ("expander", "Next Iteration Triage", True) in calls
    assert ("json", {"action": "rerun"}) in calls
