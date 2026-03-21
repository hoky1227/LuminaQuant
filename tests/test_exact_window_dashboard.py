from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd


class _FakeContainer:
    def __init__(
        self,
        calls: list[tuple[Any, ...]],
        *,
        name: str = "root",
        selectbox_values: dict[str, Any] | None = None,
    ) -> None:
        self._calls = calls
        self._name = name
        self._selectbox_values = selectbox_values or {}

    def __enter__(self) -> _FakeContainer:
        self._calls.append(("enter", self._name))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._calls.append(("exit", self._name))
        return False

    def _record(self, kind: str, *payload: Any) -> None:
        self._calls.append((kind, *payload))

    def header(self, value: str) -> None:
        self._record("header", value)

    def title(self, value: str) -> None:
        self._record("title", value)

    def subheader(self, value: str) -> None:
        self._record("subheader", value)

    def caption(self, value: str) -> None:
        self._record("caption", value)

    def warning(self, value: str) -> None:
        self._record("warning", value)

    def info(self, value: str) -> None:
        self._record("info", value)

    def write(self, value: Any) -> None:
        self._record("write", value)

    def markdown(self, value: str, **_: Any) -> None:
        self._record("markdown", value)

    def json(self, value: Any, **_: Any) -> None:
        self._record("json", value)

    def metric(self, label: str, value: Any, delta: Any = None) -> None:
        self._record("metric", label, value, delta)

    def dataframe(self, data: Any, **_: Any) -> None:
        if isinstance(data, pd.DataFrame):
            self._record("dataframe", list(data.columns), len(data))
        else:
            self._record("dataframe", type(data).__name__, None)

    def line_chart(self, data: Any, **_: Any) -> None:
        shape = tuple(data.shape) if isinstance(data, pd.DataFrame) else None
        self._record("line_chart", shape)

    def bar_chart(self, data: Any, **_: Any) -> None:
        shape = tuple(data.shape) if isinstance(data, pd.DataFrame) else None
        self._record("bar_chart", shape)

    def plotly_chart(self, figure: Any, **_: Any) -> None:
        self._record("plotly_chart", type(figure).__name__)

    def columns(self, spec: int | tuple[Any, ...] | list[Any]) -> list[_FakeContainer]:
        count = int(spec) if isinstance(spec, int) else len(spec)
        self._record("columns", count)
        return [
            _FakeContainer(
                self._calls,
                name=f"column[{idx}]",
                selectbox_values=self._selectbox_values,
            )
            for idx in range(count)
        ]

    def tabs(self, labels: list[str]) -> list[_FakeContainer]:
        self._record("tabs", list(labels))
        return [
            _FakeContainer(
                self._calls,
                name=f"tab[{label}]",
                selectbox_values=self._selectbox_values,
            )
            for label in labels
        ]

    def expander(self, label: str, *, expanded: bool = False) -> _FakeContainer:
        self._record("expander", label, expanded)
        return _FakeContainer(
            self._calls,
            name=f"expander[{label}]",
            selectbox_values=self._selectbox_values,
        )

    def selectbox(self, label: str, options: list[str], index: int = 0, **_: Any) -> str:
        resolved = list(options)
        self._record("selectbox", label, resolved, index)
        override = self._selectbox_values.get(label)
        return override if override in resolved else resolved[index]


class _FakeStreamlit(_FakeContainer):
    def __init__(self, *, selectbox_values: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        shared_selectbox_values = selectbox_values or {}
        super().__init__(self.calls, name="root", selectbox_values=shared_selectbox_values)
        self.sidebar = _FakeContainer(
            self.calls,
            name="sidebar",
            selectbox_values=shared_selectbox_values,
        )

    def set_page_config(self, **kwargs: Any) -> None:
        self._record("set_page_config", kwargs)


def _package(name: str, path: str | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [path]
    return module


def _load_dashboard_module(monkeypatch) -> Any:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "apps" / "dashboard" / "exact_window_suite.py"

    streamlit_module = types.ModuleType("streamlit")
    plotly_module = _package("plotly")
    plotly_express = types.ModuleType("plotly.express")
    plotly_graph_objects = types.ModuleType("plotly.graph_objects")
    plotly_module.express = plotly_express
    plotly_module.graph_objects = plotly_graph_objects

    apps_module = _package("apps", str(root / "apps"))
    dashboard_module = _package("apps.dashboard", str(root / "apps" / "dashboard"))
    services_module = _package("apps.dashboard.services", str(root / "apps" / "dashboard" / "services"))
    service_exact_window = types.ModuleType("apps.dashboard.services.exact_window")
    service_exact_window.load_exact_window_bundle = lambda *_args, **_kwargs: {}

    lumina_quant_module = _package("lumina_quant", str(root / "src" / "lumina_quant"))
    eval_package = _package("lumina_quant.eval", str(root / "src" / "lumina_quant" / "eval"))
    eval_module = types.ModuleType("lumina_quant.eval.exact_window_suite")
    eval_module._metrics_daily = lambda frame: frame

    for name, module in {
        "streamlit": streamlit_module,
        "plotly": plotly_module,
        "plotly.express": plotly_express,
        "plotly.graph_objects": plotly_graph_objects,
        "apps": apps_module,
        "apps.dashboard": dashboard_module,
        "apps.dashboard.services": services_module,
        "apps.dashboard.services.exact_window": service_exact_window,
        "lumina_quant": lumina_quant_module,
        "lumina_quant.eval": eval_package,
        "lumina_quant.eval.exact_window_suite": eval_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("dashboard_exact_window_renderer_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load exact-window dashboard module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_exact_window_dashboard_recovery_mode_surfaces_archive_cockpit(monkeypatch) -> None:
    module = _load_dashboard_module(monkeypatch)
    fake_st = _FakeStreamlit()
    module.st = fake_st
    module.load_exact_window_bundle = lambda: {
        "decision": {},
        "summary": {
            "evaluated_count": 12,
            "promoted_count": 2,
            "execution_profile": {"requested_timeframes": ["1m", "5m"]},
            "coverage": [{"symbol": "BTC/USDT", "coverage_start": "2026-03-01", "coverage_end": "2026-03-08", "full_start_coverage": True, "requested_oos_end": "2026-03-08"}],
        },
        "memory_evidence": {"peak_rss_mib": 321.5},
        "details": [
            {
                "candidate_id": "c1",
                "strategy_timeframe": "1m",
                "family": "pairs",
                "strategy_class": "PairTrading",
                "name": "btc-eth-pair",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "val": {"return": 0.11, "sharpe": 1.2},
                "oos": {"return": 0.13, "sharpe": 1.4},
                "hard_reject_reasons": {"gate": 1},
            }
        ],
        "followup_status": {
            "archived_stage": {
                "generated_at": "2026-03-08T00:00:00Z",
                "status": "completed",
                "best_row": {"name": "archived", "strategy_class": "Carry", "oos": {"return": 0.05, "sharpe": 0.8}},
                "memory_evidence": {"peak_rss_mib": 111.0},
            },
            "deployment_combo_latest": {"scenario_id": "saved-combo"},
            "deployment_scenarios_latest": {"scenario_count": 3},
            "backtest_log_archive_latest": {"archive": "ready"},
        },
        "registry": [{"run_id": "canonical-run", "status": "completed", "requested_timeframes": ["1m"], "updated_at_utc": "2026-03-08T00:00:00Z"}],
        "recovered_registry": [{"run_id": "recovered-run", "status": "completed", "updated_at_utc": "2026-03-07T00:00:00Z"}],
        "pipeline_manifest": {
            "families": [
                {
                    "family_id": "pairs",
                    "execution_style": "spread",
                    "target_timeframes": ["1m"],
                    "target_universe": ["crypto"],
                    "preferred_metrics": ["sharpe"],
                    "rationale": "pair thesis",
                }
            ]
        },
        "warnings": ["recovered advisory archive"],
        "paths": {"decision": "missing"},
        "followup_status_root": "/tmp/followups",
    }

    module.render_exact_window_dashboard(standalone=False)

    assert ("header", "Exact-Window Validation Dashboard") in fake_st.calls
    assert ("warning", "recovered advisory archive") in fake_st.calls
    assert any(
        call[0] == "warning" and "decision artifacts are missing" in str(call[1])
        for call in fake_st.calls
    )
    assert (
        "tabs",
        ["Saved Metrics", "Candidates", "Registry", "Pipeline Thesis", "Diagnostics"],
    ) in fake_st.calls
    assert ("metric", "Evaluated", "12", None) in fake_st.calls
    assert any(call[0] == "dataframe" and "source" in call[1] and call[2] == 2 for call in fake_st.calls)


def test_render_exact_window_dashboard_primary_mode_builds_sidebar_and_tabs(monkeypatch) -> None:
    module = _load_dashboard_module(monkeypatch)
    fake_st = _FakeStreamlit()
    module.st = fake_st
    module._oos_scatter_figure = lambda frame: None
    module._metric_heatmap_figure = lambda frame: None
    module._rss_bar_figure = lambda frame: None
    module._window_timeline_figure = lambda summary: None
    module._coverage_timeline_figure = lambda frame: None
    module._chart_frame = lambda best, field: pd.DataFrame()
    module._portfolio_chart_frame = lambda summary, field: pd.DataFrame()
    module._monthly_hurdle_frame = lambda best: pd.DataFrame()
    module.load_exact_window_bundle = lambda: {
        "decision": {
            "generated_at": "2026-03-08T00:00:00Z",
            "promoted_total": 1,
            "candidate_pool_total": 2,
            "total_evaluated": 4,
            "max_peak_rss_mib": 512.0,
            "next_action": "review",
            "timeframe_rows": [
                {
                    "timeframe": "1m",
                    "evaluated_count": 4,
                    "candidate_pool_strategy_count": 2,
                    "btc_beating_strategy_count": 1,
                    "memory_evidence": {"peak_rss_mib": 128.0},
                    "best_row": {
                        "name": "alpha",
                        "strategy_class": "PairTrading",
                        "symbols": ["BTC/USDT", "ETH/USDT"],
                        "promoted": True,
                        "candidate_pool_eligible": True,
                        "validation_score": 1.1,
                        "timeframe_selection_score": 0.9,
                        "rejection_reasons": [],
                        "hard_reject_reasons": {},
                        "params": {"lookback": 20},
                        "metadata": {"stage": "exact"},
                        "train": {"return": 0.10, "sharpe": 1.0},
                        "val": {"return": 0.11, "sharpe": 1.1},
                        "oos": {"return": 0.12, "sharpe": 1.2, "sortino": 1.3, "calmar": 1.0, "max_drawdown": -0.05, "trade_count": 10, "pbo": 0.2},
                        "return_streams": {"train": [], "val": [], "oos": []},
                    },
                }
            ],
        },
        "summary": {
            "execution_profile": {
                "requested_symbols": ["BTC/USDT", "ETH/USDT"],
                "requested_timeframes": ["1m"],
                "allow_metals": False,
                "custom_windows": False,
            },
            "eligible_symbols": ["BTC/USDT", "ETH/USDT"],
            "portfolio": {
                "construction_basis": "strict",
                "oos": {"return": 0.08, "sharpe": 1.05, "pbo": 0.21},
                "weights": [{"name": "alpha", "weight": 1.0}],
            },
            "coverage": [{"symbol": "BTC/USDT", "coverage_start": "2026-03-01", "coverage_end": "2026-03-08", "full_start_coverage": True, "requested_oos_end": "2026-03-08"}],
            "windows": {},
        },
        "memory_evidence": {"status": "completed"},
        "details": [
            {
                "candidate_id": "c1",
                "strategy_timeframe": "1m",
                "family": "pairs",
                "strategy_class": "PairTrading",
                "name": "alpha",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "val": {"return": 0.11, "sharpe": 1.1},
                "oos": {"return": 0.12, "sharpe": 1.2, "sortino": 1.3, "calmar": 1.0, "max_drawdown": -0.05, "trade_count": 10, "turnover": 0.3, "win_rate": 0.55, "avg_trade": 0.01, "pbo": 0.2},
                "hard_reject_reasons": {},
            }
        ],
        "followup_status": {},
        "warnings": [],
    }

    module.render_exact_window_dashboard(standalone=False)

    assert ("selectbox", "Timeframe", ["1m"], 0) in fake_st.calls
    assert (
        "selectbox",
        "Leaderboard focus",
        ["overall", "selected timeframe", "mixed assets", "metals"],
        0,
    ) in fake_st.calls
    assert (
        "tabs",
        [
            "Overview",
            "Deployment",
            "Leaderboards",
            "Time-Series",
            "Split Metrics",
            "Portfolio",
            "Monthly Hurdles",
            "Universe & Metals",
            "Follow-up Runs",
            "Run Registry",
            "Reject Reasons",
            "Diagnostics",
        ],
    ) in fake_st.calls
    assert ("subheader", "Deployment / Portfolio Candidate Panel") in fake_st.calls


def test_render_exact_window_dashboard_prefers_30m_default_and_metals_filter(monkeypatch) -> None:
    module = _load_dashboard_module(monkeypatch)
    fake_st = _FakeStreamlit(selectbox_values={"Leaderboard focus": "metals"})
    module.st = fake_st
    captured: dict[str, Any] = {}
    module._render_exact_window_sidebar_summary = lambda context, selection: None
    module._render_exact_window_primary_summary = lambda context, selection: None
    module._render_exact_window_deployment_panel = lambda selection: None
    module._render_exact_window_visual_cockpit = lambda context, selection: None
    module._render_exact_window_control_strip = lambda context, selection: None
    module._render_exact_window_timeframe_overview = lambda context, selection: None
    module._render_exact_window_candidate_analysis = lambda context, selection: captured.setdefault("selection", selection)
    module._render_exact_window_selected_timeframe_summary = lambda selection: None
    module._render_exact_window_selected_timeframe_tabs = lambda context, selection: None
    module.load_exact_window_bundle = lambda: {
        "decision": {
            "generated_at": "2026-03-08T00:00:00Z",
            "promoted_total": 1,
            "candidate_pool_total": 3,
            "total_evaluated": 6,
            "max_peak_rss_mib": 512.0,
            "next_action": "review",
            "timeframe_rows": [
                {
                    "timeframe": "1m",
                    "evaluated_count": 3,
                    "candidate_pool_strategy_count": 1,
                    "btc_beating_strategy_count": 1,
                    "memory_evidence": {"peak_rss_mib": 128.0},
                    "best_row": {
                        "name": "fast",
                        "strategy_class": "PairTrading",
                        "symbols": ["BTC/USDT", "ETH/USDT"],
                        "promoted": True,
                        "candidate_pool_eligible": True,
                        "validation_score": 1.0,
                        "timeframe_selection_score": 0.7,
                        "train": {"return": 0.10, "sharpe": 1.0},
                        "val": {"return": 0.11, "sharpe": 1.1},
                        "oos": {"return": 0.12, "sharpe": 1.2},
                    },
                },
                {
                    "timeframe": "30m",
                    "evaluated_count": 3,
                    "candidate_pool_strategy_count": 2,
                    "btc_beating_strategy_count": 1,
                    "memory_evidence": {"peak_rss_mib": 96.0},
                    "best_row": {
                        "name": "slow",
                        "strategy_class": "Carry",
                        "symbols": ["XAU/USDT"],
                        "promoted": True,
                        "candidate_pool_eligible": True,
                        "validation_score": 1.2,
                        "timeframe_selection_score": 0.9,
                        "train": {"return": 0.05, "sharpe": 0.8},
                        "val": {"return": 0.06, "sharpe": 0.9},
                        "oos": {"return": 0.07, "sharpe": 1.0},
                    },
                },
            ],
        },
        "summary": {
            "execution_profile": {
                "requested_symbols": ["BTC/USDT", "ETH/USDT", "XAU/USDT"],
                "requested_timeframes": ["1m", "30m"],
                "allow_metals": True,
                "custom_windows": False,
            },
            "eligible_symbols": ["BTC/USDT", "ETH/USDT", "XAU/USDT"],
            "portfolio": {"construction_basis": "strict", "oos": {"return": 0.08, "sharpe": 1.05, "pbo": 0.21}},
        },
        "details": [
            {
                "candidate_id": "crypto",
                "strategy_timeframe": "1m",
                "family": "pairs",
                "strategy_class": "PairTrading",
                "name": "crypto-basket",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "val": {"return": 0.11, "sharpe": 1.1},
                "oos": {"return": 0.12, "sharpe": 1.2},
                "hard_reject_reasons": {},
            },
            {
                "candidate_id": "metal",
                "strategy_timeframe": "30m",
                "family": "carry",
                "strategy_class": "Carry",
                "name": "metal-single",
                "symbols": ["XAU/USDT"],
                "val": {"return": 0.05, "sharpe": 0.8},
                "oos": {"return": 0.07, "sharpe": 1.0},
                "hard_reject_reasons": {},
            },
        ],
        "followup_status": {},
        "warnings": [],
    }

    module.render_exact_window_dashboard(standalone=False)

    selection = captured["selection"]
    assert selection.selected_timeframe == "30m"
    assert len(selection.candidate_scope) == 1
    assert selection.candidate_scope.iloc[0]["name"] == "metal-single"
    assert ("selectbox", "Timeframe", ["1m", "30m"], 1) in fake_st.calls
