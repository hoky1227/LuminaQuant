from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


class _StopSignal(RuntimeError):
    pass


class _FakeSidebar:
    def __init__(self, calls: list[tuple[Any, ...]], *, selected_view: str) -> None:
        self._calls = calls
        self._selected_view = selected_view

    def radio(self, label: str, options: list[str], index: int = 0, **_: Any) -> str:
        self._calls.append(("radio", label, list(options), index))
        return self._selected_view


class _CacheDecorator:
    def __call__(self, func=None, **_: Any):
        if func is None:
            return self
        return func

    def clear(self) -> None:
        return None


class _FakeStreamlit(types.ModuleType):
    def __init__(self, *, selected_view: str) -> None:
        super().__init__("streamlit")
        self.calls: list[tuple[Any, ...]] = []
        self.sidebar = _FakeSidebar(self.calls, selected_view=selected_view)
        self.cache_data = _CacheDecorator()
        self.session_state: dict[str, Any] = {}

    def set_page_config(self, **kwargs: Any) -> None:
        self.calls.append(("set_page_config", kwargs))

    def title(self, value: str) -> None:
        self.calls.append(("title", value))

    def caption(self, value: str) -> None:
        self.calls.append(("caption", value))

    def stop(self) -> None:
        self.calls.append(("stop",))
        raise _StopSignal()


class _RenderExactWindowStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, standalone: bool) -> None:
        self.calls.append({"standalone": standalone})


class _DummyConfig:
    POSTGRES_DSN = ""
    MARKET_DATA_EXCHANGE = "binance"
    TIMEFRAME = "1m"
    INITIAL_CAPITAL = 1000.0
    SYMBOLS = ["BTC/USDT"]


class _DummyBacktestConfig:
    LEVERAGE = 1


class _DummyOptimizationConfig:
    N_TRIALS = 10


class _Frame:
    def __init__(self, *, empty: bool) -> None:
        self.empty = empty


class _HelperExpander:
    def __init__(self, calls: list[tuple[Any, ...]], label: str) -> None:
        self._calls = calls
        self._label = label

    def __enter__(self) -> _HelperExpander:
        self._calls.append(("expander_enter", self._label))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._calls.append(("expander_exit", self._label))
        return False


class _HelperStreamlit:
    def __init__(self, *, button_result: bool = False) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._button_result = button_result

    def button(self, label: str, **kwargs: Any) -> bool:
        self.calls.append(("button", label, kwargs))
        return self._button_result

    def caption(self, value: str) -> None:
        self.calls.append(("caption", value))

    def dataframe(self, value: Any, **kwargs: Any) -> None:
        self.calls.append(("dataframe", value, kwargs))

    def download_button(self, **kwargs: Any) -> None:
        self.calls.append(("download_button", kwargs))

    def expander(self, label: str) -> _HelperExpander:
        self.calls.append(("expander", label))
        return _HelperExpander(self.calls, label)

    def json(self, value: Any) -> None:
        self.calls.append(("json", value))

    def subheader(self, value: str) -> None:
        self.calls.append(("subheader", value))

    def success(self, value: str) -> None:
        self.calls.append(("success", value))

    def warning(self, value: str) -> None:
        self.calls.append(("warning", value))


class _CacheRecorder:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


class _GhostCleanupStreamlit(_HelperStreamlit):
    def __init__(self, *, button_results: list[bool]) -> None:
        super().__init__()
        self._button_results = list(button_results)
        self.cache_data = _CacheRecorder()
        self.session_state: dict[str, Any] = {}

    def columns(self, count: int):
        self.calls.append(("columns", count))
        return tuple(self for _ in range(count))

    def number_input(self, label: str, **kwargs: Any) -> Any:
        self.calls.append(("number_input", label, kwargs))
        return kwargs.get("value")

    def selectbox(self, label: str, options: list[str], index: int = 0, **kwargs: Any) -> str:
        self.calls.append(("selectbox", label, list(options), index, kwargs))
        return options[index]

    def button(self, label: str, **kwargs: Any) -> bool:
        self.calls.append(("button", label, kwargs))
        return self._button_results.pop(0) if self._button_results else False

    def error(self, value: str) -> None:
        self.calls.append(("error", value))

    def info(self, value: str) -> None:
        self.calls.append(("info", value))

    def text_area(self, label: str, value: str, **kwargs: Any) -> None:
        self.calls.append(("text_area", label, value, kwargs))



def _package(name: str, path: str | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [path]
    return module



def _load_dashboard_app(monkeypatch) -> tuple[Any, _FakeStreamlit, _RenderExactWindowStub]:
    root = Path(__file__).resolve().parents[1]
    module_path = root / "apps" / "dashboard" / "app.py"

    fake_st = _FakeStreamlit(selected_view="Exact-Window Suite")
    render_exact_window = _RenderExactWindowStub()

    graph_objects_module = types.ModuleType("plotly.graph_objects")
    subplots_module = types.ModuleType("plotly.subplots")
    subplots_module.make_subplots = lambda *args, **kwargs: None
    plotly_module = _package("plotly")
    plotly_module.graph_objects = graph_objects_module
    plotly_module.subplots = subplots_module

    numpy_module = types.ModuleType("numpy")
    pandas_module = types.ModuleType("pandas")
    pandas_module.DataFrame = object
    pandas_module.Timestamp = object
    pandas_module.Timedelta = object

    config_module = types.ModuleType("lumina_quant.config")
    config_module.BaseConfig = _DummyConfig
    config_module.BacktestConfig = _DummyBacktestConfig
    config_module.OptimizationConfig = _DummyOptimizationConfig

    market_data_module = types.ModuleType("lumina_quant.market_data")
    market_data_module.normalize_symbol = lambda value: value
    market_data_module.normalize_timeframe_token = lambda value: value
    market_data_module.timeframe_to_milliseconds = lambda value: 60_000

    performance_module = types.ModuleType("lumina_quant.utils.performance")
    for name in [
        "create_alpha_beta",
        "create_annualized_volatility",
        "create_cagr",
        "create_calmar_ratio",
        "create_drawdowns",
        "create_information_ratio",
        "create_sharpe_ratio",
        "create_sortino_ratio",
    ]:
        setattr(performance_module, name, lambda *args, **kwargs: 0.0)

    strategy_registry_module = types.ModuleType("lumina_quant.strategies.registry")
    strategy_registry_module.get_strategy_names = lambda: ["RsiStrategy"]

    apps_module = _package("apps", str(root / "apps"))
    dashboard_module = _package("apps.dashboard", str(root / "apps" / "dashboard"))
    exact_window_module = types.ModuleType("apps.dashboard.exact_window_suite")
    exact_window_module.render_exact_window_dashboard = render_exact_window

    lumina_quant_module = _package("lumina_quant", str(root / "src" / "lumina_quant"))
    lumina_utils_module = _package("lumina_quant.utils", str(root / "src" / "lumina_quant" / "utils"))
    lumina_strategies_module = _package("lumina_quant.strategies", str(root / "src" / "lumina_quant" / "strategies"))
    lumina_strategies_module.__file__ = str((root / "src" / "lumina_quant" / "strategies" / "__init__.py").resolve())

    for name, module in {
        "streamlit": fake_st,
        "plotly": plotly_module,
        "plotly.graph_objects": graph_objects_module,
        "plotly.subplots": subplots_module,
        "numpy": numpy_module,
        "pandas": pandas_module,
        "apps": apps_module,
        "apps.dashboard": dashboard_module,
        "apps.dashboard.exact_window_suite": exact_window_module,
        "lumina_quant": lumina_quant_module,
        "lumina_quant.config": config_module,
        "lumina_quant.market_data": market_data_module,
        "lumina_quant.utils": lumina_utils_module,
        "lumina_quant.utils.performance": performance_module,
        "lumina_quant.strategies": lumina_strategies_module,
        "lumina_quant.strategies.registry": strategy_registry_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("dashboard_app_view_switch_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load dashboard app module")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
    except _StopSignal:
        pass
    return module, fake_st, render_exact_window



def test_dashboard_app_routes_exact_window_view_before_main_dashboard(monkeypatch) -> None:
    module, fake_st, render_exact_window = _load_dashboard_app(monkeypatch)

    assert ("set_page_config", {"layout": "wide", "page_title": "LuminaQuant Dashboard"}) in fake_st.calls
    assert ("title", "LuminaQuant: Full Trading Intelligence") in fake_st.calls
    assert (
        "radio",
        "Dashboard View",
        ["Main Dashboard", "Exact-Window Suite"],
        0,
    ) in fake_st.calls
    assert any("exact-window research view" in call[1] for call in fake_st.calls if call[0] == "caption")
    assert render_exact_window.calls == [{"standalone": False}]
    assert ("stop",) in fake_st.calls
    assert module._DASHBOARD_VIEW_OPTIONS == ("Main Dashboard", "Exact-Window Suite")



def test_dashboard_app_main_calls_main_dashboard_renderer_when_not_switched(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    fake_st = _FakeStreamlit(selected_view="Main Dashboard")
    render_exact_window = _RenderExactWindowStub()
    render_main_dashboard_calls: list[str] = []

    module.st = fake_st
    module.render_exact_window_dashboard = render_exact_window
    module.render_main_dashboard = lambda: render_main_dashboard_calls.append("rendered")

    module.main()

    assert render_main_dashboard_calls == ["rendered"]
    assert render_exact_window.calls == []
    assert module._route_dashboard_view() == "Main Dashboard"


def test_dashboard_app_main_stops_before_main_dashboard_on_exact_window_view(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    fake_st = _FakeStreamlit(selected_view="Exact-Window Suite")
    render_exact_window = _RenderExactWindowStub()
    render_main_dashboard_calls: list[str] = []

    module.st = fake_st
    module.render_exact_window_dashboard = render_exact_window
    module.render_main_dashboard = lambda: render_main_dashboard_calls.append("rendered")

    try:
        module.main()
    except _StopSignal:
        pass
    else:
        raise AssertionError("Expected dashboard routing to stop on exact-window view")

    assert render_main_dashboard_calls == []
    assert render_exact_window.calls == [{"standalone": False}]


def test_render_snapshot_report_section_builds_preview_without_saving(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _HelperStreamlit(button_result=False)
    captured: dict[str, Any] = {}
    save_calls: list[dict[str, Any]] = []
    payload = {"monthly_returns": {}, "mt5_summary": []}

    module.st = helper_st
    monkeypatch.setattr(
        module,
        "build_report_payload",
        lambda *args, **kwargs: captured.update(kwargs) or payload,
    )
    monkeypatch.setattr(
        module,
        "save_report_snapshot",
        lambda report_payload: save_calls.append(report_payload) or ("report.json", "report.md", "#"),
    )

    module._render_snapshot_report_section(
        summary={"bars": 10},
        performance={"sharpe": 1.2},
        active_run_id="run-1",
        resolved_source="postgres",
        strategy_name="RsiStrategy",
        period_preset="30d",
        df_equity=_Frame(empty=True),
        trade_analytics=_Frame(empty=True),
        df_risk=_Frame(empty=True),
        df_hb=_Frame(empty=True),
        runner_initial_capital=1500.0,
        runner_leverage=3,
        runner_symbols=["BTC/USDT", "ETH/USDT"],
        runner_timeframe="5m",
        runner_data_source="postgres",
        runner_timeout_sec=45,
        strategy_params={"lookback": 12},
        mirror_snapshot={"wins": 1},
        mirror_balance_equity=_Frame(empty=True),
    )

    assert captured["runtime_overrides"] == {
        "initial_capital": 1500.0,
        "backtest_leverage": 3,
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "5m",
        "runner_data_source": "postgres",
        "runner_timeout_sec": 45,
    }
    assert captured["strategy_params"] == {"lookback": 12}
    assert captured["mirror_snapshot"] == {"wins": 1}
    assert captured["balance_equity_series"] == []
    assert ("subheader", "Current Snapshot Preview") in helper_st.calls
    assert ("json", payload) in helper_st.calls
    assert save_calls == []


def test_render_raw_data_tab_loads_workflow_jobs_once_and_preserves_section_order(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _HelperStreamlit()
    captured: dict[str, Any] = {}
    workflow_jobs_frame = object()

    module.st = helper_st
    monkeypatch.setattr(
        module,
        "load_workflow_jobs",
        lambda db_path, refresh_counter=0: captured.update(
            {"db_path": db_path, "refresh_counter": refresh_counter}
        )
        or workflow_jobs_frame,
    )

    runs_df = object()
    equity_df = object()
    fills_df = object()
    orders_df = object()
    risk_df = object()
    hb_df = object()
    order_states_df = object()
    market_df = object()
    optimize_df = object()

    module._render_raw_data_tab(
        active_run_id="run-raw",
        resolved_source="csv",
        market_symbol="BTC/USDT",
        market_timeframe="1m",
        market_exchange="binance",
        runs_df=runs_df,
        df_equity=equity_df,
        trade_analytics=fills_df,
        df_orders=orders_df,
        df_risk=risk_df,
        df_hb=hb_df,
        df_order_states=order_states_df,
        df_market=market_df,
        df_optimize=optimize_df,
        db_path="postgres://lumina",
        refresh_counter=7,
    )

    assert captured == {"db_path": "postgres://lumina", "refresh_counter": 7}
    assert helper_st.calls[0] == (
        "caption",
        "Run: run-raw | Source: csv | Market: BTC/USDT 1m (binance)",
    )
    assert [call[1] for call in helper_st.calls if call[0] == "expander"] == [
        "Runs",
        "Equity",
        "Fills (enriched)",
        "Orders",
        "Risk Events",
        "Heartbeats",
        "Order State Events",
        "Market OHLCV",
        "Optimization Results",
        "Workflow Jobs",
    ]
    assert [call[1] for call in helper_st.calls if call[0] == "dataframe"] == [
        runs_df,
        equity_df,
        fills_df,
        orders_df,
        risk_df,
        hb_df,
        order_states_df,
        market_df,
        optimize_df,
        workflow_jobs_frame,
    ]


def test_render_missing_equity_warning_only_when_equity_is_empty(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _HelperStreamlit()
    module.st = helper_st

    module._render_missing_equity_warning(_Frame(empty=False), 1000.0)
    assert [call for call in helper_st.calls if call[0] == "warning"] == []

    module._render_missing_equity_warning(_Frame(empty=True), 1000.0)
    warnings = [call[1] for call in helper_st.calls if call[0] == "warning"]
    assert len(warnings) == 1
    assert "Configured initial equity is 1000.00." in warnings[0]


def test_build_backtest_job_launch_spec_keeps_runner_and_metadata_fields(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)

    spec = module._build_backtest_job_launch_spec(
        runner_data_source="postgres",
        market_db_path="/tmp/market",
        market_exchange="binance",
        runner_env_overrides={"LQ__RUNNER__TIMEFRAME": "5m"},
        strategy_name="RsiStrategy",
        backtest_run_id="run-backtest",
        strategy_params_path="/tmp/params.json",
    )

    assert spec.workflow == "backtest"
    assert spec.script_name == "run_backtest.py"
    assert spec.script_args == (
        "--data-source",
        "postgres",
        "--market-db-path",
        "/tmp/market",
        "--market-exchange",
        "binance",
        "--run-id",
        "run-backtest",
    )
    assert spec.env_overrides == {"LQ__RUNNER__TIMEFRAME": "5m"}
    assert spec.requested_mode == "backtest"
    assert spec.strategy == "RsiStrategy"
    assert spec.run_id == "run-backtest"
    assert spec.metadata == {"strategy_params_path": "/tmp/params.json"}


def test_build_optimize_job_launch_spec_includes_best_params_flag_and_int_metadata(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)

    spec = module._build_optimize_job_launch_spec(
        optimize_folds=4,
        optimize_trials=25,
        optimize_workers=3,
        runner_data_source="csv",
        market_db_path="/tmp/market",
        market_exchange="binanceusdm",
        persist_best_params=True,
        runner_env_overrides={"LQ__RUNNER__SYMBOLS": "BTC/USDT"},
        strategy_name="BreakoutStrategy",
        optimize_run_id="run-opt",
    )

    assert spec.workflow == "optimize"
    assert spec.script_name == "optimize.py"
    assert spec.script_args[-1] == "--save-best-params"
    assert spec.requested_mode == "optimize"
    assert spec.strategy == "BreakoutStrategy"
    assert spec.run_id == "run-opt"
    assert spec.metadata == {"folds": 4, "n_trials": 25, "max_workers": 3}
    assert spec.env_overrides == {"LQ__RUNNER__SYMBOLS": "BTC/USDT"}


def test_build_live_job_launch_spec_adds_websocket_and_real_mode_controls(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)

    spec = module._build_live_job_launch_spec(
        runner_env_overrides={"LQ__RUNNER__TIMEFRAME": "1m"},
        live_mode="real",
        market_exchange="BINANCE",
        runner_leverage=7,
        live_runner_kind="WebSocket (run_live_ws.py)",
        live_strategy_name="TrendStrategy",
        live_run_id="run-live",
        stop_file="/tmp/run-live.stop",
    )

    assert spec.workflow == "live_ws"
    assert spec.script_name == "run_live_ws.py"
    assert spec.script_args == (
        "--strategy",
        "TrendStrategy",
        "--run-id",
        "run-live",
        "--stop-file",
        "/tmp/run-live.stop",
        "--enable-live-real",
    )
    assert spec.env_overrides == {
        "LQ__RUNNER__TIMEFRAME": "1m",
        "LQ__LIVE__MODE": "real",
        "LQ__LIVE__EXCHANGE__NAME": "binance",
        "LQ__LIVE__EXCHANGE__LEVERAGE": "7",
        "LUMINA_ENABLE_LIVE_REAL": "true",
    }
    assert spec.requested_mode == "real"
    assert spec.strategy == "TrendStrategy"
    assert spec.run_id == "run-live"
    assert spec.stop_file == "/tmp/run-live.stop"
    assert spec.metadata == {"runner_kind": "WebSocket (run_live_ws.py)"}


def test_render_ghost_cleanup_section_records_dry_run_and_preserves_cache(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[True, False])
    captured: dict[str, Any] = {}

    module.st = helper_st
    monkeypatch.setattr(
        module,
        "_run_ghost_cleanup_script",
        lambda **kwargs: captured.update(kwargs)
        or {
            "ok": True,
            "elapsed_sec": 1.25,
            "command": ["python", "ghost-cleanup"],
            "payload": {"closed_runs": 2},
            "output": "done",
        },
    )

    module._render_ghost_cleanup_section(db_path="postgres://lumina", run_stale_sec=120)

    assert captured == {
        "dsn": "postgres://lumina",
        "stale_sec": 300,
        "startup_grace_sec": 90,
        "close_status": "STOPPED",
        "force_kill_stop_requested_after_sec": 0,
        "apply_changes": False,
    }
    assert helper_st.cache_data.clear_calls == 0
    assert helper_st.session_state["ghost_cleanup_last_result"]["mode"] == "dry_run"
    assert ("success", "Ghost cleanup dry_run completed in 1.25s") in helper_st.calls
    assert ("json", {"closed_runs": 2}) in helper_st.calls
    assert any(call[0] == "text_area" and call[2] == "done" for call in helper_st.calls)
