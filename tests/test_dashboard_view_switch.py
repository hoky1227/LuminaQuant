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

    def warning(self, value: str) -> None:
        self.calls.append(("warning", value))

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
    def __init__(self, *, empty: bool, size: int = 0) -> None:
        self.empty = empty
        self._size = size

    def __len__(self) -> int:
        return self._size


class _LaunchSpecRecorder:
    def __init__(self, job_id: str, sink: list[dict[str, Any]], payload: dict[str, Any]) -> None:
        self._job_id = job_id
        self._sink = sink
        self._payload = payload

    def launch(self, *, db_path: str) -> str:
        self._sink.append({"db_path": db_path, **self._payload})
        return self._job_id


class _SidebarControlsFake:
    def __init__(
        self,
        calls: list[tuple[Any, ...]],
        *,
        selectbox_results: list[str] | None = None,
        text_input_results: list[str] | None = None,
        toggle_results: list[bool] | None = None,
        slider_results: list[Any] | None = None,
        number_input_results: list[Any] | None = None,
        date_input_results: list[Any] | None = None,
    ) -> None:
        self._calls = calls
        self._selectbox_results = list(selectbox_results or [])
        self._text_input_results = list(text_input_results or [])
        self._toggle_results = list(toggle_results or [])
        self._slider_results = list(slider_results or [])
        self._number_input_results = list(number_input_results or [])
        self._date_input_results = list(date_input_results or [])

    def header(self, value: str) -> None:
        self._calls.append(("sidebar.header", value))

    def subheader(self, value: str) -> None:
        self._calls.append(("sidebar.subheader", value))

    def divider(self) -> None:
        self._calls.append(("sidebar.divider",))

    def caption(self, value: str) -> None:
        self._calls.append(("sidebar.caption", value))

    def selectbox(self, label: str, options: list[str], index: int = 0, **kwargs: Any) -> str:
        self._calls.append(("sidebar.selectbox", label, list(options), index, kwargs))
        return self._selectbox_results.pop(0) if self._selectbox_results else options[index]

    def text_input(self, label: str, value: str = "", **kwargs: Any) -> str:
        self._calls.append(("sidebar.text_input", label, value, kwargs))
        return self._text_input_results.pop(0) if self._text_input_results else value

    def toggle(self, label: str, value: bool = False, **kwargs: Any) -> bool:
        self._calls.append(("sidebar.toggle", label, value, kwargs))
        return self._toggle_results.pop(0) if self._toggle_results else value

    def slider(self, label: str, **kwargs: Any) -> Any:
        self._calls.append(("sidebar.slider", label, kwargs))
        return self._slider_results.pop(0) if self._slider_results else kwargs.get("value")

    def number_input(self, label: str, **kwargs: Any) -> Any:
        self._calls.append(("sidebar.number_input", label, kwargs))
        return self._number_input_results.pop(0) if self._number_input_results else kwargs.get("value")

    def date_input(self, label: str, value: Any = None, **kwargs: Any) -> Any:
        self._calls.append(("sidebar.date_input", label, value, kwargs))
        return self._date_input_results.pop(0) if self._date_input_results else value


class _SidebarStreamlit:
    def __init__(self, sidebar: _SidebarControlsFake) -> None:
        self.sidebar = sidebar


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
    def __init__(
        self,
        *,
        button_results: list[bool],
        selectbox_results: list[str] | None = None,
        checkbox_results: list[bool] | None = None,
        text_input_results: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._button_results = list(button_results)
        self._selectbox_results = list(selectbox_results or [])
        self._checkbox_results = list(checkbox_results or [])
        self._text_input_results = list(text_input_results or [])
        self.cache_data = _CacheRecorder()
        self.session_state: dict[str, Any] = {}

    def __enter__(self) -> _GhostCleanupStreamlit:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def columns(self, count: int):
        self.calls.append(("columns", count))
        return tuple(self for _ in range(count))

    def number_input(self, label: str, **kwargs: Any) -> Any:
        self.calls.append(("number_input", label, kwargs))
        return kwargs.get("value")

    def selectbox(self, label: str, options: list[str], index: int = 0, **kwargs: Any) -> str:
        self.calls.append(("selectbox", label, list(options), index, kwargs))
        return self._selectbox_results.pop(0) if self._selectbox_results else options[index]

    def checkbox(self, label: str, **kwargs: Any) -> bool:
        self.calls.append(("checkbox", label, kwargs))
        return self._checkbox_results.pop(0) if self._checkbox_results else False

    def button(self, label: str, **kwargs: Any) -> bool:
        self.calls.append(("button", label, kwargs))
        return self._button_results.pop(0) if self._button_results else False

    def error(self, value: str) -> None:
        self.calls.append(("error", value))

    def info(self, value: str) -> None:
        self.calls.append(("info", value))

    def metric(self, label: str, value: str) -> None:
        self.calls.append(("metric", label, value))

    def plotly_chart(self, figure: Any, **kwargs: Any) -> None:
        self.calls.append(("plotly_chart", figure, kwargs))

    def text_input(self, label: str, **kwargs: Any) -> str:
        self.calls.append(("text_input", label, kwargs))
        return self._text_input_results.pop(0) if self._text_input_results else ""

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


def test_execute_query_logs_when_fetchall_fails(monkeypatch, caplog) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    caplog.set_level("WARNING")

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query: str, params: tuple[Any, ...]) -> None:
            self.query = query
            self.params = params

        def fetchall(self):
            raise RuntimeError("fetchall failed")

    class _Conn:
        def __init__(self):
            self.cursor_obj = _Cursor()
            self.committed = False
            self.closed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

        def close(self):
            self.closed = True

    conn = _Conn()
    monkeypatch.setattr(module, "_connect_state_store", lambda _dsn: conn)

    rows = module._execute_query("postgres://lumina", "SELECT 1")

    assert rows == []
    assert conn.committed is True
    assert conn.closed is True
    assert any(
        "fell back to an empty result set" in record.message for record in caplog.records
    )


def test_count_market_rows_warns_and_returns_zero_on_query_failure(monkeypatch, caplog) -> None:
    module, fake_st, _ = _load_dashboard_app(monkeypatch)
    caplog.set_level("WARNING")
    module.st = fake_st
    monkeypatch.setattr(
        module,
        "_execute_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("query failed")),
    )

    count = module._count_market_rows("postgres://lumina")

    assert count == 0
    assert ("warning", "Unable to count market rows right now; dashboard is falling back to zero rows.") in fake_st.calls
    assert any(
        "fell back to zero" in record.message for record in caplog.records
    )


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


def test_save_report_snapshot_writes_under_dashboard_var_reports(monkeypatch, tmp_path: Path) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    monkeypatch.setattr(module, "DASHBOARD_REPORT_DIR", tmp_path / "var" / "dashboard" / "reports")
    payload = {
        "generated_at": "2026-03-22T00:00:00Z",
        "run_id": "run-1",
        "source": "postgres",
        "strategy": "RsiStrategy",
        "summary": {
            "period_start": "2026-03-01",
            "period_end": "2026-03-22",
            "bars": 10,
            "fills": 2,
            "buy_fills": 1,
            "sell_fills": 1,
            "fills_per_day": 0.5,
            "avg_qty": 1.0,
            "avg_notional": 100.0,
            "total_commission": 0.1,
            "realized_pnl": 1.5,
            "win_rate": 0.5,
            "avg_trade_return_pct": 0.2,
            "best_trade_pnl": 2.0,
            "worst_trade_pnl": -0.5,
            "gross_profit": 2.0,
            "gross_loss": -0.5,
            "profit_factor": 4.0,
            "recovery_factor": 1.25,
            "long_trades_win_pct": "1 (100.0%)",
            "short_trades_win_pct": "0 (0.0%)",
            "holding_time_min_sec": 60.0,
            "holding_time_avg_sec": 120.0,
            "holding_time_max_sec": 180.0,
            "equity_drawdown_absolute": 0.1,
            "equity_drawdown_maximal": 0.2,
            "equity_drawdown_relative_pct": 0.01,
            "balance_drawdown_absolute": 0.1,
            "balance_drawdown_maximal": 0.2,
            "balance_drawdown_relative_pct": 0.01,
            "win_streak_max": 1,
            "loss_streak_max": 1,
            "win_streak_avg": 1.0,
            "loss_streak_avg": 1.0,
            "max_consecutive_profit_amount": 2.0,
            "max_consecutive_loss_amount": -0.5,
        },
        "mirror_snapshot": {},
        "balance_equity_series": [],
    }

    json_path, md_path, markdown = module.save_report_snapshot(payload)

    assert json_path.startswith(str(tmp_path / "var" / "dashboard" / "reports"))
    assert md_path.startswith(str(tmp_path / "var" / "dashboard" / "reports"))
    assert Path(json_path).is_file()
    assert Path(md_path).is_file()
    assert "Dashboard Snapshot Report" in markdown


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


def test_render_risk_health_section_delegates_to_service_seam(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[])
    captured: dict[str, Any] = {}
    df_orders = object()
    df_risk = object()
    df_hb = object()
    df_order_states = object()

    module.st = helper_st
    monkeypatch.setattr(
        module,
        "_render_risk_health_section_data",
        lambda **kwargs: captured.update(kwargs),
    )

    module._render_risk_health_section(
        df_orders=df_orders,
        df_risk=df_risk,
        df_hb=df_hb,
        df_order_states=df_order_states,
    )

    assert captured == {
        "streamlit": helper_st,
        "df_orders": df_orders,
        "df_risk": df_risk,
        "df_hb": df_hb,
        "df_order_states": df_order_states,
    }


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
    assert spec.command == (
        "uv",
        "run",
        "lq",
        "backtest",
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
    assert spec.command[:4] == ("uv", "run", "lq", "optimize")
    assert spec.command[-1] == "--save-best-params"
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
        live_runner_kind="WebSocket (uv run lq live --transport ws)",
        live_strategy_name="TrendStrategy",
        live_run_id="run-live",
        stop_file="/tmp/run-live.stop",
    )

    assert spec.workflow == "live_ws"
    assert spec.command == (
        "uv",
        "run",
        "lq",
        "live",
        "--transport",
        "ws",
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
    assert spec.metadata == {
        "runner_kind": "WebSocket (uv run lq live --transport ws)",
        "transport": "ws",
    }


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


def test_render_workflow_jobs_section_reports_empty_state(monkeypatch) -> None:
    import numpy as real_np
    import pandas as real_pd

    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[])

    monkeypatch.setitem(sys.modules, "numpy", real_np)
    monkeypatch.setitem(sys.modules, "pandas", real_pd)
    module.st = helper_st
    module.pd = real_pd
    monkeypatch.setattr(module, "load_workflow_jobs", lambda db_path, refresh_counter=0: real_pd.DataFrame())

    module._render_workflow_jobs_section(db_path="postgres://lumina", refresh_counter=3)

    assert ("subheader", "Workflow Jobs") in helper_st.calls
    assert ("info", "No workflow jobs recorded yet.") in helper_st.calls


def test_render_workflow_jobs_section_handles_graceful_stop_and_log_tail(monkeypatch) -> None:
    import numpy as real_np
    import pandas as real_pd

    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[True, False])
    captured_stop: dict[str, Any] = {}
    update_calls: list[dict[str, Any]] = []

    workflow_jobs = real_pd.DataFrame(
        [
            {
                "job_id": "job-1",
                "started_at": "2026-03-21T00:00:00Z",
                "workflow": "live",
                "status": "RUNNING",
                "requested_mode": "paper",
                "strategy": "TrendStrategy",
                "pid": 1234,
                "run_id": "run-1",
                "exit_code": None,
                "command_json": "[\"uv\", \"run\", \"lq\", \"live\", \"--transport\", \"poll\"]",
                "stop_file": "/tmp/job-1.stop",
                "log_path": "/tmp/job-1.log",
            }
        ]
    )

    monkeypatch.setitem(sys.modules, "numpy", real_np)
    monkeypatch.setitem(sys.modules, "pandas", real_pd)
    module.st = helper_st
    module.pd = real_pd
    monkeypatch.setattr(module, "load_workflow_jobs", lambda db_path, refresh_counter=0: workflow_jobs)
    monkeypatch.setattr(
        module,
        "_request_job_stop",
        lambda db_path, stop_file: captured_stop.update({"db_path": db_path, "stop_file": stop_file})
        or True,
    )
    monkeypatch.setattr(
        module,
        "_update_workflow_job_row",
        lambda db_path, job_id, **kwargs: update_calls.append(
            {"db_path": db_path, "job_id": job_id, **kwargs}
        ),
    )
    monkeypatch.setattr(module, "_tail_text_file", lambda path, max_chars=25000: f"tail:{path}:{max_chars}")

    module._render_workflow_jobs_section(db_path="postgres://lumina", refresh_counter=7)

    assert captured_stop == {"db_path": "postgres://lumina", "stop_file": "/tmp/job-1.stop"}
    assert update_calls == [
        {"db_path": "postgres://lumina", "job_id": "job-1", "status": "STOP_REQUESTED"}
    ]
    assert helper_st.cache_data.clear_calls == 1
    assert ("success", "Stop requested for job-1") in helper_st.calls
    assert ("caption", "Log path: /tmp/job-1.log") in helper_st.calls
    assert any(
        call[0] == "text_area" and call[2] == "tail:/tmp/job-1.log:25000" for call in helper_st.calls
    )


def test_render_optimization_results_tab_reports_empty_state(monkeypatch) -> None:
    import numpy as real_np
    import pandas as real_pd

    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[])

    monkeypatch.setitem(sys.modules, "numpy", real_np)
    monkeypatch.setitem(sys.modules, "pandas", real_pd)
    module.st = helper_st
    module.pd = real_pd

    module._render_optimization_results_tab(real_pd.DataFrame())

    assert ("subheader", "Optimization Results") in helper_st.calls
    assert ("info", "No optimization_results rows found in Postgres yet.") in helper_st.calls


def test_render_optimization_results_tab_summarizes_best_row_without_scatter(monkeypatch) -> None:
    import numpy as real_np
    import pandas as real_pd

    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[])

    monkeypatch.setitem(sys.modules, "numpy", real_np)
    monkeypatch.setitem(sys.modules, "pandas", real_pd)
    module.st = helper_st
    module.pd = real_pd

    df_optimize = real_pd.DataFrame(
        [
            {
                "created_at": "2026-03-21T00:00:00Z",
                "run_id": "opt-1",
                "stage": "train",
                "sharpe": 1.25,
                "train_sharpe": None,
                "robustness_score": 0.7,
                "cagr": 0.12,
                "mdd": -0.05,
                "params": {"lookback": 14},
                "extra": {"bucket": "top"},
            }
        ]
    )

    module._render_optimization_results_tab(df_optimize)

    assert ("metric", "Rows", "1") in helper_st.calls
    assert ("metric", "Best Sharpe", "1.2500") in helper_st.calls
    assert ("metric", "Median Sharpe", "1.2500") in helper_st.calls
    assert ("metric", "Median Robustness", "0.7000") in helper_st.calls
    dataframe_calls = [call for call in helper_st.calls if call[0] == "dataframe"]
    assert len(dataframe_calls) == 1
    rendered_df = dataframe_calls[0][1]
    assert list(rendered_df.columns) == [
        "created_at",
        "run_id",
        "stage",
        "sharpe",
        "train_sharpe",
        "robustness_score",
        "cagr",
        "mdd",
        "params_view",
    ]
    assert ("caption", "Best row by Sharpe") in helper_st.calls
    assert (
        "json",
        {
            "run_id": "opt-1",
            "stage": "train",
            "sharpe": 1.25,
            "params": {"lookback": 14},
            "extra": {"bucket": "top"},
        },
    ) in helper_st.calls
    assert not any(call[0] == "plotly_chart" for call in helper_st.calls)


def test_select_active_live_jobs_filters_running_live_rows(monkeypatch) -> None:
    import numpy as real_np
    import pandas as real_pd

    module, _, _ = _load_dashboard_app(monkeypatch)
    monkeypatch.setitem(sys.modules, "numpy", real_np)
    monkeypatch.setitem(sys.modules, "pandas", real_pd)
    module.pd = real_pd

    workflow_jobs = real_pd.DataFrame(
        [
            {"job_id": "keep-1", "workflow": "live", "status": "RUNNING"},
            {"job_id": "keep-2", "workflow": "live_ws", "status": "STOP_REQUESTED"},
            {"job_id": "drop-1", "workflow": "backtest", "status": "RUNNING"},
            {"job_id": "drop-2", "workflow": "live", "status": "COMPLETED"},
        ]
    )

    active = module._select_active_live_jobs(workflow_jobs)

    assert active["job_id"].astype(str).tolist() == ["keep-1", "keep-2"]


def test_is_live_real_mode_armed_requires_both_checks_and_phrase(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)

    assert (
        module._is_live_real_mode_armed(
            arm_ack_1=True,
            arm_ack_2=True,
            arm_phrase="enable real",
        )
        is True
    )
    assert (
        module._is_live_real_mode_armed(
            arm_ack_1=True,
            arm_ack_2=False,
            arm_phrase="ENABLE REAL",
        )
        is False
    )
    assert (
        module._is_live_real_mode_armed(
            arm_ack_1=True,
            arm_ack_2=True,
            arm_phrase="not armed",
        )
        is False
    )


def test_render_live_runner_settings_surfaces_real_mode_lock(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(
        button_results=[],
        selectbox_results=[
            "WebSocket (uv run lq live --transport ws)",
            "real",
            "TrendStrategy",
        ],
        checkbox_results=[True, False],
        text_input_results=["ENABLE REAL"],
    )

    module.st = helper_st

    selection = module._render_live_runner_settings(
        strategy_options=["RsiStrategy", "TrendStrategy"],
        strategy_name="RsiStrategy",
    )

    assert selection == module._LiveRunnerSelection(
        runner_kind="WebSocket (uv run lq live --transport ws)",
        live_mode="real",
        strategy_name="TrendStrategy",
        real_armed=False,
    )
    assert any(
        call[0] == "warning"
        and "Real mode sends live exchange orders" in call[1]
        for call in helper_st.calls
    )
    assert ("info", "Real mode is locked until all arm checks are completed.") in helper_st.calls


def test_render_managed_run_launch_controls_starts_backtest_job(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[True, False, False])
    launch_calls: list[dict[str, Any]] = []

    module.st = helper_st
    monkeypatch.setattr(module, "_save_strategy_params", lambda strategy_name, params: "/tmp/params.json")
    monkeypatch.setattr(module.uuid, "uuid4", lambda: "backtest-run-id")
    monkeypatch.setattr(
        module,
        "_build_backtest_job_launch_spec",
        lambda **kwargs: _LaunchSpecRecorder("job-backtest", launch_calls, kwargs),
    )

    module._render_managed_run_launch_controls(
        db_path="postgres://lumina",
        launch_context=module._ManagedRunLaunchContext(
            strategy_name="RsiStrategy",
            strategy_params={"lookback": 14},
            runner_data_source="postgres",
            market_db_path="/tmp/market",
            market_exchange="binance",
            runner_env_overrides={"LQ__RUNNER__TIMEFRAME": "5m"},
            optimize_folds=4,
            optimize_trials=12,
            optimize_workers=2,
            persist_best_params=True,
            runner_leverage=3,
        ),
        live_runner_selection=module._LiveRunnerSelection(
            runner_kind="Polling (uv run lq live --transport poll)",
            live_mode="paper",
            strategy_name="RsiStrategy",
            real_armed=True,
        ),
        active_live_jobs=_Frame(empty=True),
    )

    assert launch_calls == [
        {
            "db_path": "postgres://lumina",
            "runner_data_source": "postgres",
            "market_db_path": "/tmp/market",
            "market_exchange": "binance",
            "runner_env_overrides": {"LQ__RUNNER__TIMEFRAME": "5m"},
            "strategy_name": "RsiStrategy",
            "backtest_run_id": "backtest-run-id",
            "strategy_params_path": "/tmp/params.json",
        }
    ]
    assert ("success", "Backtest job launched: job-backtest") in helper_st.calls
    assert helper_st.cache_data.clear_calls == 1


def test_render_managed_run_launch_controls_disables_live_when_real_mode_not_armed(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[False, False, False])

    module.st = helper_st

    module._render_managed_run_launch_controls(
        db_path="postgres://lumina",
        launch_context=module._ManagedRunLaunchContext(
            strategy_name="RsiStrategy",
            strategy_params={"lookback": 14},
            runner_data_source="postgres",
            market_db_path="/tmp/market",
            market_exchange="binance",
            runner_env_overrides={"LQ__RUNNER__TIMEFRAME": "5m"},
            optimize_folds=4,
            optimize_trials=12,
            optimize_workers=2,
            persist_best_params=True,
            runner_leverage=3,
        ),
        live_runner_selection=module._LiveRunnerSelection(
            runner_kind="Polling (uv run lq live --transport poll)",
            live_mode="real",
            strategy_name="RsiStrategy",
            real_armed=False,
        ),
        active_live_jobs=_Frame(empty=True),
    )

    live_button_calls = [call for call in helper_st.calls if call[0] == "button" and call[1] == "Start Live Job"]
    assert len(live_button_calls) == 1
    assert live_button_calls[0][2]["disabled"] is True


def test_render_report_tab_orchestrates_subsections_and_warnings(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    helper_st = _GhostCleanupStreamlit(button_results=[])
    calls: list[tuple[str, Any]] = []
    workflow_jobs = object()
    active_live_jobs = _Frame(empty=False, size=2)
    df_equity = _Frame(empty=True)
    trade_analytics = _Frame(empty=True)
    df_risk = _Frame(empty=True)
    df_hb = _Frame(empty=True)
    mirror_balance_equity = _Frame(empty=True)
    live_selection = module._LiveRunnerSelection(
        runner_kind="Polling (uv run lq live --transport poll)",
        live_mode="paper",
        strategy_name="TrendStrategy",
        real_armed=True,
    )

    module.st = helper_st
    monkeypatch.setattr(module, "load_workflow_jobs", lambda db_path, refresh_counter=0: workflow_jobs)
    monkeypatch.setattr(module, "_select_active_live_jobs", lambda rows: active_live_jobs)
    monkeypatch.setattr(
        module,
        "_render_live_runner_settings",
        lambda **kwargs: calls.append(("live_settings", kwargs)) or live_selection,
    )
    monkeypatch.setattr(
        module,
        "_render_managed_run_launch_controls",
        lambda **kwargs: calls.append(("launch_controls", kwargs)),
    )
    monkeypatch.setattr(
        module,
        "_render_workflow_jobs_section",
        lambda **kwargs: calls.append(("workflow_jobs", kwargs)),
    )
    monkeypatch.setattr(
        module,
        "_render_ghost_cleanup_section",
        lambda **kwargs: calls.append(("ghost_cleanup", kwargs)),
    )
    monkeypatch.setattr(
        module,
        "_render_snapshot_report_section",
        lambda **kwargs: calls.append(("snapshot_report", kwargs)),
    )

    module._render_report_tab(
        db_path="postgres://lumina",
        refresh_counter=5,
        strategy_options=["RsiStrategy", "TrendStrategy"],
        strategy_name="RsiStrategy",
        strategy_params={"lookback": 14},
        runner_initial_capital=1500.0,
        runner_leverage=3,
        runner_symbols=["BTC/USDT"],
        runner_timeframe="5m",
        runner_data_source="postgres",
        runner_timeout_sec=60,
        runner_env_overrides={"LQ__RUNNER__TIMEFRAME": "5m"},
        market_db_path="/tmp/market",
        market_exchange="binance",
        optimize_folds=4,
        optimize_trials=12,
        optimize_workers=2,
        persist_best_params=True,
        opt_space_error="invalid search space",
        run_stale_sec=300,
        summary={"bars": 10},
        performance={"sharpe": 1.2},
        active_run_id="run-1",
        resolved_source="postgres",
        period_preset="30d",
        df_equity=df_equity,
        trade_analytics=trade_analytics,
        df_risk=df_risk,
        df_hb=df_hb,
        mirror_snapshot={"wins": 1},
        mirror_balance_equity=mirror_balance_equity,
    )

    assert ("subheader", "No-Code Workflow Control") in helper_st.calls
    assert any(call[0] == "caption" and "initial_equity=1500.00" in call[1] for call in helper_st.calls)
    assert ("error", "invalid search space") in helper_st.calls
    assert any(call[0] == "warning" and "live job(s) already active" in call[1] for call in helper_st.calls)
    assert calls[:4] == [
        ("live_settings", {"strategy_options": ["RsiStrategy", "TrendStrategy"], "strategy_name": "RsiStrategy"}),
        (
            "launch_controls",
            {
                "db_path": "postgres://lumina",
                "launch_context": module._ManagedRunLaunchContext(
                    strategy_name="RsiStrategy",
                    strategy_params={"lookback": 14},
                    runner_data_source="postgres",
                    market_db_path="/tmp/market",
                    market_exchange="binance",
                    runner_env_overrides={"LQ__RUNNER__TIMEFRAME": "5m"},
                    optimize_folds=4,
                    optimize_trials=12,
                    optimize_workers=2,
                    persist_best_params=True,
                    runner_leverage=3,
                ),
                "live_runner_selection": live_selection,
                "active_live_jobs": active_live_jobs,
            },
        ),
        ("workflow_jobs", {"db_path": "postgres://lumina", "refresh_counter": 5}),
        ("ghost_cleanup", {"db_path": "postgres://lumina", "run_stale_sec": 300}),
    ]
    snapshot_name, snapshot_kwargs = calls[4]
    assert snapshot_name == "snapshot_report"
    assert snapshot_kwargs["summary"] == {"bars": 10}
    assert snapshot_kwargs["performance"] == {"sharpe": 1.2}
    assert snapshot_kwargs["active_run_id"] == "run-1"
    assert snapshot_kwargs["resolved_source"] == "postgres"
    assert snapshot_kwargs["strategy_name"] == "RsiStrategy"
    assert snapshot_kwargs["period_preset"] == "30d"
    assert snapshot_kwargs["df_equity"] is df_equity
    assert snapshot_kwargs["trade_analytics"] is trade_analytics
    assert snapshot_kwargs["df_risk"] is df_risk
    assert snapshot_kwargs["df_hb"] is df_hb
    assert snapshot_kwargs["mirror_balance_equity"] is mirror_balance_equity


def test_render_dashboard_selection_controls_clamps_timeframe_and_collects_sidebar_values(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    calls: list[tuple[Any, ...]] = []
    sidebar = _SidebarControlsFake(
        calls,
        selectbox_results=["Postgres", "TrendStrategy", "30D"],
        text_input_results=["postgres://lumina", "/tmp/market", "binanceusdm", "15s"],
        toggle_results=[False, True, False, True],
        slider_results=[9, 12000, 1800, 600],
    )
    module.st = _SidebarStreamlit(sidebar)
    monkeypatch.setattr(module.strategy_registry, "get_strategy_names", lambda: ["RsiStrategy", "TrendStrategy"])
    monkeypatch.setattr(module, "timeframe_to_milliseconds", lambda value: 15_000)

    controls = module._render_dashboard_selection_controls()

    assert controls == module._DashboardSelectionControls(
        data_source="Postgres",
        db_path="postgres://lumina",
        market_db_path="/tmp/market",
        market_exchange="binanceusdm",
        market_timeframe="1m",
        strategy_options=("RsiStrategy", "TrendStrategy"),
        strategy_name="TrendStrategy",
        auto_refresh_enabled=False,
        refresh_interval_sec=9,
        max_points=12000,
        auto_downsample=True,
        downsample_target_points=1800,
        pin_to_running=False,
        filter_runs_by_strategy=True,
        run_stale_sec=600,
        period_preset="30D",
        custom_start=None,
        custom_end=None,
    )
    assert (
        "sidebar.caption",
        "Market chart timeframe clamped to 1m minimum for dashboard performance.",
    ) in calls


def test_render_execution_lab_controls_parses_symbols_and_keeps_runner_inputs(monkeypatch) -> None:
    module, _, _ = _load_dashboard_app(monkeypatch)
    calls: list[tuple[Any, ...]] = []
    sidebar = _SidebarControlsFake(
        calls,
        number_input_results=[2500.0, 5],
        text_input_results=["ETH/USDT, BTC/USDT", "15m"],
        slider_results=[1200],
        selectbox_results=["db"],
    )
    module.st = _SidebarStreamlit(sidebar)

    controls = module._render_execution_lab_controls()

    assert controls == module._ExecutionLabControls(
        runner_initial_capital=2500.0,
        runner_leverage=5,
        runner_symbols=("ETH/USDT", "BTC/USDT"),
        runner_timeframe="15m",
        runner_timeout_sec=1200,
        runner_data_source="db",
    )
    assert ("sidebar.divider",) in calls
    assert ("sidebar.subheader", "Execution Lab") in calls
