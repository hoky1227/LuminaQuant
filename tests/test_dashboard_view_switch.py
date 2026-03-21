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
