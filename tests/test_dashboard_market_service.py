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


class _FakeScatter:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name")


class _FakeFigure:
    def __init__(self) -> None:
        self.data: list[Any] = []
        self.layout = types.SimpleNamespace(title=types.SimpleNamespace(text=None))
        self.hlines: list[dict[str, Any]] = []

    def add_trace(self, trace: Any) -> None:
        self.data.append(trace)

    def add_hline(self, **kwargs: Any) -> None:
        self.hlines.append(kwargs)

    def update_layout(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "title":
                self.layout.title = types.SimpleNamespace(text=value)
            else:
                setattr(self.layout, key, value)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "market_dashboard.py"


def _load_module(monkeypatch):
    fake_plotly = types.ModuleType("plotly")
    fake_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_graph_objects.Figure = _FakeFigure
    fake_graph_objects.Bar = _FakeBar
    fake_graph_objects.Scatter = _FakeScatter
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_graph_objects)

    spec = importlib.util.spec_from_file_location("dashboard_market_service", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_build_market_summary_metrics_and_figures(monkeypatch) -> None:
    market_dashboard = _load_module(monkeypatch)
    plot_market = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-03-22T00:00:00Z", "2026-03-22T00:01:00Z"],
                utc=True,
            ),
            "close": [101.25, 103.5],
            "high": [102.0, 104.0],
            "low": [100.5, 101.0],
            "volume": [12.0, 14.5],
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Bar=_FakeBar, Scatter=_FakeScatter)

    metrics = market_dashboard.build_market_summary_metrics(plot_market)
    close_figure = market_dashboard.build_market_close_figure(
        plot_market,
        market_symbol="BTC/USDT",
        market_timeframe="1h",
        go_module=fake_go,
    )
    volume_figure = market_dashboard.build_market_volume_figure(plot_market, go_module=fake_go)

    assert metrics == {
        "market_bars": "2",
        "first_price": "101.2500",
        "last_price": "103.5000",
        "range": "100.5000 - 104.0000",
    }
    assert close_figure.layout.title.text == "BTC/USDT Close Price (1h)"
    assert [trace.name for trace in close_figure.data] == ["Close"]
    assert volume_figure.layout.title.text == "Market Volume"
    assert [trace.name for trace in volume_figure.data] == ["Volume"]


def test_pair_indicator_helpers_return_summary_and_figures(monkeypatch) -> None:
    market_dashboard = _load_module(monkeypatch)
    pair_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-03-22T00:00:00Z", "2026-03-22T00:01:00Z"],
                utc=True,
            ),
            "close_x": [10.0, 11.0],
            "close_y": [20.0, 19.0],
            "zscore": [0.5, 1.25],
            "hedge_ratio": [0.8, 0.85],
            "correlation": [0.9, 0.87],
            "spread": [0.1, 0.3],
            "spread_mean": [0.15, 0.15],
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)

    summary = market_dashboard.build_pair_indicator_summary(
        pair_frame,
        pair_symbol_x="BTC/USDT",
        pair_symbol_y="ETH/USDT",
    )
    price_figure = market_dashboard.build_pair_price_inputs_figure(
        pair_frame,
        pair_symbol_x="BTC/USDT",
        pair_symbol_y="ETH/USDT",
        go_module=fake_go,
    )
    zscore_figure = market_dashboard.build_pair_zscore_figure(
        pair_frame,
        entry_z=2.0,
        exit_z=0.35,
        stop_z=3.5,
        go_module=fake_go,
    )
    spread_figure = market_dashboard.build_pair_spread_figure(pair_frame, go_module=fake_go)

    assert summary == {
        "pair": "BTC/USDT vs ETH/USDT",
        "latest_z": "1.250",
        "hedge_ratio": "0.8500",
        "correlation": "0.8700",
    }
    assert [trace.name for trace in price_figure.data] == [
        "BTC/USDT (normalized)",
        "ETH/USDT (normalized)",
    ]
    assert zscore_figure.layout.title.text == "Pair Z-Score with Entry/Exit/Stop Bands"
    assert len(zscore_figure.hlines) == 6
    assert [trace.name for trace in spread_figure.data] == ["Spread", "Spread Mean"]


def test_build_pair_indicator_summary_handles_missing_numeric_values(monkeypatch) -> None:
    market_dashboard = _load_module(monkeypatch)
    pair_frame = pd.DataFrame(
        {
            "zscore": [None, float("nan")],
            "hedge_ratio": [float("nan"), None],
            "correlation": [None, float("nan")],
        }
    )

    summary = market_dashboard.build_pair_indicator_summary(
        pair_frame,
        pair_symbol_x="BTC/USDT",
        pair_symbol_y="ETH/USDT",
    )

    assert summary == {
        "pair": "BTC/USDT vs ETH/USDT",
        "latest_z": "N/A",
        "hedge_ratio": "N/A",
        "correlation": "N/A",
    }


def test_rsi_helpers_return_summary_and_figures(monkeypatch) -> None:
    market_dashboard = _load_module(monkeypatch)
    indicator_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-03-22T00:00:00Z",
                    "2026-03-22T00:01:00Z",
                    "2026-03-22T00:02:00Z",
                ],
                utc=True,
            ),
            "close": [100.0, 99.0, 101.0],
            "rsi": [35.0, 25.0, 75.0],
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)

    summary = market_dashboard.build_rsi_summary_metrics(
        indicator_df,
        rsi_period=14,
        oversold=30.0,
        overbought=70.0,
    )
    rsi_figure = market_dashboard.build_rsi_figure(
        indicator_df,
        rsi_period=14,
        oversold=30.0,
        overbought=70.0,
        go_module=fake_go,
    )
    signal_figure = market_dashboard.build_rsi_signal_figure(
        indicator_df,
        oversold=30.0,
        overbought=70.0,
        go_module=fake_go,
    )

    assert summary == {
        "rsi_period": "14",
        "latest_rsi": "75.00",
        "rsi_zone": "Overbought",
    }
    assert rsi_figure.layout.title.text == "RSI (14) with Oversold/Overbought Bands"
    assert len(rsi_figure.hlines) == 2
    assert [trace.name for trace in signal_figure.data] == [
        "Close",
        "RSI Long Trigger",
        "RSI Exit Trigger",
    ]


def test_moving_average_helpers_return_summary_and_trigger_figure(monkeypatch) -> None:
    market_dashboard = _load_module(monkeypatch)
    indicator_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-03-22T00:00:00Z",
                    "2026-03-22T00:01:00Z",
                    "2026-03-22T00:02:00Z",
                    "2026-03-22T00:03:00Z",
                ],
                utc=True,
            ),
            "close": [100.0, 101.0, 99.0, 100.0],
            "short_ma": [1.0, 0.0, 3.0, 0.0],
            "long_ma": [2.0, 1.0, 2.0, 1.0],
        }
    )
    fake_go = types.SimpleNamespace(Figure=_FakeFigure, Scatter=_FakeScatter)

    summary = market_dashboard.build_moving_average_summary_metrics(
        short_window=10,
        long_window=30,
    )
    figure = market_dashboard.build_moving_average_figure(
        indicator_df,
        short_window=10,
        long_window=30,
        go_module=fake_go,
    )

    assert summary == {
        "short_window": "10",
        "long_window": "30",
    }
    assert figure.layout.title.text == "Moving Average Strategy Inputs and Cross Triggers"
    assert [trace.name for trace in figure.data] == [
        "Close",
        "Short MA (10)",
        "Long MA (30)",
        "MA Long Trigger",
        "MA Exit Trigger",
    ]
