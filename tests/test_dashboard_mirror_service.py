from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd


class _FakeStreamlit(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.calls: list[tuple[Any, ...]] = []

    def markdown(self, value: str, **kwargs: Any) -> None:
        self.calls.append(("markdown", value, kwargs))


class _FakeScatter:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name")


class _FakeFigure:
    def __init__(self) -> None:
        self.data: list[_FakeScatter] = []
        self.layout = types.SimpleNamespace(
            title=types.SimpleNamespace(text=None),
            annotations=[],
            yaxis=types.SimpleNamespace(tickprefix=None),
        )

    def add_trace(self, trace: _FakeScatter, secondary_y: bool | None = None) -> None:
        self.data.append(trace)

    def update_layout(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "title":
                self.layout.title = types.SimpleNamespace(text=value)
            else:
                setattr(self.layout, key, value)

    def update_xaxes(self, **kwargs: Any) -> None:
        self.layout.xaxis = types.SimpleNamespace(**kwargs)

    def update_yaxes(self, **kwargs: Any) -> None:
        secondary_y = bool(kwargs.pop("secondary_y", False))
        attr = "yaxis2" if secondary_y else "yaxis"
        target = getattr(self.layout, attr, types.SimpleNamespace())
        for key, value in kwargs.items():
            setattr(target, key, value)
        setattr(self.layout, attr, target)

    def add_hline(self, **kwargs: Any) -> None:
        self.layout.hlines = [*getattr(self.layout, "hlines", []), kwargs]

    def add_annotation(self, **kwargs: Any) -> None:
        self.layout.annotations = [*self.layout.annotations, types.SimpleNamespace(**kwargs)]


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "mirror_dashboard.py"


def _load_module(monkeypatch):
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    fake_plotly = types.ModuleType("plotly")
    fake_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_graph_objects.Figure = _FakeFigure
    fake_graph_objects.Scatter = _FakeScatter
    fake_subplots = types.ModuleType("plotly.subplots")
    fake_subplots.make_subplots = lambda **kwargs: _FakeFigure()
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_graph_objects)
    monkeypatch.setitem(sys.modules, "plotly.subplots", fake_subplots)
    spec = importlib.util.spec_from_file_location("dashboard_mirror", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, fake_streamlit


def test_build_mirror_snapshot_prefers_balance_equity_tail(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)
    frame = pd.DataFrame(
        {
            "cum_total_pnl": [1.5, 2.5],
            "open_pnl": [0.25, 0.75],
        }
    )

    snapshot = module.build_mirror_snapshot(
        {"fills": 7, "wins": 3, "losses": 4, "total_net_profit": "2.25", "equity_drawdown_maximal": 0.5},
        frame,
        safe_float=lambda value, default=0.0: default if value is None else float(value),
        safe_div=lambda numerator, denominator, default=0.0: default if not denominator else numerator / denominator,
    )

    assert snapshot == {
        "total_trades": 7,
        "wins": 3,
        "losses": 4,
        "win_rate": 0.0,
        "closed_pnl": 2.25,
        "open_pnl": 0.75,
        "total_c_plus_o": 2.5,
        "equity_mdd": 0.5,
        "equity_mdd_rel": 0.0,
        "r_mdd": 5.0,
    }


def test_render_mirror_cards_emits_styled_cards(monkeypatch) -> None:
    module, fake_streamlit = _load_module(monkeypatch)

    module.render_mirror_cards(
        {
            "total_trades": 2,
            "wins": 1,
            "losses": 1,
            "win_rate": 0.5,
            "closed_pnl": 1.23,
            "open_pnl": -0.45,
            "total_c_plus_o": 0.78,
            "equity_mdd": 0.12,
            "equity_mdd_rel": 0.034,
            "r_mdd": 6.5,
        },
        safe_float=lambda value, default=0.0: default if value is None else float(value),
        tone_class=lambda value, invert=False: "down" if value and float(value) < 0 else "up",
        format_signed_dollar=lambda value, digits=2: f"${float(value):.{digits}f}",
        st_module=fake_streamlit,
    )

    assert fake_streamlit.calls[0][0] == "markdown"
    assert ".lq-mirror-grid" in fake_streamlit.calls[0][1]
    assert fake_streamlit.calls[1][0] == "markdown"
    assert "TOTAL TRADES" in fake_streamlit.calls[1][1]
    assert "0.78" in fake_streamlit.calls[1][1]


def test_build_mirror_figures_use_shared_layout(monkeypatch) -> None:
    module, _ = _load_module(monkeypatch)
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-03-22T00:00:00Z", "2026-03-22T00:01:00Z"], utc=True
            ),
            "cum_total_pnl": [0.0, 2.5],
            "drawdown_signed": [0.0, -0.5],
            "equity": [100.0, 102.5],
            "balance": [100.0, 101.8],
        }
    )

    equity_fig = module.build_mirror_equity_curve_figure(frame)
    balance_fig = module.build_mirror_balance_equity_figure(
        frame,
        {"equity_mdd": 0.5},
        safe_float=lambda value, default=0.0: default if value is None else float(value),
    )

    assert equity_fig.layout.title.text == "EQUITY CURVE (CUMULATIVE PNL)"
    assert [trace.name for trace in equity_fig.data] == ["Cumulative PnL"]
    assert equity_fig.layout.yaxis.tickprefix == "$"

    assert [trace.name for trace in balance_fig.data] == ["Drawdown", "Equity", "Balance"]
    assert balance_fig.layout.title.text == "BALANCE & EQUITY (TIME-BASED)"
    assert balance_fig.layout.annotations[0].text == "Eq MDD: $0.50"
