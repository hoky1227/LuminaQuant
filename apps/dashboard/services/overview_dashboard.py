"""Helpers for overview-tab equity and performance charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_equity_curve_figure(plot_equity: pd.DataFrame, *, go_module=go):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=plot_equity["datetime"],
            y=plot_equity["total"],
            mode="lines",
            name="Strategy Equity",
            line=dict(color="#0db39e", width=2),
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_benchmark_price_figure(plot_equity: pd.DataFrame, *, go_module=go):
    benchmark_series = pd.to_numeric(plot_equity.get("benchmark_price"), errors="coerce")
    if not benchmark_series.notna().any():
        return None

    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=plot_equity["datetime"],
            y=benchmark_series,
            mode="lines",
            line=dict(color="#805ad5", width=1.5),
            name="Benchmark Price",
        )
    )
    fig.update_layout(
        title="Benchmark Price (from Equity Metadata)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_funding_figure(plot_equity: pd.DataFrame, *, go_module=go):
    funding_series = pd.to_numeric(plot_equity.get("funding"), errors="coerce")
    if not funding_series.notna().any():
        return None

    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=plot_equity["datetime"],
            y=funding_series,
            mode="lines",
            line=dict(color="#dd6b20", width=2),
            name="Funding (Net)",
        )
    )
    fig.update_layout(
        title="Funding (Net) Over Time",
        xaxis_title="Date",
        yaxis_title="Funding",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_cumulative_return_figure(
    df_equity: pd.DataFrame,
    performance: dict[str, object] | None,
    *,
    go_module=go,
):
    if not performance:
        return None
    cum_return_series = performance.get("cum_return_series", [])
    if len(cum_return_series) != len(df_equity.index) - 1:
        return None

    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=df_equity["datetime"].iloc[1:],
            y=pd.Series(cum_return_series),
            mode="lines",
            line=dict(color="#2b6cb0", width=2),
            name="Cumulative Return",
        )
    )
    fig.update_layout(
        title="Cumulative Return",
        xaxis_title="Date",
        yaxis_title="Return",
        template="plotly_white",
        hovermode="x unified",
        yaxis_tickformat=".2%",
    )
    return fig


def build_drawdown_figure(plot_equity: pd.DataFrame, *, go_module=go):
    roll_max = plot_equity["total"].cummax()
    drawdown = (plot_equity["total"] - roll_max) / roll_max
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=plot_equity["datetime"],
            y=drawdown,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#f05a66"),
        )
    )
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown", template="plotly_white")
    return fig


def build_monthly_returns_heatmap(
    monthly_table: pd.DataFrame,
    *,
    safe_float,
    go_module=go,
):
    values = monthly_table.to_numpy(dtype=float)
    text_vals = [[f"{safe_float(v):.2%}" if np.isfinite(v) else "" for v in row] for row in values]
    fig = go_module.Figure(
        data=[
            go_module.Heatmap(
                z=values,
                x=list(monthly_table.columns),
                y=[str(idx) for idx in monthly_table.index],
                colorscale="RdBu",
                zmid=0.0,
                text=text_vals,
                texttemplate="%{text}",
                hovertemplate="Year %{y} | %{x}: %{z:.2%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
    )
    return fig


__all__ = [
    "build_benchmark_price_figure",
    "build_cumulative_return_figure",
    "build_drawdown_figure",
    "build_equity_curve_figure",
    "build_funding_figure",
    "build_monthly_returns_heatmap",
]
