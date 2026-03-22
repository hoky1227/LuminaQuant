"""Helpers for market-tab summary and pair-visualization panels."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go


def build_market_summary_metrics(plot_market: pd.DataFrame) -> dict[str, str]:
    high_val = float(pd.to_numeric(plot_market["high"], errors="coerce").max())
    low_val = float(pd.to_numeric(plot_market["low"], errors="coerce").min())
    return {
        "market_bars": f"{len(plot_market)}",
        "first_price": f"{float(plot_market['close'].iloc[0]):.4f}",
        "last_price": f"{float(plot_market['close'].iloc[-1]):.4f}",
        "range": f"{low_val:.4f} - {high_val:.4f}",
    }


def build_market_close_figure(
    plot_market: pd.DataFrame,
    *,
    market_symbol: str,
    market_timeframe: str,
    go_module=go,
):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=plot_market["datetime"],
            y=plot_market["close"],
            mode="lines",
            name="Close",
        )
    )
    fig.update_layout(
        title=f"{market_symbol} Close Price ({market_timeframe})",
        template="plotly_white",
    )
    return fig


def build_market_volume_figure(plot_market: pd.DataFrame, *, go_module=go):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Bar(
            x=plot_market["datetime"],
            y=plot_market["volume"],
            name="Volume",
        )
    )
    fig.update_layout(title="Market Volume", template="plotly_white")
    return fig


def build_pair_indicator_summary(
    pair_indicator_df: pd.DataFrame,
    *,
    pair_symbol_x: str,
    pair_symbol_y: str,
) -> dict[str, str]:
    z_series = pd.to_numeric(pair_indicator_df["zscore"], errors="coerce")
    hedge_series = pd.to_numeric(pair_indicator_df["hedge_ratio"], errors="coerce")
    corr_series = pd.to_numeric(pair_indicator_df["correlation"], errors="coerce")

    latest_z = float(z_series.dropna().iloc[-1]) if z_series.notna().any() else float("nan")
    latest_beta = (
        float(hedge_series.dropna().iloc[-1]) if hedge_series.notna().any() else float("nan")
    )
    latest_corr = float(corr_series.dropna().iloc[-1]) if corr_series.notna().any() else float("nan")

    return {
        "pair": f"{pair_symbol_x} vs {pair_symbol_y}",
        "latest_z": f"{latest_z:.3f}" if math.isfinite(latest_z) else "N/A",
        "hedge_ratio": f"{latest_beta:.4f}" if math.isfinite(latest_beta) else "N/A",
        "correlation": f"{latest_corr:.4f}" if math.isfinite(latest_corr) else "N/A",
    }


def build_pair_price_inputs_figure(
    pair_plot_df: pd.DataFrame,
    *,
    pair_symbol_x: str,
    pair_symbol_y: str,
    go_module=go,
):
    fig = go_module.Figure()
    normalized_x = pair_plot_df["close_x"] / float(pair_plot_df["close_x"].iloc[0])
    normalized_y = pair_plot_df["close_y"] / float(pair_plot_df["close_y"].iloc[0])
    fig.add_trace(
        go_module.Scatter(
            x=pair_plot_df["datetime"],
            y=normalized_x,
            mode="lines",
            name=f"{pair_symbol_x} (normalized)",
            line=dict(color="#2b6cb0", width=1.8),
        )
    )
    fig.add_trace(
        go_module.Scatter(
            x=pair_plot_df["datetime"],
            y=normalized_y,
            mode="lines",
            name=f"{pair_symbol_y} (normalized)",
            line=dict(color="#ed8936", width=1.8),
        )
    )
    fig.update_layout(
        title="Pair Price Inputs (Normalized)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_pair_zscore_figure(
    pair_plot_df: pd.DataFrame,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    go_module=go,
):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=pair_plot_df["datetime"],
            y=pair_plot_df["zscore"],
            mode="lines",
            name="Z-Score",
            line=dict(color="#2b6cb0", width=2),
        )
    )
    fig.add_hline(y=entry_z, line_dash="dash", line_color="#c53030")
    fig.add_hline(y=-entry_z, line_dash="dash", line_color="#c53030")
    fig.add_hline(y=exit_z, line_dash="dot", line_color="#2f855a")
    fig.add_hline(y=-exit_z, line_dash="dot", line_color="#2f855a")
    fig.add_hline(y=stop_z, line_dash="dash", line_color="#7b341e")
    fig.add_hline(y=-stop_z, line_dash="dash", line_color="#7b341e")
    fig.update_layout(
        title="Pair Z-Score with Entry/Exit/Stop Bands",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_pair_spread_figure(pair_plot_df: pd.DataFrame, *, go_module=go):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=pair_plot_df["datetime"],
            y=pair_plot_df["spread"],
            mode="lines",
            name="Spread",
            line=dict(color="#4a5568", width=1.6),
        )
    )
    fig.add_trace(
        go_module.Scatter(
            x=pair_plot_df["datetime"],
            y=pair_plot_df["spread_mean"],
            mode="lines",
            name="Spread Mean",
            line=dict(color="#2f855a", width=1.2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Hedge-Adjusted Spread",
        xaxis_title="Date",
        yaxis_title="Spread",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


__all__ = [
    "build_market_close_figure",
    "build_market_summary_metrics",
    "build_market_volume_figure",
    "build_pair_indicator_summary",
    "build_pair_price_inputs_figure",
    "build_pair_spread_figure",
    "build_pair_zscore_figure",
]
