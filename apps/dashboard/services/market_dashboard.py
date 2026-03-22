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


def build_rsi_summary_metrics(
    indicator_df: pd.DataFrame,
    *,
    rsi_period: int,
    oversold: float,
    overbought: float,
) -> dict[str, str]:
    rsi_series = pd.to_numeric(indicator_df["rsi"], errors="coerce")
    latest_rsi = float(rsi_series.dropna().iloc[-1]) if rsi_series.notna().any() else float("nan")
    rsi_zone = "N/A"
    if math.isfinite(latest_rsi):
        if latest_rsi <= oversold:
            rsi_zone = "Oversold"
        elif latest_rsi >= overbought:
            rsi_zone = "Overbought"
        else:
            rsi_zone = "Neutral"

    return {
        "rsi_period": f"{rsi_period}",
        "latest_rsi": f"{latest_rsi:.2f}" if math.isfinite(latest_rsi) else "N/A",
        "rsi_zone": rsi_zone,
    }


def build_rsi_figure(
    indicator_df: pd.DataFrame,
    *,
    rsi_period: int,
    oversold: float,
    overbought: float,
    go_module=go,
):
    rsi_series = pd.to_numeric(indicator_df["rsi"], errors="coerce")
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=indicator_df["datetime"],
            y=rsi_series,
            mode="lines",
            name="RSI",
            line=dict(color="#2b6cb0", width=2),
        )
    )
    fig.add_hline(y=oversold, line_dash="dash", line_color="#2f855a")
    fig.add_hline(y=overbought, line_dash="dash", line_color="#c53030")
    fig.update_layout(
        title=f"RSI ({rsi_period}) with Oversold/Overbought Bands",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_rsi_signal_figure(
    indicator_df: pd.DataFrame,
    *,
    oversold: float,
    overbought: float,
    go_module=go,
):
    rsi_series = pd.to_numeric(indicator_df["rsi"], errors="coerce")
    prev_rsi = rsi_series.shift(1)
    long_entries = indicator_df[(prev_rsi >= oversold) & (rsi_series < oversold)]
    exits = indicator_df[(prev_rsi <= overbought) & (rsi_series > overbought)]

    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=indicator_df["datetime"],
            y=indicator_df["close"],
            mode="lines",
            name="Close",
            line=dict(color="#4a5568", width=1.5),
        )
    )
    if not long_entries.empty:
        fig.add_trace(
            go_module.Scatter(
                x=long_entries["datetime"],
                y=long_entries["close"],
                mode="markers",
                name="RSI Long Trigger",
                marker=dict(color="#2f855a", size=8, symbol="triangle-up"),
            )
        )
    if not exits.empty:
        fig.add_trace(
            go_module.Scatter(
                x=exits["datetime"],
                y=exits["close"],
                mode="markers",
                name="RSI Exit Trigger",
                marker=dict(color="#c53030", size=8, symbol="triangle-down"),
            )
        )
    fig.update_layout(
        title="Price with RSI Trigger Points",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_moving_average_summary_metrics(*, short_window: int, long_window: int) -> dict[str, str]:
    return {
        "short_window": f"{short_window}",
        "long_window": f"{long_window}",
    }


def build_moving_average_figure(
    indicator_df: pd.DataFrame,
    *,
    short_window: int,
    long_window: int,
    go_module=go,
):
    short_ma = pd.to_numeric(indicator_df.get("short_ma"), errors="coerce")
    long_ma = pd.to_numeric(indicator_df.get("long_ma"), errors="coerce")
    prev_short = short_ma.shift(1)
    prev_long = long_ma.shift(1)
    cross_up = indicator_df[(prev_short <= prev_long) & (short_ma > long_ma)]
    cross_down = indicator_df[(prev_short >= prev_long) & (short_ma < long_ma)]

    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=indicator_df["datetime"],
            y=indicator_df["close"],
            mode="lines",
            name="Close",
            line=dict(color="#4a5568", width=1.5),
        )
    )
    fig.add_trace(
        go_module.Scatter(
            x=indicator_df["datetime"],
            y=short_ma,
            mode="lines",
            name=f"Short MA ({short_window})",
            line=dict(color="#2b6cb0", width=1.8),
        )
    )
    fig.add_trace(
        go_module.Scatter(
            x=indicator_df["datetime"],
            y=long_ma,
            mode="lines",
            name=f"Long MA ({long_window})",
            line=dict(color="#ed8936", width=1.8),
        )
    )
    if not cross_up.empty:
        fig.add_trace(
            go_module.Scatter(
                x=cross_up["datetime"],
                y=cross_up["close"],
                mode="markers",
                name="MA Long Trigger",
                marker=dict(color="#2f855a", size=8, symbol="triangle-up"),
            )
        )
    if not cross_down.empty:
        fig.add_trace(
            go_module.Scatter(
                x=cross_down["datetime"],
                y=cross_down["close"],
                mode="markers",
                name="MA Exit Trigger",
                marker=dict(color="#c53030", size=8, symbol="triangle-down"),
            )
        )

    fig.update_layout(
        title="Moving Average Strategy Inputs and Cross Triggers",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


__all__ = [
    "build_market_close_figure",
    "build_market_summary_metrics",
    "build_market_volume_figure",
    "build_moving_average_figure",
    "build_moving_average_summary_metrics",
    "build_pair_indicator_summary",
    "build_pair_price_inputs_figure",
    "build_pair_spread_figure",
    "build_pair_zscore_figure",
    "build_rsi_figure",
    "build_rsi_signal_figure",
    "build_rsi_summary_metrics",
]
