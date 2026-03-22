"""Helpers for execution analytics metrics and charts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import plotly.graph_objects as go


def build_execution_metric_rows(
    summary: dict[str, object],
    *,
    format_duration_seconds: Callable[[object], str],
) -> list[list[tuple[str, str]]]:
    return [
        [
            ("BUY fills", f"{summary['buy_fills']}"),
            ("SELL fills", f"{summary['sell_fills']}"),
            ("Avg Qty", f"{summary['avg_qty']:.4f}"),
            ("Avg Notional", f"{summary['avg_notional']:.2f}"),
        ],
        [
            ("Total Commission", f"{summary['total_commission']:.4f}"),
            ("Avg Trade Return", f"{summary['avg_trade_return_pct']:.4f}%"),
            ("Best Trade PnL", f"{summary['best_trade_pnl']:.4f}"),
            ("Worst Trade PnL", f"{summary['worst_trade_pnl']:.4f}"),
        ],
        [
            ("Max Win Streak", f"{int(summary['win_streak_max'])}"),
            ("Max Loss Streak", f"{int(summary['loss_streak_max'])}"),
            (
                "Avg Win/Loss Streak",
                f"{summary['win_streak_avg']:.2f} / {summary['loss_streak_avg']:.2f}",
            ),
            ("Avg Holding Time", format_duration_seconds(summary["holding_time_avg_sec"])),
        ],
    ]


def build_direction_table(
    summary: dict[str, object],
    *,
    safe_float: Callable[[object, float], float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Direction": "Long",
                "Closed Trades": int(summary.get("long_trades", 0)),
                "Win Rate": f"{safe_float(summary.get('long_win_rate'), 0.0):.2%}",
            },
            {
                "Direction": "Short",
                "Closed Trades": int(summary.get("short_trades", 0)),
                "Win Rate": f"{safe_float(summary.get('short_win_rate'), 0.0):.2%}",
            },
        ]
    )


def filter_closed_trade_analytics(trade_analytics: pd.DataFrame) -> pd.DataFrame:
    if "closed_qty" in trade_analytics.columns:
        return trade_analytics[trade_analytics["closed_qty"] > 0]
    return trade_analytics[trade_analytics["realized_pnl"] != 0]


def build_trade_pnl_figure(closed: pd.DataFrame, *, go_module=go):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Bar(
            x=closed["datetime"],
            y=closed["realized_pnl"],
            name="Realized PnL per closing trade",
        )
    )
    fig.update_layout(title="Trade-by-Trade Realized PnL", template="plotly_white")
    return fig


def build_cumulative_realized_pnl_figure(closed: pd.DataFrame, *, go_module=go):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=closed["datetime"],
            y=closed["cum_realized_pnl"],
            mode="lines",
            name="Cumulative Realized PnL",
        )
    )
    fig.update_layout(title="Cumulative Realized PnL", template="plotly_white")
    return fig


def build_streak_distribution_figure(
    closed: pd.DataFrame,
    *,
    streak_groups: Callable[[Sequence[bool]], Sequence[tuple[bool, int]]],
    go_module=go,
):
    decisive = closed[pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0) != 0.0]
    if decisive.empty:
        return None

    outcomes = list((decisive["realized_pnl"] > 0.0).to_numpy(dtype=bool))
    streak_rows = [
        {"Result": "Win" if flag else "Loss", "Length": int(length)}
        for flag, length in streak_groups(outcomes)
    ]
    if not streak_rows:
        return None

    streak_df = pd.DataFrame(streak_rows)
    dist = (
        streak_df.groupby(["Length", "Result"], observed=False)
        .size()
        .reset_index(name="Count")
    )
    fig = go_module.Figure()
    for label, color in [("Win", "#2f855a"), ("Loss", "#c53030")]:
        part = dist[dist["Result"] == label]
        if part.empty:
            continue
        fig.add_trace(
            go_module.Bar(
                x=part["Length"],
                y=part["Count"],
                name=label,
                marker_color=color,
            )
        )
    fig.update_layout(
        title="Win/Loss Streak Distribution",
        xaxis_title="Streak Length",
        yaxis_title="Occurrences",
        barmode="group",
        template="plotly_white",
    )
    return fig


def build_order_status_figure(df_orders: pd.DataFrame, *, go_module=go):
    status_counts = df_orders["status"].fillna("UNKNOWN").astype(str).value_counts()
    fig = go_module.Figure(
        data=[
            go_module.Pie(
                labels=status_counts.index.tolist(),
                values=status_counts.values.tolist(),
                hole=0.4,
            )
        ]
    )
    fig.update_layout(title="Order Status Distribution")
    return fig


__all__ = [
    "build_cumulative_realized_pnl_figure",
    "build_direction_table",
    "build_execution_metric_rows",
    "build_order_status_figure",
    "build_streak_distribution_figure",
    "build_trade_pnl_figure",
    "filter_closed_trade_analytics",
]
