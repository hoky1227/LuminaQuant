"""Helpers for the dashboard risk/heartbeat/process-trace panels."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_risk_reason_figure(df_risk: pd.DataFrame, *, go_module=go):
    reason_counts = df_risk["reason"].fillna("UNKNOWN").astype(str).value_counts()
    fig = go_module.Figure(
        data=[go_module.Bar(x=reason_counts.index.tolist(), y=reason_counts.values.tolist())]
    )
    fig.update_layout(title="Risk Event Counts by Reason", template="plotly_white")
    return fig


def prepare_heartbeat_interval_frame(df_hb: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    hb = df_hb.copy().sort_values("heartbeat_time")
    hb["delta_sec"] = hb["heartbeat_time"].diff().dt.total_seconds()
    avg_hb = float(hb["delta_sec"].dropna().mean()) if hb["delta_sec"].notna().any() else 0.0
    return hb, avg_hb


def build_heartbeat_interval_figure(hb: pd.DataFrame, *, go_module=go):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=hb["heartbeat_time"],
            y=hb["delta_sec"],
            mode="lines+markers",
            name="Heartbeat interval",
        )
    )
    fig.update_layout(title="Heartbeat Interval Trend", template="plotly_white")
    return fig


def build_order_state_figure(df_order_states: pd.DataFrame, *, go_module=go):
    state_counts = df_order_states["state"].fillna("UNKNOWN").astype(str).value_counts()
    fig = go_module.Figure(
        data=[go_module.Bar(x=state_counts.index.tolist(), y=state_counts.values.tolist())]
    )
    fig.update_layout(title="Order State Event Counts", template="plotly_white")
    return fig


def build_strategy_process_trace_frame(
    *,
    df_orders: pd.DataFrame,
    df_risk: pd.DataFrame,
    df_hb: pd.DataFrame,
    df_order_states: pd.DataFrame,
) -> pd.DataFrame:
    trace_parts: list[pd.DataFrame] = []
    if not df_orders.empty and "created_at" in df_orders.columns:
        orders = df_orders[["created_at", "symbol", "side", "status"]].copy()
        orders = orders.rename(columns={"created_at": "event_time", "side": "event_detail"})
        orders["event_type"] = "order"
        trace_parts.append(orders)
    if not df_risk.empty and "event_time" in df_risk.columns:
        risk = df_risk[["event_time", "reason"]].copy()
        risk["symbol"] = ""
        risk["status"] = ""
        risk = risk.rename(columns={"reason": "event_detail"})
        risk["event_type"] = "risk"
        trace_parts.append(risk)
    if not df_hb.empty and "heartbeat_time" in df_hb.columns:
        heartbeats = df_hb[["heartbeat_time", "status"]].copy()
        heartbeats = heartbeats.rename(columns={"heartbeat_time": "event_time"})
        heartbeats["symbol"] = ""
        heartbeats["event_detail"] = "heartbeat"
        heartbeats["event_type"] = "heartbeat"
        trace_parts.append(heartbeats)
    if not df_order_states.empty and "event_time" in df_order_states.columns:
        order_states = df_order_states[["event_time", "symbol", "state", "message"]].copy()
        order_states = order_states.rename(columns={"state": "status", "message": "event_detail"})
        order_states["event_type"] = "order_state"
        trace_parts.append(order_states)

    if not trace_parts:
        return pd.DataFrame()

    trace_df = pd.concat(trace_parts, ignore_index=True)
    return trace_df.sort_values("event_time", ascending=False).head(500)


__all__ = [
    "build_heartbeat_interval_figure",
    "build_order_state_figure",
    "build_risk_reason_figure",
    "build_strategy_process_trace_frame",
    "prepare_heartbeat_interval_frame",
]
