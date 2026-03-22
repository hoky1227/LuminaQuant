"""Dashboard helpers for the mirror-style snapshot cards and charts."""

from __future__ import annotations

import html

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

MIRROR_DASHBOARD_CSS = """
<style>
.lq-mirror-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.65rem;
    margin: 0.2rem 0 0.7rem 0;
}
.lq-mirror-card {
    background: radial-gradient(120% 120% at 0% 0%, #102348 0%, #0b1834 68%, #081124 100%);
    border: 1px solid #17356e;
    border-radius: 12px;
    padding: 0.78rem 0.9rem;
    box-shadow: inset 0 0 0 1px rgba(13, 183, 158, 0.05), 0 8px 20px rgba(0, 0, 0, 0.18);
}
.lq-mirror-card .label {
    color: #7ea4db;
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.34rem;
}
.lq-mirror-card .value {
    font-size: 1.72rem;
    font-weight: 760;
    letter-spacing: 0.01em;
    line-height: 1.08;
    margin-bottom: 0.23rem;
    color: #dce9ff;
}
.lq-mirror-card .sub {
    color: #92b3df;
    font-size: 0.81rem;
    line-height: 1.2;
}
.lq-mirror-card.up .value {
    color: #24d37b;
}
.lq-mirror-card.down .value {
    color: #ff5f5f;
}
</style>
"""


def build_mirror_snapshot(summary, balance_equity_df, *, safe_float, safe_div) -> dict[str, float | int]:
    total_trades = int(summary.get("closed_trades", summary.get("fills", 0)) or 0)
    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    closed_pnl = safe_float(summary.get("total_net_profit", summary.get("realized_pnl", 0.0)), 0.0)
    total_c_plus_o = safe_float(summary.get("total_c_plus_o"), 0.0)
    open_pnl = safe_float(summary.get("open_pnl"), 0.0)
    if not balance_equity_df.empty:
        total_c_plus_o = safe_float(balance_equity_df["cum_total_pnl"].iloc[-1], total_c_plus_o)
        open_pnl = safe_float(balance_equity_df["open_pnl"].iloc[-1], open_pnl)

    equity_mdd = safe_float(summary.get("equity_drawdown_maximal"), 0.0)
    equity_mdd_rel = safe_float(summary.get("equity_drawdown_relative_pct"), 0.0)
    r_mdd = safe_div(total_c_plus_o, equity_mdd, 0.0)
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": safe_float(summary.get("win_rate"), 0.0),
        "closed_pnl": closed_pnl,
        "open_pnl": open_pnl,
        "total_c_plus_o": total_c_plus_o,
        "equity_mdd": equity_mdd,
        "equity_mdd_rel": equity_mdd_rel,
        "r_mdd": r_mdd,
    }


def render_mirror_cards(
    snapshot,
    *,
    safe_float,
    tone_class,
    format_signed_dollar,
    st_module=st,
    html_module=html,
    css=MIRROR_DASHBOARD_CSS,
) -> None:
    cards = [
        {
            "label": "TOTAL TRADES",
            "value": f"{int(snapshot.get('total_trades', 0)):,}",
            "sub": f"{int(snapshot.get('wins', 0))}W / {int(snapshot.get('losses', 0))}L",
            "tone": "",
        },
        {
            "label": "WIN RATE",
            "value": f"{safe_float(snapshot.get('win_rate'), 0.0):.1%}",
            "sub": "closed trades",
            "tone": tone_class(snapshot.get("win_rate")),
        },
        {
            "label": "CLOSED PNL",
            "value": format_signed_dollar(snapshot.get("closed_pnl"), digits=2),
            "sub": "realized",
            "tone": tone_class(snapshot.get("closed_pnl")),
        },
        {
            "label": "OPEN P/L",
            "value": format_signed_dollar(snapshot.get("open_pnl"), digits=2),
            "sub": "unrealized",
            "tone": tone_class(snapshot.get("open_pnl")),
        },
        {
            "label": "TOTAL (C+O)",
            "value": format_signed_dollar(snapshot.get("total_c_plus_o"), digits=2),
            "sub": "closed + open",
            "tone": tone_class(snapshot.get("total_c_plus_o")),
        },
        {
            "label": "EQUITY MDD",
            "value": format_signed_dollar(-safe_float(snapshot.get("equity_mdd"), 0.0), digits=2),
            "sub": f"{safe_float(snapshot.get('equity_mdd_rel'), 0.0):.2%}",
            "tone": tone_class(snapshot.get("equity_mdd"), invert=True),
        },
        {
            "label": "R/MDD",
            "value": f"{safe_float(snapshot.get('r_mdd'), 0.0):.2f}x",
            "sub": "total pnl / eq mdd",
            "tone": tone_class(snapshot.get("r_mdd")),
        },
    ]
    html_blocks = []
    for card in cards:
        tone = str(card.get("tone") or "")
        html_blocks.append(
            "".join(
                [
                    f"<div class='lq-mirror-card {tone}'>",
                    f"<div class='label'>{html_module.escape(str(card['label']))}</div>",
                    f"<div class='value'>{html_module.escape(str(card['value']))}</div>",
                    f"<div class='sub'>{html_module.escape(str(card['sub']))}</div>",
                    "</div>",
                ]
            )
        )
    st_module.markdown(css, unsafe_allow_html=True)
    st_module.markdown(f"<div class='lq-mirror-grid'>{''.join(html_blocks)}</div>", unsafe_allow_html=True)


def apply_mirror_figure_style(fig) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#060b17",
        plot_bgcolor="#09162e",
        font=dict(color="#adc5eb"),
        margin=dict(l=56, r=24, t=54, b=52),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.13, xanchor="left", x=0.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(45,80,136,0.32)", zeroline=False)


def build_mirror_equity_curve_figure(balance_equity_df, *, go_module=go, apply_figure_style=apply_mirror_figure_style):
    fig = go_module.Figure()
    fig.add_trace(
        go_module.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["cum_total_pnl"],
            mode="lines",
            name="Cumulative PnL",
            line=dict(color="#1fd27c", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(31,210,124,0.20)",
        )
    )
    apply_figure_style(fig)
    fig.update_layout(title="EQUITY CURVE (CUMULATIVE PNL)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(45,80,136,0.32)", tickprefix="$")
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(157,180,221,0.5)")
    return fig


def build_mirror_balance_equity_figure(
    balance_equity_df,
    snapshot,
    *,
    go_module=go,
    make_subplots_fn=make_subplots,
    apply_figure_style=apply_mirror_figure_style,
    safe_float,
):
    fig = make_subplots_fn(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go_module.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["drawdown_signed"],
            mode="lines",
            name="Drawdown",
            line=dict(color="#ff5f5f", width=1.6, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go_module.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#22d37f", width=2.2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go_module.Scatter(
            x=balance_equity_df["datetime"],
            y=balance_equity_df["balance"],
            mode="lines",
            name="Balance",
            line=dict(color="#3d84ff", width=2.0, shape="hv"),
        ),
        secondary_y=False,
    )
    apply_figure_style(fig)
    fig.update_layout(title="BALANCE & EQUITY (TIME-BASED)")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(45,80,136,0.32)",
        tickprefix="$",
        secondary_y=False,
    )
    fig.update_yaxes(
        showgrid=False,
        tickprefix="$",
        secondary_y=True,
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(157,180,221,0.5)", secondary_y=True)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=1.13,
        text=(f"Eq MDD: ${safe_float(snapshot.get('equity_mdd'), 0.0):,.2f}"),
        showarrow=False,
        font=dict(size=14, color="#ff5f5f"),
        xanchor="right",
    )
    return fig


__all__ = [
    "MIRROR_DASHBOARD_CSS",
    "apply_mirror_figure_style",
    "build_mirror_balance_equity_figure",
    "build_mirror_equity_curve_figure",
    "build_mirror_snapshot",
    "render_mirror_cards",
]
