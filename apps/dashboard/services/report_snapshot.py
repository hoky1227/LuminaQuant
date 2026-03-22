"""Dashboard snapshot-report helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def serialize_balance_equity_frame(
    df_balance_equity: pd.DataFrame,
    *,
    limit: int,
    downsample_frame,
    safe_float,
) -> list[dict[str, Any]]:
    if df_balance_equity.empty:
        return []
    view = df_balance_equity[
        ["datetime", "equity", "balance", "open_pnl", "drawdown_signed"]
    ].copy()
    if len(view) > int(limit):
        view = downsample_frame(view, int(limit))
    payload: list[dict[str, Any]] = []
    for row in view.itertuples(index=False):
        dt = pd.to_datetime(row.datetime, errors="coerce")
        payload.append(
            {
                "datetime": dt.isoformat() if pd.notna(dt) else None,
                "equity": safe_float(row.equity, 0.0),
                "balance": safe_float(row.balance, 0.0),
                "open_pnl": safe_float(row.open_pnl, 0.0),
                "drawdown": safe_float(row.drawdown_signed, 0.0),
            }
        )
    return payload


def build_report_payload(
    *,
    summary: dict[str, Any],
    performance: dict[str, Any],
    run_id: str | None,
    source: str,
    strategy_name: str,
    period_preset: str,
    df_equity: pd.DataFrame,
    df_trades: pd.DataFrame,
    df_risk: pd.DataFrame,
    df_hb: pd.DataFrame,
    runtime_overrides: dict[str, Any],
    strategy_params: dict[str, Any],
    build_mt5_summary_rows,
    build_monthly_returns_table,
    mirror_snapshot: dict[str, Any] | None = None,
    balance_equity_series: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    perf_export = {
        key: value
        for key, value in performance.items()
        if key not in {"benchmark_series", "return_series", "cum_return_series"}
    }
    mt5_rows = build_mt5_summary_rows(summary)
    monthly_table = build_monthly_returns_table(df_equity, performance)
    monthly_payload: dict[str, dict[str, float | None]] = {}
    if not monthly_table.empty:
        monthly_payload = {
            str(idx): {
                str(col): (
                    None
                    if pd.isna(monthly_table.loc[idx, col])
                    else float(monthly_table.loc[idx, col])
                )
                for col in monthly_table.columns
            }
            for idx in monthly_table.index
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source": source,
        "strategy": strategy_name,
        "strategy_params": strategy_params,
        "period_preset": period_preset,
        "runtime_overrides": runtime_overrides,
        "summary": summary,
        "performance": perf_export,
        "equity_rows": len(df_equity),
        "trade_rows": len(df_trades),
        "risk_rows": len(df_risk),
        "heartbeat_rows": len(df_hb),
        "mt5_summary": mt5_rows.to_dict(orient="records"),
        "monthly_returns": monthly_payload,
        "mirror_snapshot": mirror_snapshot or {},
        "balance_equity_series": balance_equity_series or [],
    }


def build_dashboard_report_runtime_overrides(
    *,
    runner_initial_capital,
    runner_leverage,
    runner_symbols,
    runner_timeframe,
    runner_data_source,
    runner_timeout_sec,
) -> dict[str, Any]:
    return {
        "initial_capital": float(runner_initial_capital),
        "backtest_leverage": int(runner_leverage),
        "symbols": runner_symbols,
        "timeframe": runner_timeframe,
        "runner_data_source": runner_data_source,
        "runner_timeout_sec": int(runner_timeout_sec),
    }


def build_report_markdown(
    payload: dict[str, Any],
    *,
    safe_float,
    format_signed_dollar,
    format_metric_value,
    format_duration_seconds,
) -> str:
    summary = payload["summary"]
    mirror = payload.get("mirror_snapshot") or {}
    balance_series = payload.get("balance_equity_series") or []
    mirror_block = ""
    if mirror:
        mirror_block = (
            f"\n## Mirror KPI Strip\n"
            f"- Total Trades: {int(mirror.get('total_trades', 0)):,} "
            f"({int(mirror.get('wins', 0))}W / {int(mirror.get('losses', 0))}L)\n"
            f"- Win Rate: {safe_float(mirror.get('win_rate'), 0.0):.2%}\n"
            f"- Closed PnL: {format_signed_dollar(mirror.get('closed_pnl'), digits=2)}\n"
            f"- Open P/L: {format_signed_dollar(mirror.get('open_pnl'), digits=2)}\n"
            f"- Total (C+O): {format_signed_dollar(mirror.get('total_c_plus_o'), digits=2)}\n"
            f"- Equity MDD: ${safe_float(mirror.get('equity_mdd'), 0.0):,.2f} "
            f"({safe_float(mirror.get('equity_mdd_rel'), 0.0):.2%})\n"
            f"- R/MDD: {safe_float(mirror.get('r_mdd'), 0.0):.2f}x\n"
        )

    return (
        f"# Dashboard Snapshot Report\n\n"
        f"- Generated: {payload['generated_at']}\n"
        f"- Run ID: {payload['run_id']}\n"
        f"- Source: {payload['source']}\n"
        f"- Strategy: {payload['strategy']}\n\n"
        f"## Summary\n"
        f"- Period: {summary['period_start']} -> {summary['period_end']}\n"
        f"- Bars: {summary['bars']}\n"
        f"- Fills: {summary['fills']} (BUY {summary['buy_fills']} / SELL {summary['sell_fills']})\n"
        f"- Avg fills/day: {summary['fills_per_day']:.2f}\n"
        f"- Avg qty: {summary['avg_qty']:.4f}\n"
        f"- Avg notional: {summary['avg_notional']:.2f}\n"
        f"- Commission: {summary['total_commission']:.4f}\n"
        f"- Realized PnL: {summary['realized_pnl']:.4f}\n"
        f"- Win rate: {summary['win_rate']:.2%}\n"
        f"- Avg trade return: {summary['avg_trade_return_pct']:.4f}%\n"
        f"- Best trade PnL: {summary['best_trade_pnl']:.4f}\n"
        f"- Worst trade PnL: {summary['worst_trade_pnl']:.4f}\n"
        f"- Gross Profit / Gross Loss: {summary['gross_profit']:.4f} / {summary['gross_loss']:.4f}\n"
        f"- Profit Factor: {format_metric_value('Profit Factor', summary['profit_factor'])}\n"
        f"- Recovery Factor: {summary['recovery_factor']:.4f}\n"
        f"- Long Trades (Win %): {summary['long_trades_win_pct']}\n"
        f"- Short Trades (Win %): {summary['short_trades_win_pct']}\n"
        f"- Holding (min/avg/max): {format_duration_seconds(summary['holding_time_min_sec'])} / "
        f"{format_duration_seconds(summary['holding_time_avg_sec'])} / "
        f"{format_duration_seconds(summary['holding_time_max_sec'])}\n"
        f"\n## Drawdown\n"
        f"- Equity DD (Abs/Max/Rel): {summary['equity_drawdown_absolute']:.4f} / "
        f"{summary['equity_drawdown_maximal']:.4f} / {summary['equity_drawdown_relative_pct']:.2%}\n"
        f"- Balance DD (Abs/Max/Rel): {summary['balance_drawdown_absolute']:.4f} / "
        f"{summary['balance_drawdown_maximal']:.4f} / {summary['balance_drawdown_relative_pct']:.2%}\n"
        f"\n## Streaks\n"
        f"- Max win/loss streak: {int(summary['win_streak_max'])} / {int(summary['loss_streak_max'])}\n"
        f"- Avg win/loss streak: {summary['win_streak_avg']:.2f} / {summary['loss_streak_avg']:.2f}\n"
        f"- Max consecutive profit/loss: {summary['max_consecutive_profit_amount']:.4f} / "
        f"{summary['max_consecutive_loss_amount']:.4f}\n"
        f"\n## Export Payload\n"
        f"- Balance/Equity points: {len(balance_series)}\n"
        f"{mirror_block}"
    )


def save_report_snapshot(
    payload: dict[str, Any],
    *,
    out_dir: Path,
    markdown_builder,
) -> tuple[str, str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"dashboard_report_{ts}.json"
    md_path = out_dir / f"dashboard_report_{ts}.md"
    markdown = markdown_builder(payload)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown)
    return str(json_path), str(md_path), markdown
