"""Python-backed payload builders for the Next dashboard cutover-prep surfaces."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from apps.dashboard.services.state_store import (
    load_fills_state_frame,
    load_heartbeats_state_frame,
    load_metrics_state_frame,
    load_orders_state_frame,
    load_risk_events_state_frame,
    load_runs_frame,
)
from lumina_quant.dashboard.bridge import resolve_dashboard_bridge_contract
from lumina_quant.dashboard.overview_service import (
    build_overview_payload_from_frames,
    coerce_datetime_series,
    resolve_dashboard_postgres_dsn,
)


def _dashboard_contract() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    return resolve_dashboard_bridge_contract(
        launch_mode="next",
        streamlit_app_path=repo_root / "apps" / "dashboard" / "app.py",
        next_app_dir=repo_root / "apps" / "dashboard_web",
    )


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if pd.notna(parsed) else float(default)


def _isoformat(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _empty_surface_payload(*, run_id: str = "", reason: str) -> dict[str, Any]:
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "status": reason,
    }


def compute_trade_analytics(df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_trades.empty:
        return df_trades.copy()

    df = df_trades.copy().sort_values("datetime").reset_index(drop=True)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    for column in ("quantity", "price", "fill_cost", "commission"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        else:
            df[column] = 0.0
    if "direction" not in df.columns:
        df["direction"] = ""
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"

    positions: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    entry_times: dict[str, Any] = {}
    realized_pnl: list[float] = []
    realized_return: list[float] = []
    position_after: list[float] = []
    avg_cost_after: list[float] = []
    closed_qty: list[float] = []
    close_side: list[str | None] = []
    holding_sec: list[float] = []

    for _, row in df.iterrows():
        symbol = str(row["symbol"])
        qty = float(row["quantity"])
        price = float(row["price"])
        commission = float(row["commission"])
        direction = str(row["direction"]).upper()
        signed = qty if direction == "BUY" else -qty
        event_time = row.get("datetime", pd.NaT)
        event_time = event_time if pd.notna(event_time) else pd.NaT

        pos = float(positions.get(symbol, 0.0))
        avg = float(avg_cost.get(symbol, 0.0))
        entry_time = entry_times.get(symbol)

        pnl = 0.0
        ret = float("nan")
        closes = 0.0
        close_label = None
        hold_seconds = float("nan")

        if pos == 0 or (pos > 0 and signed > 0) or (pos < 0 and signed < 0):
            new_pos = pos + signed
            if new_pos != 0:
                if pos == 0:
                    new_avg = price
                else:
                    new_avg = ((abs(pos) * avg) + (abs(signed) * price)) / abs(new_pos)
            else:
                new_avg = 0.0
            if pos == 0 and new_pos != 0:
                new_entry_time = event_time
            elif new_pos == 0:
                new_entry_time = None
            else:
                new_entry_time = entry_time
        else:
            closes = min(abs(pos), abs(signed))
            if pos > 0 and signed < 0:
                pnl = (price - avg) * closes - commission
                close_label = "LONG"
            elif pos < 0 and signed > 0:
                pnl = (avg - price) * closes - commission
                close_label = "SHORT"

            basis = max(abs(avg * closes), abs(price * closes))
            if basis > 1e-12:
                ret = pnl / basis

            if pd.notna(event_time) and entry_time is not None and pd.notna(entry_time):
                hold_seconds = max(0.0, float((event_time - entry_time).total_seconds()))

            new_pos = pos + signed
            if new_pos == 0:
                new_avg = 0.0
                new_entry_time = None
            elif (pos > 0 and new_pos > 0) or (pos < 0 and new_pos < 0):
                new_avg = avg
                new_entry_time = entry_time
            else:
                new_avg = price
                new_entry_time = event_time

        positions[symbol] = new_pos
        avg_cost[symbol] = new_avg
        entry_times[symbol] = new_entry_time
        realized_pnl.append(pnl)
        realized_return.append(ret)
        position_after.append(new_pos)
        avg_cost_after.append(new_avg)
        closed_qty.append(closes)
        close_side.append(close_label)
        holding_sec.append(hold_seconds)

    df["realized_pnl"] = realized_pnl
    df["realized_return_pct"] = pd.Series(realized_return, dtype="float64") * 100.0
    df["position_after"] = position_after
    df["avg_cost_after"] = avg_cost_after
    df["closed_qty"] = closed_qty
    df["close_side"] = close_side
    df["holding_sec"] = holding_sec
    df["cum_realized_pnl"] = df["realized_pnl"].cumsum()
    df["notional"] = df["quantity"] * df["price"]
    return df


def _closed_trade_analytics(trade_analytics: pd.DataFrame) -> pd.DataFrame:
    if trade_analytics.empty:
        return trade_analytics
    if "closed_qty" in trade_analytics.columns:
        return trade_analytics[trade_analytics["closed_qty"] > 0].copy()
    return trade_analytics[trade_analytics["realized_pnl"] != 0].copy()


def _streak_groups(outcomes: list[bool]) -> list[tuple[bool, int]]:
    if not outcomes:
        return []
    groups: list[tuple[bool, int]] = []
    current = outcomes[0]
    length = 1
    for outcome in outcomes[1:]:
        if outcome == current:
            length += 1
            continue
        groups.append((current, length))
        current = outcome
        length = 1
    groups.append((current, length))
    return groups


def _load_runs(dsn: str, *, run_limit: int) -> pd.DataFrame:
    return load_runs_frame(
        dsn,
        coerce_datetime=coerce_datetime_series,
        limit=run_limit,
    )


def _load_metrics(dsn: str, run_id: str, *, point_limit: int) -> pd.DataFrame:
    return load_metrics_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        parse_json_dict=_parse_json_dict,
        max_points=point_limit,
    )


def _load_fills(dsn: str, run_id: str, *, fill_limit: int) -> pd.DataFrame:
    return load_fills_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=fill_limit,
    )


def _load_orders(dsn: str, run_id: str, *, order_limit: int) -> pd.DataFrame:
    return load_orders_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=order_limit,
    )


def _load_risk_events(dsn: str, run_id: str, *, limit: int) -> pd.DataFrame:
    return load_risk_events_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=limit,
    )


def _load_heartbeats(dsn: str, run_id: str, *, limit: int) -> pd.DataFrame:
    return load_heartbeats_state_frame(
        dsn,
        run_id,
        coerce_datetime=coerce_datetime_series,
        max_points=limit,
    )


def build_performance_price_payload(
    *,
    overview_payload: dict[str, Any],
    metrics_frame: pd.DataFrame,
    fills_frame: pd.DataFrame,
) -> dict[str, Any]:
    payload = {
        **_empty_surface_payload(
            run_id=str(overview_payload.get("source", {}).get("run_id") or ""),
            reason=str(overview_payload.get("source", {}).get("status") or "unknown"),
        ),
        "source": overview_payload.get("source", {}),
        "summary_metrics": overview_payload.get("summary_metrics", []),
        "performance_metrics": overview_payload.get("performance_metrics", {}),
        "equity_curve": overview_payload.get("equity_curve", []),
        "drawdown_curve": overview_payload.get("drawdown_curve", []),
        "benchmark_curve": [],
        "funding_curve": [],
        "trade_markers": [],
    }
    if payload["status"] != "ok":
        return payload

    if not metrics_frame.empty:
        benchmark_series = pd.to_numeric(metrics_frame.get("benchmark_price"), errors="coerce")
        funding_series = pd.to_numeric(metrics_frame.get("funding"), errors="coerce").fillna(0.0)
        payload["benchmark_curve"] = [
            {
                "timestamp": _isoformat(row.datetime),
                "price": float(row.benchmark_price),
            }
            for row in metrics_frame.assign(benchmark_price=benchmark_series).itertuples(index=False)
            if pd.notna(row.benchmark_price) and _isoformat(row.datetime) is not None
        ]
        payload["funding_curve"] = [
            {
                "timestamp": _isoformat(row.datetime),
                "funding": float(row.funding),
            }
            for row in metrics_frame.assign(funding=funding_series).itertuples(index=False)
            if _isoformat(row.datetime) is not None
        ]

    trade_analytics = compute_trade_analytics(fills_frame)
    payload["trade_markers"] = [
        {
            "timestamp": _isoformat(row.datetime),
            "symbol": str(row.symbol),
            "direction": str(row.direction),
            "price": _safe_float(row.price),
            "quantity": _safe_float(row.quantity),
            "realized_pnl": _safe_float(row.realized_pnl),
            "realized_return_pct": None if pd.isna(row.realized_return_pct) else _safe_float(row.realized_return_pct),
            "position_after": _safe_float(row.position_after),
        }
        for row in trade_analytics.tail(12).itertuples(index=False)
        if _isoformat(row.datetime) is not None
    ]
    return payload


def build_execution_analytics_payload(
    *,
    run_id: str,
    fills_frame: pd.DataFrame,
    orders_frame: pd.DataFrame,
) -> dict[str, Any]:
    payload = {
        **_empty_surface_payload(run_id=run_id, reason="ok"),
        "summary": {
            "buy_fills": 0,
            "sell_fills": 0,
            "avg_qty": 0.0,
            "avg_notional": 0.0,
            "total_commission": 0.0,
            "avg_trade_return_pct": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "win_streak_max": 0,
            "loss_streak_max": 0,
            "win_streak_avg": 0.0,
            "loss_streak_avg": 0.0,
            "holding_time_avg_sec": 0.0,
            "long_trades": 0,
            "long_win_rate": 0.0,
            "short_trades": 0,
            "short_win_rate": 0.0,
            "order_count": int(len(orders_frame.index)),
            "closed_trade_count": 0,
        },
        "direction_breakdown": [],
        "order_status": [],
        "recent_closed_trades": [],
    }

    if fills_frame.empty and orders_frame.empty:
        payload["status"] = "no_execution_data"
        return payload

    trade_analytics = compute_trade_analytics(fills_frame)
    closed = _closed_trade_analytics(trade_analytics)
    summary = payload["summary"]
    if not fills_frame.empty:
        direction = fills_frame.get("direction", pd.Series(dtype="object")).fillna("").astype(str).str.upper()
        summary["buy_fills"] = int((direction == "BUY").sum())
        summary["sell_fills"] = int((direction == "SELL").sum())
        summary["avg_qty"] = round(_safe_float(fills_frame.get("quantity", pd.Series(dtype="float64")).mean()), 6)
        summary["avg_notional"] = round(_safe_float(trade_analytics.get("notional", pd.Series(dtype="float64")).mean()), 6)
        summary["total_commission"] = round(
            _safe_float(fills_frame.get("commission", pd.Series(dtype="float64")).sum()),
            6,
        )

    if not closed.empty:
        returns = pd.to_numeric(closed["realized_return_pct"], errors="coerce").dropna()
        pnls = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0)
        decisive = pnls[pnls != 0.0]
        summary["closed_trade_count"] = int(len(closed.index))
        summary["avg_trade_return_pct"] = round(_safe_float(returns.mean()), 6)
        summary["best_trade_pnl"] = round(_safe_float(pnls.max()), 6)
        summary["worst_trade_pnl"] = round(_safe_float(pnls.min()), 6)
        summary["holding_time_avg_sec"] = round(
            _safe_float(pd.to_numeric(closed["holding_sec"], errors="coerce").dropna().mean()),
            6,
        )

        streaks = _streak_groups((decisive > 0.0).tolist())
        wins = [length for is_win, length in streaks if is_win]
        losses = [length for is_win, length in streaks if not is_win]
        summary["win_streak_max"] = max(wins, default=0)
        summary["loss_streak_max"] = max(losses, default=0)
        summary["win_streak_avg"] = round(sum(wins) / len(wins), 6) if wins else 0.0
        summary["loss_streak_avg"] = round(sum(losses) / len(losses), 6) if losses else 0.0

        for label, close_side in (("Long", "LONG"), ("Short", "SHORT")):
            part = closed[closed["close_side"] == close_side]
            trade_count = int(len(part.index))
            win_rate = 0.0
            if trade_count:
                win_rate = float((pd.to_numeric(part["realized_pnl"], errors="coerce").fillna(0.0) > 0.0).mean())
            payload["direction_breakdown"].append(
                {
                    "direction": label,
                    "closed_trades": trade_count,
                    "win_rate": round(win_rate, 6),
                }
            )
            if label == "Long":
                summary["long_trades"] = trade_count
                summary["long_win_rate"] = round(win_rate, 6)
            else:
                summary["short_trades"] = trade_count
                summary["short_win_rate"] = round(win_rate, 6)

        payload["recent_closed_trades"] = [
            {
                "timestamp": _isoformat(row.datetime),
                "symbol": str(row.symbol),
                "close_side": str(row.close_side or "n/a"),
                "realized_pnl": _safe_float(row.realized_pnl),
                "realized_return_pct": None if pd.isna(row.realized_return_pct) else _safe_float(row.realized_return_pct),
                "holding_sec": None if pd.isna(row.holding_sec) else _safe_float(row.holding_sec),
            }
            for row in closed.tail(10).itertuples(index=False)
            if _isoformat(row.datetime) is not None
        ]

    if not orders_frame.empty:
        order_status = (
            orders_frame.get("status", pd.Series(dtype="object"))
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
        )
        payload["order_status"] = [
            {"status": str(status), "count": int(count)}
            for status, count in order_status.items()
        ]

    return payload


def _build_markdown_report(report: dict[str, Any], cutover_gate: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in cutover_gate["evidence"])
    return (
        f"# {report['title']}\n\n"
        f"- Generated: {report['generated_at']}\n"
        f"- Run ID: {report['run_id'] or 'n/a'}\n"
        f"- Strategy: {report['strategy'] or 'unknown'}\n"
        f"- Total Return: {report['total_return']}\n"
        f"- Latest Equity: {report['latest_equity']}\n"
        f"- Realized PnL: {report['realized_pnl']}\n"
        f"- Closed Trades: {report['closed_trade_count']}\n"
        f"- Risk Events: {report['risk_event_count']}\n"
        f"- Heartbeats: {report['heartbeat_count']}\n"
        f"- Default Launcher: {cutover_gate['default_launcher']}\n\n"
        f"## Cutover Gate Evidence\n{evidence}\n"
    )


def build_report_export_payload(
    *,
    run_row: dict[str, Any],
    overview_payload: dict[str, Any],
    fills_frame: pd.DataFrame,
    risk_frame: pd.DataFrame,
    heartbeats_frame: pd.DataFrame,
) -> dict[str, Any]:
    run_id = str(run_row.get("run_id") or "")
    status = str(overview_payload.get("source", {}).get("status") or "unknown")
    payload = {
        **_empty_surface_payload(run_id=run_id, reason=status),
        "filenames": {
            "json": "luminaquant-dashboard-report.json",
            "markdown": "luminaquant-dashboard-report.md",
        },
        "json_report": {},
        "markdown_report": "",
        "cutover_gate": {
            "default_launcher": "streamlit",
            "status": "guarded",
            "evidence": [
                "Performance & Price route is available in Next.js.",
                "Execution Analytics route is available in Next.js.",
                "Report Export route is available in Next.js.",
                "Streamlit remains the default launcher until full cutover review is approved.",
            ],
        },
    }
    if status != "ok":
        return payload

    trade_analytics = compute_trade_analytics(fills_frame)
    closed = _closed_trade_analytics(trade_analytics)
    latest_equity = None
    total_return = None
    for metric in overview_payload.get("summary_metrics", []):
        key = str(metric.get("key") or "")
        if key == "latest_equity":
            latest_equity = metric.get("value")
        elif key == "total_return":
            total_return = metric.get("value")

    generated_at = datetime.now(UTC).isoformat()
    report = {
        "title": "LuminaQuant Dashboard Snapshot",
        "generated_at": generated_at,
        "run_id": run_id,
        "strategy": str(run_row.get("strategy") or "unknown"),
        "mode": str(run_row.get("mode") or ""),
        "status": str(run_row.get("status") or ""),
        "period_start": _isoformat(run_row.get("started_at")),
        "period_end": overview_payload.get("equity_curve", [{}])[-1].get("timestamp") if overview_payload.get("equity_curve") else None,
        "total_return": total_return,
        "latest_equity": latest_equity,
        "realized_pnl": round(_safe_float(closed.get("realized_pnl", pd.Series(dtype="float64")).sum()), 6),
        "closed_trade_count": int(len(closed.index)),
        "risk_event_count": int(len(risk_frame.index)),
        "heartbeat_count": int(len(heartbeats_frame.index)),
        "performance_metrics": overview_payload.get("performance_metrics", {}),
    }
    date_prefix = generated_at[:10]
    run_token = run_id or "no-run"
    payload["filenames"] = {
        "json": f"{date_prefix}-{run_token}-dashboard-report.json",
        "markdown": f"{date_prefix}-{run_token}-dashboard-report.md",
    }
    payload["json_report"] = report
    payload["markdown_report"] = _build_markdown_report(report, payload["cutover_gate"])
    return payload


def load_performance_price_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 80,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "source": {"status": "missing_dsn"},
            "summary_metrics": [],
            "performance_metrics": {},
            "equity_curve": [],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "funding_curve": [],
            "trade_markers": [],
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "source": {"status": "missing_dsn"},
            "summary_metrics": [],
            "performance_metrics": {},
            "equity_curve": [],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "funding_curve": [],
            "trade_markers": [],
        }

    runs = _load_runs(resolved_dsn, run_limit=10)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "source": {"status": "no_runs"},
            "summary_metrics": [],
            "performance_metrics": {},
            "equity_curve": [],
            "drawdown_curve": [],
            "benchmark_curve": [],
            "funding_curve": [],
            "trade_markers": [],
        }

    run_id = str(runs.iloc[0]["run_id"] or "")
    metrics = _load_metrics(resolved_dsn, run_id, point_limit=point_limit)
    fills = _load_fills(resolved_dsn, run_id, fill_limit=fill_limit)
    overview_payload = build_overview_payload_from_frames(
        contract=_dashboard_contract(),
        runs_frame=runs,
        equity_frame=metrics[["datetime", "total"]].copy() if {"datetime", "total"}.issubset(metrics.columns) else pd.DataFrame(),
    )
    return build_performance_price_payload(
        overview_payload=overview_payload,
        metrics_frame=metrics,
        fills_frame=fills,
    )


def load_execution_analytics_payload(
    *,
    dsn: str | None = None,
    fill_limit: int = 200,
    order_limit: int = 200,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "summary": {},
            "direction_breakdown": [],
            "order_status": [],
            "recent_closed_trades": [],
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "summary": {},
            "direction_breakdown": [],
            "order_status": [],
            "recent_closed_trades": [],
        }
    runs = _load_runs(resolved_dsn, run_limit=1)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "summary": {},
            "direction_breakdown": [],
            "order_status": [],
            "recent_closed_trades": [],
        }
    run_id = str(runs.iloc[0]["run_id"] or "")
    fills = _load_fills(resolved_dsn, run_id, fill_limit=fill_limit)
    orders = _load_orders(resolved_dsn, run_id, order_limit=order_limit)
    return build_execution_analytics_payload(
        run_id=run_id,
        fills_frame=fills,
        orders_frame=orders,
    )


def load_report_export_payload(
    *,
    dsn: str | None = None,
    point_limit: int = 240,
    fill_limit: int = 200,
    event_limit: int = 50,
) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "filenames": {},
            "json_report": {},
            "markdown_report": "",
            "cutover_gate": {},
        }
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return {
            **_empty_surface_payload(reason="missing_dsn"),
            "filenames": {},
            "json_report": {},
            "markdown_report": "",
            "cutover_gate": {},
        }
    runs = _load_runs(resolved_dsn, run_limit=10)
    if runs.empty:
        return {
            **_empty_surface_payload(reason="no_runs"),
            "filenames": {},
            "json_report": {},
            "markdown_report": "",
            "cutover_gate": {},
        }
    run_row = runs.iloc[0].to_dict()
    run_id = str(run_row.get("run_id") or "")
    metrics = _load_metrics(resolved_dsn, run_id, point_limit=point_limit)
    fills = _load_fills(resolved_dsn, run_id, fill_limit=fill_limit)
    risk = _load_risk_events(resolved_dsn, run_id, limit=event_limit)
    heartbeats = _load_heartbeats(resolved_dsn, run_id, limit=event_limit)
    overview_payload = build_overview_payload_from_frames(
        contract=_dashboard_contract(),
        runs_frame=runs,
        equity_frame=metrics[["datetime", "total"]].copy() if {"datetime", "total"}.issubset(metrics.columns) else pd.DataFrame(),
    )
    return build_report_export_payload(
        run_row=run_row,
        overview_payload=overview_payload,
        fills_frame=fills,
        risk_frame=risk,
        heartbeats_frame=heartbeats,
    )


__all__ = [
    "build_execution_analytics_payload",
    "build_performance_price_payload",
    "build_report_export_payload",
    "compute_trade_analytics",
    "load_execution_analytics_payload",
    "load_performance_price_payload",
    "load_report_export_payload",
]
