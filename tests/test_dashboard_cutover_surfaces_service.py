from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from lumina_quant.dashboard.cutover_surfaces_service import (
    build_execution_analytics_payload,
    build_performance_price_payload,
    build_report_export_payload,
    compute_trade_analytics,
    load_execution_analytics_payload,
    load_performance_price_payload,
    load_report_export_payload,
)


def _overview_payload() -> dict[str, object]:
    return {
        "source": {
            "status": "ok",
            "run_id": "run-123",
            "mode": "next",
            "backend": "streamlit",
        },
        "summary_metrics": [
            {"key": "latest_equity", "label": "Latest Equity", "value": 1105.0},
            {"key": "total_return", "label": "Total Return", "value": 0.105},
        ],
        "performance_metrics": {
            "cagr": 0.22,
            "max_drawdown": -0.08,
        },
        "equity_curve": [
            {"timestamp": "2026-03-21T00:00:00+00:00", "equity": 1000.0},
            {"timestamp": "2026-03-22T00:00:00+00:00", "equity": 1105.0},
        ],
        "drawdown_curve": [
            {"timestamp": "2026-03-21T00:00:00+00:00", "drawdown": 0.0},
            {"timestamp": "2026-03-22T00:00:00+00:00", "drawdown": -0.08},
        ],
    }


def test_compute_trade_analytics_tracks_realized_pnl_and_position() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
        ]
    )

    analytics = compute_trade_analytics(fills)

    assert analytics["position_after"].tolist() == [2.0, 1.0]
    assert analytics["realized_pnl"].tolist() == [0.0, 9.5]
    assert analytics["close_side"].tolist() == [None, "LONG"]


def test_build_performance_price_payload_exposes_benchmark_and_trade_markers() -> None:
    metrics = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "total": 1000.0,
                "benchmark_price": 100.0,
                "funding": 0.0,
            },
            {
                "datetime": "2026-03-22T00:00:00Z",
                "total": 1105.0,
                "benchmark_price": 108.0,
                "funding": 1.5,
            },
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
        ]
    )

    payload = build_performance_price_payload(
        overview_payload=_overview_payload(),
        metrics_frame=metrics,
        fills_frame=fills,
    )

    assert payload["status"] == "ok"
    assert payload["benchmark_curve"][-1]["price"] == 108.0
    assert payload["funding_curve"][-1]["funding"] == 1.5
    assert payload["trade_markers"][-1]["realized_pnl"] == 9.5


def test_build_execution_analytics_payload_summarizes_fills_orders_and_streaks() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
            {
                "datetime": "2026-03-21T00:10:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 95.0,
                "commission": 0.5,
            },
        ]
    )
    orders = pd.DataFrame(
        [
            {"status": "FILLED"},
            {"status": "CANCELLED"},
            {"status": "FILLED"},
        ]
    )

    payload = build_execution_analytics_payload(
        run_id="run-123",
        fills_frame=fills,
        orders_frame=orders,
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["buy_fills"] == 1
    assert payload["summary"]["sell_fills"] == 2
    assert payload["summary"]["closed_trade_count"] == 2
    assert payload["summary"]["win_streak_max"] == 1
    assert payload["summary"]["loss_streak_max"] == 1
    assert payload["direction_breakdown"][0] == {
        "direction": "Long",
        "closed_trades": 2,
        "win_rate": 0.5,
    }
    assert payload["order_status"][0]["count"] == 2


def test_build_report_export_payload_generates_json_and_markdown() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 2.0,
                "price": 100.0,
                "commission": 0.0,
            },
            {
                "datetime": "2026-03-21T00:05:00Z",
                "symbol": "BTC/USDT",
                "direction": "SELL",
                "quantity": 1.0,
                "price": 110.0,
                "commission": 0.5,
            },
        ]
    )

    payload = build_report_export_payload(
        run_row={
            "run_id": "run-123",
            "strategy": "RsiStrategy",
            "mode": "backtest",
            "status": "COMPLETED",
            "started_at": "2026-03-21T00:00:00Z",
        },
        overview_payload=_overview_payload(),
        fills_frame=fills,
        risk_frame=pd.DataFrame([{"event_time": "2026-03-22T00:00:00Z"}]),
        heartbeats_frame=pd.DataFrame([{"heartbeat_time": "2026-03-22T00:00:00Z"}]),
    )

    assert payload["status"] == "ok"
    assert payload["json_report"]["closed_trade_count"] == 1
    assert payload["json_report"]["risk_event_count"] == 1
    assert "Streamlit remains the default launcher" in payload["markdown_report"]
    assert payload["filenames"]["json"].endswith("-dashboard-report.json")


def test_cutover_surface_loaders_short_circuit_on_blank_dsn() -> None:
    assert load_performance_price_payload(dsn="")["status"] == "missing_dsn"
    assert load_execution_analytics_payload(dsn="")["status"] == "missing_dsn"
    assert load_report_export_payload(dsn="")["status"] == "missing_dsn"
