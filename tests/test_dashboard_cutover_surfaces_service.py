from __future__ import annotations

import pandas as pd

from lumina_quant.dashboard.cutover_surfaces_service import (
    build_execution_analytics_payload,
    build_market_data_payload,
    build_optimization_insights_payload,
    build_performance_price_payload,
    build_raw_data_payload,
    build_report_export_payload,
    compute_trade_analytics,
    load_execution_analytics_payload,
    load_market_data_payload,
    load_optimization_insights_payload,
    load_performance_price_payload,
    load_raw_data_payload,
    load_report_export_payload,
)


def _overview_payload() -> dict[str, object]:
    return {
        "source": {
            "status": "ok",
            "run_id": "run-123",
            "mode": "next",
            "backend": "python",
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
    assert "Default Launcher: next" in payload["markdown_report"]
    assert payload["filenames"]["json"].endswith("-dashboard-report.json")
    assert "Market Data route is available in Next.js." in payload["cutover_gate"]["evidence"]


def test_build_market_data_payload_exposes_context_and_recent_bars() -> None:
    fills = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "quantity": 1.0,
                "price": 100.0,
                "commission": 0.0,
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "datetime": "2026-03-21T00:00:00Z",
                "open": 99.0,
                "high": 101.0,
                "low": 98.5,
                "close": 100.0,
                "volume": 12.0,
            },
            {
                "datetime": "2026-03-21T00:01:00Z",
                "open": 100.0,
                "high": 102.0,
                "low": 99.5,
                "close": 101.5,
                "volume": 18.0,
            },
        ]
    )

    payload = build_market_data_payload(
        run_row={"run_id": "run-123", "strategy": "RsiStrategy"},
        fills_frame=fills,
        market_frame=market,
    )

    assert payload["status"] == "ok"
    assert payload["market_context"]["symbol"] == "BTC/USDT"
    assert payload["summary_metrics"][0]["value"] == 2
    assert payload["recent_bars"][-1]["close"] == 101.5


def test_build_optimization_insights_payload_exposes_stage_breakdown_and_best_candidate() -> None:
    optimization = pd.DataFrame(
        [
            {
                "created_at": "2026-03-21T00:00:00Z",
                "run_id": "run-123",
                "stage": "explore",
                "sharpe": 1.1,
                "train_sharpe": 1.3,
                "robustness_score": 0.7,
                "cagr": 0.2,
                "mdd": -0.1,
                "params": {"window": 10},
            },
            {
                "created_at": "2026-03-21T00:05:00Z",
                "run_id": "run-123",
                "stage": "promote",
                "sharpe": 1.4,
                "train_sharpe": 1.5,
                "robustness_score": 0.8,
                "cagr": 0.24,
                "mdd": -0.08,
                "params": {"window": 20},
            },
        ]
    )

    payload = build_optimization_insights_payload(
        run_id="run-123",
        optimization_frame=optimization,
    )

    assert payload["status"] == "ok"
    assert payload["summary_metrics"][0]["value"] == 2
    assert payload["stage_breakdown"][0]["stage"] == "explore"
    assert payload["best_candidate"]["stage"] == "promote"
    assert payload["top_candidates"][0]["params"] == {"window": 20}


def test_build_raw_data_payload_exposes_frame_summaries_and_previews() -> None:
    payload = build_raw_data_payload(
        run_id="run-123",
        context={"source": "postgres", "market": "BTC/USDT 1m (binance)"},
        frames=[
            ("Runs", pd.DataFrame([{"run_id": "run-123", "status": "COMPLETED"}])),
            ("Orders", pd.DataFrame([{"status": "FILLED", "quantity": 1.0}])),
        ],
    )

    assert payload["status"] == "ok"
    assert payload["frame_summaries"] == [
        {"label": "Runs", "rows": 1, "columns": 2},
        {"label": "Orders", "rows": 1, "columns": 2},
    ]
    assert payload["previews"][0]["rows"][0]["run_id"] == "run-123"


def test_cutover_surface_loaders_short_circuit_on_blank_dsn() -> None:
    assert load_performance_price_payload(dsn="")["status"] == "missing_dsn"
    assert load_execution_analytics_payload(dsn="")["status"] == "missing_dsn"
    assert load_market_data_payload(dsn="")["status"] == "missing_dsn"
    assert load_optimization_insights_payload(dsn="")["status"] == "missing_dsn"
    assert load_raw_data_payload(dsn="")["status"] == "missing_dsn"
    assert load_report_export_payload(dsn="")["status"] == "missing_dsn"
