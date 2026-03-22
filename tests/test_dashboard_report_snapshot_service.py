from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "report_snapshot.py"
SPEC = importlib.util.spec_from_file_location("dashboard_report_snapshot", MODULE_PATH)
report_snapshot = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(report_snapshot)


def test_serialize_balance_equity_frame_uses_downsample_and_safe_float() -> None:
    frame = pd.DataFrame(
        [
            {
                "datetime": "2026-03-22T00:00:00Z",
                "equity": "101.5",
                "balance": 100.0,
                "open_pnl": "1.5",
                "drawdown_signed": "-0.1",
            },
            {
                "datetime": "2026-03-22T00:01:00Z",
                "equity": "102.5",
                "balance": 101.0,
                "open_pnl": "1.5",
                "drawdown_signed": "-0.2",
            },
        ]
    )
    calls: dict[str, object] = {}

    payload = report_snapshot.serialize_balance_equity_frame(
        frame,
        limit=1,
        downsample_frame=lambda df, limit: calls.update({"rows": len(df), "limit": limit}) or df.iloc[:1],
        safe_float=lambda value, default=0.0: float(value) if value is not None else default,
    )

    assert calls == {"rows": 2, "limit": 1}
    assert payload == [
        {
            "datetime": "2026-03-22T00:00:00+00:00",
            "equity": 101.5,
            "balance": 100.0,
            "open_pnl": 1.5,
            "drawdown": -0.1,
        }
    ]


def test_build_report_payload_filters_series_fields_and_serializes_monthlies() -> None:
    monthly_table = pd.DataFrame(
        [[1.25, None]],
        index=["2026"],
        columns=["Jan", "Feb"],
    )
    payload = report_snapshot.build_report_payload(
        summary={"bars": 10},
        performance={
            "sharpe": 1.2,
            "benchmark_series": [1, 2],
            "return_series": [3],
            "cum_return_series": [4],
        },
        run_id="run-1",
        source="postgres",
        strategy_name="RsiStrategy",
        period_preset="30d",
        df_equity=pd.DataFrame([{"equity": 1}]),
        df_trades=pd.DataFrame([{"trade": 1}, {"trade": 2}]),
        df_risk=pd.DataFrame(),
        df_hb=pd.DataFrame([{"hb": 1}]),
        runtime_overrides={"symbols": ["BTC/USDT"]},
        strategy_params={"lookback": 12},
        build_mt5_summary_rows=lambda summary: pd.DataFrame([{"bars": summary["bars"]}]),
        build_monthly_returns_table=lambda _equity, _performance: monthly_table,
        mirror_snapshot={"wins": 1},
        balance_equity_series=[{"datetime": "2026-03-22T00:00:00Z"}],
    )

    assert payload["performance"] == {"sharpe": 1.2}
    assert payload["mt5_summary"] == [{"bars": 10}]
    assert payload["monthly_returns"] == {"2026": {"Jan": 1.25, "Feb": None}}
    assert payload["equity_rows"] == 1
    assert payload["trade_rows"] == 2
    assert payload["heartbeat_rows"] == 1


def test_build_monthly_returns_table_groups_returns_by_calendar_month() -> None:
    equity = pd.DataFrame(
        {
            "datetime": [
                "2026-01-31T00:00:00Z",
                "2026-02-01T00:00:00Z",
                "2026-02-02T00:00:00Z",
                "2026-03-01T00:00:00Z",
            ]
        }
    )

    monthly = report_snapshot.build_monthly_returns_table(
        equity,
        {"return_series": [0.10, -0.05, 0.20]},
    )

    assert list(monthly.columns[:3]) == ["Jan", "Feb", "Mar"]
    assert round(float(monthly.loc[2026, "Feb"]), 6) == 0.045
    assert round(float(monthly.loc[2026, "Mar"]), 6) == 0.2


def test_format_metric_value_formats_percentages_durations_and_profit_factor() -> None:
    def safe_float(value, default=0.0):
        return default if value is None else float(value)

    def format_duration_seconds(value):
        return f"{int(float(value))}s"

    assert (
        report_snapshot.format_metric_value(
            "Win Rate",
            0.125,
            safe_float=safe_float,
            format_duration_seconds=format_duration_seconds,
        )
        == "12.50%"
    )
    assert (
        report_snapshot.format_metric_value(
            "Avg Holding Time",
            90,
            safe_float=safe_float,
            format_duration_seconds=format_duration_seconds,
        )
        == "90s"
    )
    assert (
        report_snapshot.format_metric_value(
            "Profit Factor",
            "inf",
            safe_float=safe_float,
            format_duration_seconds=format_duration_seconds,
        )
        == "inf"
    )


def test_build_mt5_summary_rows_uses_formatter_across_sections() -> None:
    calls: list[str] = []
    rows = report_snapshot.build_mt5_summary_rows(
        {
            "total_net_profit": 1,
            "open_pnl": 2,
            "total_c_plus_o": 3,
            "gross_profit": 4,
            "gross_loss": -5,
            "profit_factor": 6,
            "expected_payoff": 7,
            "recovery_factor": 8,
            "r_mdd": 9,
            "ahpr": 10,
            "ghpr": 11,
            "lr_correlation": 12,
            "lr_std_error": 13,
            "z_score": 14,
            "long_trades_win_pct": "1/2",
            "short_trades_win_pct": "3/4",
            "profit_trades_text": "5",
            "loss_trades_text": "6",
            "avg_profit_trade": 15,
            "avg_loss_trade": -16,
            "payoff_ratio": 17,
            "holding_time_min_sec": 18,
            "holding_time_avg_sec": 19,
            "holding_time_max_sec": 20,
            "equity_drawdown_absolute": 21,
            "equity_drawdown_maximal": 22,
            "equity_drawdown_relative_pct": 0.23,
            "balance_drawdown_absolute": 24,
            "balance_drawdown_maximal": 25,
            "balance_drawdown_relative_pct": 0.26,
        },
        format_metric_value=lambda name, value: calls.append(name) or f"{name}={value}",
    )

    assert not rows.empty
    assert rows.iloc[0].to_dict() == {
        "Section": "Profitability",
        "Metric": "Total Net Profit",
        "Value": "Total Net Profit=1",
    }
    assert "Balance Drawdown Relative %" in calls
    assert len(calls) == len(rows)


def test_save_report_snapshot_writes_json_and_markdown(tmp_path: Path) -> None:
    payload = {"summary": {"bars": 1}, "strategy": "RsiStrategy"}

    json_path, md_path, markdown = report_snapshot.save_report_snapshot(
        payload,
        out_dir=tmp_path / "reports",
        markdown_builder=lambda current_payload: f"# {current_payload['strategy']}",
    )

    assert Path(json_path).is_file()
    assert Path(md_path).is_file()
    assert markdown == "# RsiStrategy"
    assert Path(md_path).read_text(encoding="utf-8") == "# RsiStrategy"
