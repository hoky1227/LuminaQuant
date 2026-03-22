import importlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lumina_quant.config import BacktestConfig, BaseConfig, OptimizationConfig
from lumina_quant.market_data import (
    normalize_symbol,
    normalize_timeframe_token,
    timeframe_to_milliseconds,
)
from lumina_quant.utils.performance import (
    create_alpha_beta,
    create_annualized_volatility,
    create_cagr,
    create_calmar_ratio,
    create_drawdowns,
    create_information_ratio,
    create_sharpe_ratio,
    create_sortino_ratio,
)
from apps.dashboard.services.workflow_jobs import (
    build_backtest_job_launch_spec as _build_backtest_job_launch_spec_data,
    build_live_job_launch_spec as _build_live_job_launch_spec_data,
    build_optimize_job_launch_spec as _build_optimize_job_launch_spec_data,
    WORKFLOW_CONTROL_DIR,
    WORKFLOW_LOG_DIR,
    WORKFLOW_RUNTIME_ROOT,
    build_runtime_env_overrides as _build_runtime_env_overrides,
    load_workflow_jobs_frame as _load_workflow_jobs_frame_data,
    build_stop_file_path as _build_stop_file_path_data,
    launch_managed_job as _launch_managed_job_data,
    refresh_workflow_jobs as _refresh_workflow_jobs_data,
    request_job_stop as _request_job_stop_data,
)
from apps.dashboard.services.process_control import (
    is_process_running as _is_process_running_data,
    tail_text_file as _tail_text_file_data,
    terminate_process as _terminate_process_data,
)
from apps.dashboard.services.execution_dashboard import (
    build_cumulative_realized_pnl_figure as _build_cumulative_realized_pnl_figure_data,
    build_direction_table as _build_direction_table_data,
    build_execution_metric_rows as _build_execution_metric_rows_data,
    build_order_status_figure as _build_order_status_figure_data,
    build_streak_distribution_figure as _build_streak_distribution_figure_data,
    build_trade_pnl_figure as _build_trade_pnl_figure_data,
    filter_closed_trade_analytics as _filter_closed_trade_analytics_data,
)
from apps.dashboard.services.overview_dashboard import (
    build_benchmark_price_figure as _build_benchmark_price_figure_data,
    build_cumulative_return_figure as _build_cumulative_return_figure_data,
    build_drawdown_figure as _build_drawdown_figure_data,
    build_equity_curve_figure as _build_equity_curve_figure_data,
    build_funding_figure as _build_funding_figure_data,
    build_monthly_returns_heatmap as _build_monthly_returns_heatmap_data,
)
from apps.dashboard.services.ghost_cleanup import (
    run_ghost_cleanup_script as _run_ghost_cleanup_script_data,
)
from apps.dashboard.services.mirror_dashboard import (
    apply_mirror_figure_style as _apply_mirror_figure_style_data,
    build_mirror_balance_equity_figure as _build_mirror_balance_equity_figure_data,
    build_mirror_equity_curve_figure as _build_mirror_equity_curve_figure_data,
    build_mirror_snapshot as _build_mirror_snapshot_data,
    render_mirror_cards as _render_mirror_cards_data,
)
from apps.dashboard.services.report_snapshot import (
    build_dashboard_report_runtime_overrides as _build_dashboard_report_runtime_overrides_data,
    build_monthly_returns_table as _build_monthly_returns_table_data,
    build_mt5_summary_rows as _build_mt5_summary_rows_data,
    build_report_markdown as _build_report_markdown_data,
    build_report_payload as _build_report_payload_data,
    format_metric_value as _format_metric_value_data,
    save_report_snapshot as _save_report_snapshot_data,
    serialize_balance_equity_frame as _serialize_balance_equity_frame_data,
)
from apps.dashboard.services.risk_dashboard import (
    build_heartbeat_interval_figure as _build_heartbeat_interval_figure_data,
    build_order_state_figure as _build_order_state_figure_data,
    build_risk_reason_figure as _build_risk_reason_figure_data,
    build_strategy_process_trace_frame as _build_strategy_process_trace_frame_data,
    prepare_heartbeat_interval_frame as _prepare_heartbeat_interval_frame_data,
)
from apps.dashboard.services.market_dashboard import (
    build_market_close_figure as _build_market_close_figure_data,
    build_market_summary_metrics as _build_market_summary_metrics_data,
    build_market_volume_figure as _build_market_volume_figure_data,
    build_moving_average_figure as _build_moving_average_figure_data,
    build_moving_average_summary_metrics as _build_moving_average_summary_metrics_data,
    build_pair_indicator_summary as _build_pair_indicator_summary_data,
    build_pair_price_inputs_figure as _build_pair_price_inputs_figure_data,
    build_pair_spread_figure as _build_pair_spread_figure_data,
    build_pair_zscore_figure as _build_pair_zscore_figure_data,
    build_rsi_figure as _build_rsi_figure_data,
    build_rsi_signal_figure as _build_rsi_signal_figure_data,
    build_rsi_summary_metrics as _build_rsi_summary_metrics_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_project_strategies_module(project_root: Path) -> None:
    strategies_dir = project_root / "src" / "lumina_quant" / "strategies"
    expected_init = str((strategies_dir / "__init__.py").resolve())
    loaded = sys.modules.get("lumina_quant.strategies")
    if loaded is None:
        return

    loaded_file = getattr(loaded, "__file__", None)
    loaded_file = str(Path(loaded_file).resolve()) if loaded_file else ""
    if loaded_file == expected_init:
        return

    sys.modules.pop("lumina_quant.strategies", None)
    sys.modules.pop("lumina_quant.strategies.registry", None)


_ensure_project_strategies_module(PROJECT_ROOT)

strategy_registry = importlib.import_module("lumina_quant.strategies.registry")
render_exact_window_dashboard = importlib.import_module(
    "apps.dashboard.exact_window_suite"
).render_exact_window_dashboard

_DASHBOARD_VIEW_OPTIONS = ("Main Dashboard", "Exact-Window Suite")


def _render_dashboard_page_shell() -> None:
    st.set_page_config(layout="wide", page_title="LuminaQuant Dashboard")
    st.title("LuminaQuant: Full Trading Intelligence")


def _route_dashboard_view() -> str:
    dashboard_view = st.sidebar.radio(
        "Dashboard View",
        list(_DASHBOARD_VIEW_OPTIONS),
        index=0,
    )
    if dashboard_view == "Exact-Window Suite":
        st.caption("Switched from the main dashboard menu into the exact-window research view.")
        render_exact_window_dashboard(standalone=False)
        st.stop()
    return str(dashboard_view)

DEFAULT_DB_PATH = str(
    os.getenv("LQ_POSTGRES_DSN")
    or getattr(BaseConfig, "POSTGRES_DSN", "")
    or ""
)


def _resolve_postgres_dsn(dsn: str | None = None) -> str:
    token = str(
        dsn
        or os.getenv("LQ_POSTGRES_DSN")
        or getattr(BaseConfig, "POSTGRES_DSN", "")
        or ""
    ).strip()
    return token


class _StateCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, query, params=None):
        self._cursor.execute(str(query), tuple(params or ()))
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def close(self):
        self._cursor.close()

    @property
    def description(self):
        return self._cursor.description


class _StateConnection:
    def __init__(self, connection):
        self._conn = connection

    def cursor(self):
        return _StateCursor(self._conn.cursor())

    def execute(self, query, params=None):
        cursor = self.cursor()
        cursor.execute(query, params)
        return cursor

    def executescript(self, script):
        with self._conn.cursor() as cursor:
            for statement in str(script).split(";"):
                payload = statement.strip()
                if not payload:
                    continue
                cursor.execute(payload)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _connect_state_store(dsn: str):
    resolved = _resolve_postgres_dsn(dsn)
    if not resolved:
        raise RuntimeError("Postgres DSN is required.")
    from lumina_quant.postgres_state import _connect_postgres

    return _StateConnection(_connect_postgres(resolved))


def _execute_query(dsn: str, query: str, params=None):
    conn = _connect_state_store(dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params or ()))
            try:
                rows = cursor.fetchall()
            except Exception:
                rows = []
        conn.commit()
        return rows
    finally:
        conn.close()


def _read_sql_query(dsn: str, query: str, params=None):
    conn = _connect_state_store(dsn)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def _count_market_rows(db_path):
    try:
        rows = _execute_query(db_path, "SELECT COUNT(*) FROM market_ohlcv_1m")
        row = rows[0] if rows else None
        if row is None:
            return 0
        return int(row[0])
    except Exception:
        return 0


def _resolve_default_market_db_path():
    configured = str(
        getattr(BaseConfig, "MARKET_DATA_PARQUET_PATH", "")
        or os.getenv("LQ__STORAGE__MARKET_DATA_PARQUET_PATH")
        or os.getenv("LQ_MARKET_PARQUET_PATH")
        or "data/market_parquet"
    ).strip()
    return configured


DEFAULT_MARKET_DB_PATH = _resolve_default_market_db_path()
DEFAULT_REFRESH_INTERVAL_SEC = 5
DEFAULT_WINDOW_POINTS = 2500
DEFAULT_DOWNSAMPLE_TARGET_POINTS = 5000
DEFAULT_RUN_STALE_SEC = 180
DASHBOARD_REPORT_DIR = WORKFLOW_RUNTIME_ROOT / "reports"
METRIC_DEFINITIONS = {
    "Total Return": "(Final Equity / Initial Equity) - 1 over selected period.",
    "Cumulative Return": "Compounded series from periodic returns across selected window.",
    "CAGR": "Annualized growth rate from period start to end.",
    "Ann. Volatility": "Annualized standard deviation of periodic returns.",
    "Sharpe Ratio": "Risk-adjusted return using total volatility.",
    "Sortino Ratio": "Risk-adjusted return using downside volatility only.",
    "Calmar Ratio": "CAGR divided by absolute max drawdown.",
    "Max Drawdown": "Largest peak-to-trough equity decline in selected period.",
    "Total Net Profit": "Closed-trade net result after commission over selected period.",
    "Gross Profit": "Sum of all positive closed-trade realized PnL.",
    "Gross Loss": "Sum of all negative closed-trade realized PnL.",
    "Profit Factor": "Gross Profit divided by absolute Gross Loss.",
    "Expected Payoff": "Average realized PnL per closed trade.",
    "Recovery Factor": "Total net profit divided by maximal equity drawdown amount.",
    "Open P/L": "Unrealized PnL estimated as total (closed + open) minus closed-trade net PnL.",
    "Total (C+O)": "Combined closed and open PnL relative to initial equity.",
    "R/MDD": "Total (closed + open) PnL divided by equity maximal drawdown.",
    "Profit Trades (% of total)": "Count of profitable closed trades and share versus total closed trades.",
    "Loss Trades (% of total)": "Count of losing closed trades and share versus total closed trades.",
    "Avg Profit Trade": "Average realized PnL across profitable closed trades.",
    "Avg Loss Trade": "Average realized PnL across losing closed trades.",
    "Payoff Ratio": "Average profit trade divided by absolute average loss trade.",
    "AHPR": "Arithmetic mean of per-trade holding period growth factors.",
    "GHPR": "Geometric mean of per-trade holding period growth factors.",
    "LR Correlation": "Correlation between equity curve and linear time trend.",
    "LR Std Error": "Standard error of equity against linear trend fit.",
    "Z-Score": "Runs-test Z-score of win/loss sequence randomness.",
    "Long Trades (Win %)": "Closed long-side trades and long-side win rate.",
    "Short Trades (Win %)": "Closed short-side trades and short-side win rate.",
    "Avg Holding Time": "Average holding time of closed quantities.",
    "Max Holding Time": "Maximum holding time of closed quantities.",
    "Min Holding Time": "Minimum holding time of closed quantities.",
    "Equity Drawdown Absolute": "Initial equity minus minimum equity (never below zero).",
    "Equity Drawdown Maximal": "Largest absolute peak-to-trough equity drawdown amount.",
    "Equity Drawdown Relative %": "Largest relative peak-to-trough equity drawdown percent.",
    "Balance Drawdown Absolute": "Initial balance minus minimum realized-balance level.",
    "Balance Drawdown Maximal": "Largest absolute peak-to-trough realized-balance drawdown amount.",
    "Balance Drawdown Relative %": "Largest relative peak-to-trough realized-balance drawdown percent.",
    "Alpha": "Strategy excess annualized return versus benchmark after beta adjustment.",
    "Beta": "Sensitivity of strategy returns to benchmark returns.",
    "Information Ratio": "Mean active return divided by active return volatility.",
    "Funding (Net)": "Accumulated funding payments (positive means paid by longs, received by shorts when negative).",
    "Win Rate": "Proportion of closed trades with positive realized PnL.",
}

def _safe_interval_sec(value):
    try:
        parsed = int(value)
    except Exception:
        return DEFAULT_REFRESH_INTERVAL_SEC
    return max(1, min(300, parsed))


def _setup_auto_refresh(enabled, interval_sec):
    if not enabled:
        return 0, "manual"

    interval_sec = _safe_interval_sec(interval_sec)
    interval_ms = interval_sec * 1000
    try:
        module = importlib.import_module("streamlit_autorefresh")
        counter = int(
            module.st_autorefresh(interval=interval_ms, key=f"lq-autorefresh-{interval_ms}")
        )
        return counter, "streamlit_autorefresh"
    except Exception:
        st.markdown(f"<meta http-equiv='refresh' content='{interval_sec}'>", unsafe_allow_html=True)
        return int(time.time() // interval_sec), "meta_refresh"


def _coerce_datetime(df, column):
    if df.empty or column not in df.columns:
        return df
    df[column] = pd.to_datetime(df[column], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def _resolve_dashboard_market_timeframe(value):
    """Clamp dashboard market chart queries to >=1m to avoid heavy 1s scans."""
    try:
        token = normalize_timeframe_token(value)
    except Exception:
        return "1m", True
    try:
        if int(timeframe_to_milliseconds(token)) < 60_000:
            return "1m", True
    except Exception:
        return "1m", True
    return token, False


def _utc_now_iso():
    return datetime.now(UTC).isoformat()


def _ensure_workflow_jobs_schema(db_path):
    if not db_path:
        return
    try:
        conn = _connect_state_store(db_path)
    except Exception:
        return
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workflow_jobs (
                job_id TEXT PRIMARY KEY,
                workflow TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_mode TEXT,
                strategy TEXT,
                command_json TEXT,
                env_json TEXT,
                pid INTEGER,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                exit_code INTEGER,
                log_path TEXT,
                stop_file TEXT,
                metadata_json TEXT,
                last_updated TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_workflow_jobs_started_at
                ON workflow_jobs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_workflow_jobs_status
                ON workflow_jobs(status);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_workflow_job_row(db_path, row):
    _ensure_workflow_jobs_schema(db_path)
    conn = _connect_state_store(db_path)
    try:
        conn.execute(
            """
            INSERT INTO workflow_jobs(
                job_id, workflow, status, requested_mode, strategy, command_json,
                env_json, pid, run_id, started_at, ended_at, exit_code,
                log_path, stop_file, metadata_json, last_updated
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO UPDATE SET
                workflow = EXCLUDED.workflow,
                status = EXCLUDED.status,
                requested_mode = EXCLUDED.requested_mode,
                strategy = EXCLUDED.strategy,
                command_json = EXCLUDED.command_json,
                env_json = EXCLUDED.env_json,
                pid = EXCLUDED.pid,
                run_id = EXCLUDED.run_id,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                exit_code = EXCLUDED.exit_code,
                log_path = EXCLUDED.log_path,
                stop_file = EXCLUDED.stop_file,
                metadata_json = EXCLUDED.metadata_json,
                last_updated = EXCLUDED.last_updated
            """,
            (
                row.get("job_id"),
                row.get("workflow"),
                row.get("status"),
                row.get("requested_mode"),
                row.get("strategy"),
                row.get("command_json"),
                row.get("env_json"),
                row.get("pid"),
                row.get("run_id"),
                row.get("started_at"),
                row.get("ended_at"),
                row.get("exit_code"),
                row.get("log_path"),
                row.get("stop_file"),
                row.get("metadata_json"),
                row.get("last_updated") or _utc_now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _update_workflow_job_row(db_path, job_id, **updates):
    if not updates:
        return
    _ensure_workflow_jobs_schema(db_path)
    fields = dict(updates)
    fields["last_updated"] = _utc_now_iso()
    assignments = ", ".join(f"{key} = %s" for key in fields)
    values = [*list(fields.values()), job_id]
    conn = _connect_state_store(db_path)
    try:
        conn.execute(
            f"UPDATE workflow_jobs SET {assignments} WHERE job_id = %s",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def _is_process_running(pid):
    return _is_process_running_data(pid)


def _terminate_process(pid):
    return _terminate_process_data(pid)


def _tail_text_file(path, max_chars=20000):
    return _tail_text_file_data(path, max_chars=max_chars)


@st.cache_data
def load_workflow_jobs(db_path, refresh_counter=0, limit=200):
    _ = refresh_counter
    return _load_workflow_jobs_frame_data(
        db_path=db_path,
        limit=limit,
        resolve_postgres_dsn=_resolve_postgres_dsn,
        ensure_workflow_jobs_schema=_ensure_workflow_jobs_schema,
        connect_state_store=_connect_state_store,
        coerce_datetime=_coerce_datetime,
    )


def _refresh_workflow_jobs(db_path):
    _refresh_workflow_jobs_data(
        db_path=db_path,
        session_state=st.session_state,
        resolve_postgres_dsn=_resolve_postgres_dsn,
        ensure_workflow_jobs_schema=_ensure_workflow_jobs_schema,
        connect_state_store=_connect_state_store,
        is_process_running=_is_process_running,
        update_workflow_job_row=_update_workflow_job_row,
        utc_now_iso=_utc_now_iso,
    )


def _annotate_run_health(runs_df, stale_after_sec=DEFAULT_RUN_STALE_SEC):
    if runs_df.empty:
        return runs_df
    out = runs_df.copy()
    out = _coerce_datetime(out, "last_heartbeat_at")
    out = _coerce_datetime(out, "last_equity_at")
    now = pd.Timestamp.utcnow().tz_localize(None)
    effective = []
    telemetry_age = []

    for _, row in out.iterrows():
        status = str(row.get("status", "")).upper()
        latest = row.get("last_heartbeat_at")
        eq = row.get("last_equity_at")
        if pd.notna(eq) and (pd.isna(latest) or eq > latest):
            latest = eq

        age_sec = None
        if pd.notna(latest):
            age_sec = max(0.0, float((now - latest).total_seconds()))
        telemetry_age.append(age_sec)

        if status == "RUNNING":
            if age_sec is None:
                effective.append("RUNNING_NO_TELEMETRY")
            elif age_sec > float(stale_after_sec):
                effective.append("RUNNING_STALE")
            else:
                effective.append("RUNNING_HEALTHY")
        else:
            effective.append(status or "UNKNOWN")

    out["telemetry_age_sec"] = telemetry_age
    out["effective_status"] = effective
    return out


def _build_stop_file_path(job_id):
    return _build_stop_file_path_data(job_id, control_dir=WORKFLOW_CONTROL_DIR)


@dataclass(frozen=True)
class _ManagedJobLaunchSpec:
    workflow: str
    command: tuple[str, ...]
    env_overrides: dict[str, str]
    requested_mode: str | None
    strategy: str | None
    run_id: str | None
    stop_file: str | None = None
    metadata: dict[str, object] | None = None

    def launch(self, *, db_path):
        return _launch_managed_job_data(
            db_path=db_path,
            workflow=self.workflow,
            command=self.command,
            env_overrides=self.env_overrides,
            workflow_log_dir=str(WORKFLOW_LOG_DIR),
            session_state=st.session_state,
            insert_workflow_job_row=_insert_workflow_job_row,
            utc_now_iso=_utc_now_iso,
            requested_mode=self.requested_mode,
            strategy=self.strategy,
            run_id=self.run_id,
            stop_file=self.stop_file,
            metadata=self.metadata,
            cwd=str(PROJECT_ROOT),
        )


@dataclass(frozen=True)
class _LiveRunnerSelection:
    runner_kind: str
    live_mode: str
    strategy_name: str
    real_armed: bool


@dataclass(frozen=True)
class _ManagedRunLaunchContext:
    strategy_name: str
    strategy_params: dict[str, object]
    runner_data_source: str
    market_db_path: str
    market_exchange: str
    runner_env_overrides: dict[str, str]
    optimize_folds: int
    optimize_trials: int
    optimize_workers: int
    persist_best_params: bool
    runner_leverage: int


@dataclass(frozen=True)
class _DashboardSelectionControls:
    data_source: str
    db_path: str
    market_db_path: str
    market_exchange: str
    market_timeframe: str
    strategy_options: tuple[str, ...]
    strategy_name: str
    auto_refresh_enabled: bool
    refresh_interval_sec: int
    max_points: int
    auto_downsample: bool
    downsample_target_points: int
    pin_to_running: bool
    filter_runs_by_strategy: bool
    run_stale_sec: int
    period_preset: str
    custom_start: object | None
    custom_end: object | None


@dataclass(frozen=True)
class _ExecutionLabControls:
    runner_initial_capital: float
    runner_leverage: int
    runner_symbols: tuple[str, ...]
    runner_timeframe: str
    runner_timeout_sec: int
    runner_data_source: str


def _render_dashboard_selection_controls() -> _DashboardSelectionControls:
    st.sidebar.header("Configuration")
    data_source = st.sidebar.selectbox("Data Source", ["Auto", "Postgres", "CSV"])
    db_path = st.sidebar.text_input("Postgres DSN", value=DEFAULT_DB_PATH)
    market_db_path = st.sidebar.text_input("Market Data Parquet Path", value=DEFAULT_MARKET_DB_PATH)
    market_exchange = st.sidebar.text_input(
        "Market Exchange", value=getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance")
    )
    market_timeframe_requested = st.sidebar.text_input(
        "Market Timeframe", value=getattr(BaseConfig, "TIMEFRAME", "1m")
    )
    market_timeframe, market_timeframe_clamped = _resolve_dashboard_market_timeframe(
        market_timeframe_requested
    )
    if market_timeframe_clamped:
        st.sidebar.caption("Market chart timeframe clamped to 1m minimum for dashboard performance.")

    strategy_options = tuple(strategy_registry.get_strategy_names())
    default_strategy_name = "RsiStrategy"
    default_strategy_index = (
        strategy_options.index(default_strategy_name) if default_strategy_name in strategy_options else 0
    )
    strategy_name = st.sidebar.selectbox(
        "Select Strategy",
        list(strategy_options),
        index=default_strategy_index,
    )
    auto_refresh_enabled = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval_sec = st.sidebar.slider(
        "Refresh Interval (sec)", min_value=1, max_value=120, value=5
    )
    max_points = st.sidebar.slider("Max Data Points", min_value=500, max_value=100000, value=2500)
    auto_downsample = st.sidebar.toggle("Auto Downsample Plots", value=True)
    downsample_target_points = st.sidebar.slider(
        "Downsample Target Points",
        min_value=1000,
        max_value=50000,
        value=DEFAULT_DOWNSAMPLE_TARGET_POINTS,
        step=500,
        disabled=not auto_downsample,
    )
    pin_to_running = st.sidebar.toggle("Pin to RUNNING live run", value=True)
    filter_runs_by_strategy = st.sidebar.toggle("Filter Run IDs By Strategy", value=True)
    run_stale_sec = st.sidebar.slider(
        "RUNNING Stale Threshold (sec)",
        min_value=30,
        max_value=3600,
        value=DEFAULT_RUN_STALE_SEC,
        step=30,
    )
    period_preset = st.sidebar.selectbox(
        "Chart Period",
        ["All", "1D", "7D", "30D", "90D", "180D", "365D", "Custom"],
        index=2,
    )
    custom_start = None
    custom_end = None
    if period_preset == "Custom":
        default_custom_end = pd.Timestamp.utcnow().date()
        default_custom_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).date()
        custom_start = st.sidebar.date_input("Custom Start", value=default_custom_start)
        custom_end = st.sidebar.date_input("Custom End", value=default_custom_end)

    return _DashboardSelectionControls(
        data_source=str(data_source),
        db_path=str(db_path),
        market_db_path=str(market_db_path),
        market_exchange=str(market_exchange),
        market_timeframe=str(market_timeframe),
        strategy_options=strategy_options,
        strategy_name=str(strategy_name),
        auto_refresh_enabled=bool(auto_refresh_enabled),
        refresh_interval_sec=int(refresh_interval_sec),
        max_points=int(max_points),
        auto_downsample=bool(auto_downsample),
        downsample_target_points=int(downsample_target_points),
        pin_to_running=bool(pin_to_running),
        filter_runs_by_strategy=bool(filter_runs_by_strategy),
        run_stale_sec=int(run_stale_sec),
        period_preset=str(period_preset),
        custom_start=custom_start,
        custom_end=custom_end,
    )


def _render_execution_lab_controls() -> _ExecutionLabControls:
    st.sidebar.divider()
    st.sidebar.subheader("Execution Lab")
    runner_initial_capital = st.sidebar.number_input(
        "Initial Equity Override",
        min_value=100.0,
        max_value=100000000.0,
        value=float(BaseConfig.INITIAL_CAPITAL),
        step=100.0,
    )
    runner_leverage = st.sidebar.number_input(
        "Backtest Leverage Override",
        min_value=1,
        max_value=20,
        value=int(BacktestConfig.LEVERAGE),
        step=1,
    )
    runner_symbols_raw = st.sidebar.text_input(
        "Symbols (comma-separated)", value=", ".join(list(BaseConfig.SYMBOLS))
    )
    runner_symbols = _parse_symbols_csv(runner_symbols_raw)
    if not runner_symbols:
        runner_symbols = list(BaseConfig.SYMBOLS)

    runner_timeframe = st.sidebar.text_input("Timeframe Override", value=str(BaseConfig.TIMEFRAME))
    runner_timeout_sec = st.sidebar.slider(
        "Runner Timeout (sec)", min_value=30, max_value=3600, value=900, step=30
    )
    runner_data_source = st.sidebar.selectbox("Runner Data Source", ["auto", "csv", "db"], index=0)

    return _ExecutionLabControls(
        runner_initial_capital=float(runner_initial_capital),
        runner_leverage=int(runner_leverage),
        runner_symbols=tuple(runner_symbols),
        runner_timeframe=str(runner_timeframe),
        runner_timeout_sec=int(runner_timeout_sec),
        runner_data_source=str(runner_data_source),
    )


def _build_backtest_job_launch_spec(
    *,
    runner_data_source,
    market_db_path,
    market_exchange,
    runner_env_overrides,
    strategy_name,
    backtest_run_id,
    strategy_params_path,
) -> _ManagedJobLaunchSpec:
    return _ManagedJobLaunchSpec(
        **_build_backtest_job_launch_spec_data(
            runner_data_source=runner_data_source,
            market_db_path=market_db_path,
            market_exchange=market_exchange,
            runner_env_overrides=runner_env_overrides,
            strategy_name=strategy_name,
            backtest_run_id=backtest_run_id,
            strategy_params_path=strategy_params_path,
        )
    )


def _build_optimize_job_launch_spec(
    *,
    optimize_folds,
    optimize_trials,
    optimize_workers,
    runner_data_source,
    market_db_path,
    market_exchange,
    persist_best_params,
    runner_env_overrides,
    strategy_name,
    optimize_run_id,
) -> _ManagedJobLaunchSpec:
    return _ManagedJobLaunchSpec(
        **_build_optimize_job_launch_spec_data(
            optimize_folds=optimize_folds,
            optimize_trials=optimize_trials,
            optimize_workers=optimize_workers,
            runner_data_source=runner_data_source,
            market_db_path=market_db_path,
            market_exchange=market_exchange,
            persist_best_params=persist_best_params,
            runner_env_overrides=runner_env_overrides,
            strategy_name=strategy_name,
            optimize_run_id=optimize_run_id,
        )
    )


def _build_live_job_launch_spec(
    *,
    runner_env_overrides,
    live_mode,
    market_exchange,
    runner_leverage,
    live_runner_kind,
    live_strategy_name,
    live_run_id,
    stop_file,
) -> _ManagedJobLaunchSpec:
    return _ManagedJobLaunchSpec(
        **_build_live_job_launch_spec_data(
            runner_env_overrides=runner_env_overrides,
            live_mode=live_mode,
            market_exchange=market_exchange,
            runner_leverage=runner_leverage,
            live_runner_kind=live_runner_kind,
            live_strategy_name=live_strategy_name,
            live_run_id=live_run_id,
            stop_file=stop_file,
        )
    )


def _select_active_live_jobs(workflow_jobs):
    if workflow_jobs.empty:
        return pd.DataFrame()
    return workflow_jobs[
        (workflow_jobs["workflow"].isin(["live", "live_ws"]))
        & (workflow_jobs["status"].isin(["RUNNING", "STOP_REQUESTED"]))
    ]


def _is_live_real_mode_armed(*, arm_ack_1, arm_ack_2, arm_phrase) -> bool:
    return bool(arm_ack_1 and arm_ack_2 and str(arm_phrase).strip().upper() == "ENABLE REAL")


def _render_live_runner_settings(*, strategy_options, strategy_name) -> _LiveRunnerSelection:
    live_col_1, live_col_2, live_col_3 = st.columns(3)
    live_runner_kind = live_col_1.selectbox(
        "Live Runner",
        [
            "Polling (uv run lq live --transport poll)",
            "WebSocket (uv run lq live --transport ws)",
        ],
        key="live_runner_kind",
    )
    live_mode = live_col_2.selectbox("Live Mode", ["paper", "real"], key="live_mode")
    live_strategy_index = strategy_options.index(strategy_name) if strategy_name in strategy_options else 0
    live_strategy_name = live_col_3.selectbox(
        "Live Strategy",
        strategy_options,
        index=live_strategy_index,
        key="live_strategy_name",
    )

    live_real_armed = True
    if live_mode == "real":
        st.warning(
            "Real mode sends live exchange orders. Use only after paper/soak validation is complete."
        )
        arm_col_1, arm_col_2 = st.columns(2)
        arm_ack_1 = arm_col_1.checkbox(
            "I understand this can place real orders.",
            key="arm_ack_1",
        )
        arm_ack_2 = arm_col_2.checkbox(
            "I confirmed API keys/account/margin settings.",
            key="arm_ack_2",
        )
        arm_phrase = st.text_input("Type ENABLE REAL to arm", key="arm_phrase")
        live_real_armed = _is_live_real_mode_armed(
            arm_ack_1=arm_ack_1,
            arm_ack_2=arm_ack_2,
            arm_phrase=arm_phrase,
        )
        if not live_real_armed:
            st.info("Real mode is locked until all arm checks are completed.")

    return _LiveRunnerSelection(
        runner_kind=str(live_runner_kind),
        live_mode=str(live_mode),
        strategy_name=str(live_strategy_name),
        real_armed=bool(live_real_armed),
    )


def _render_managed_run_launch_controls(
    *,
    db_path,
    launch_context: _ManagedRunLaunchContext,
    live_runner_selection: _LiveRunnerSelection,
    active_live_jobs,
) -> None:
    run_col_1, run_col_2, run_col_3 = st.columns(3)
    with run_col_1:
        if st.button("Start Backtest Job", type="primary", use_container_width=True):
            params_path = _save_strategy_params(
                launch_context.strategy_name,
                launch_context.strategy_params,
            )
            backtest_run_id = str(uuid.uuid4())
            job_id = _build_backtest_job_launch_spec(
                runner_data_source=launch_context.runner_data_source,
                market_db_path=launch_context.market_db_path,
                market_exchange=launch_context.market_exchange,
                runner_env_overrides=launch_context.runner_env_overrides,
                strategy_name=launch_context.strategy_name,
                backtest_run_id=backtest_run_id,
                strategy_params_path=params_path,
            ).launch(
                db_path=db_path,
            )
            st.success(f"Backtest job launched: {job_id}")
            st.cache_data.clear()

    with run_col_2:
        if st.button("Start Optimization Job", use_container_width=True):
            optimize_run_id = str(uuid.uuid4())
            job_id = _build_optimize_job_launch_spec(
                optimize_folds=launch_context.optimize_folds,
                optimize_trials=launch_context.optimize_trials,
                optimize_workers=launch_context.optimize_workers,
                runner_data_source=launch_context.runner_data_source,
                market_db_path=launch_context.market_db_path,
                market_exchange=launch_context.market_exchange,
                persist_best_params=launch_context.persist_best_params,
                runner_env_overrides=launch_context.runner_env_overrides,
                strategy_name=launch_context.strategy_name,
                optimize_run_id=optimize_run_id,
            ).launch(
                db_path=db_path,
            )
            st.success(f"Optimization job launched: {job_id}")
            st.cache_data.clear()

    with run_col_3:
        start_live_disabled = (
            live_runner_selection.live_mode == "real" and not live_runner_selection.real_armed
        ) or not active_live_jobs.empty
        if st.button("Start Live Job", use_container_width=True, disabled=start_live_disabled):
            live_run_id = str(uuid.uuid4())
            stop_file = _build_stop_file_path(live_run_id)
            job_id = _build_live_job_launch_spec(
                runner_env_overrides=launch_context.runner_env_overrides,
                live_mode=live_runner_selection.live_mode,
                market_exchange=launch_context.market_exchange,
                runner_leverage=launch_context.runner_leverage,
                live_runner_kind=live_runner_selection.runner_kind,
                live_strategy_name=live_runner_selection.strategy_name,
                live_run_id=live_run_id,
                stop_file=stop_file,
            ).launch(
                db_path=db_path,
            )
            st.success(f"Live job launched: {job_id}")
            st.cache_data.clear()



def _request_job_stop(db_path, stop_file):
    return _request_job_stop_data(stop_file, timestamp=_utc_now_iso())


@st.cache_data
def load_runs(db_path, refresh_counter=0):
    _ = refresh_counter
    if not _resolve_postgres_dsn(db_path):
        return pd.DataFrame()
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                r.run_id,
                r.mode,
                r.started_at,
                r.ended_at,
                r.status,
                r.metadata_json AS metadata,
                COALESCE(
                    (r.metadata_json ->> 'strategy'),
                    (
                        SELECT w.strategy
                        FROM workflow_jobs w
                        WHERE w.run_id = r.run_id
                        ORDER BY w.started_at DESC
                        LIMIT 1
                    )
                ) AS strategy,
                COALESCE(eq.equity_rows, 0) AS equity_rows,
                COALESCE(fl.fill_rows, 0) AS fill_rows,
                COALESCE(od.order_rows, 0) AS order_rows,
                COALESCE(rk.risk_rows, 0) AS risk_rows,
                COALESCE(hb.hb_rows, 0) AS hb_rows,
                eq.last_equity_at,
                hb.last_heartbeat_at
            FROM runs r
            LEFT JOIN (
                SELECT run_id, COUNT(*) AS equity_rows, MAX(timeindex) AS last_equity_at
                FROM equity
                GROUP BY run_id
            ) eq ON eq.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS fill_rows FROM fills GROUP BY run_id) fl ON fl.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS order_rows FROM orders GROUP BY run_id) od ON od.run_id = r.run_id
            LEFT JOIN (SELECT run_id, COUNT(*) AS risk_rows FROM risk_events GROUP BY run_id) rk ON rk.run_id = r.run_id
            LEFT JOIN (
                SELECT run_id, COUNT(*) AS hb_rows, MAX(heartbeat_time) AS last_heartbeat_at
                FROM heartbeats
                GROUP BY run_id
            ) hb ON hb.run_id = r.run_id
            ORDER BY r.started_at DESC
            LIMIT 300
            """,
            conn,
        )
        df = _coerce_datetime(df, "started_at")
        df = _coerce_datetime(df, "ended_at")
        df = _coerce_datetime(df, "last_equity_at")
        df = _coerce_datetime(df, "last_heartbeat_at")
        return df
    finally:
        conn.close()


@st.cache_data
def load_equity_state(db_path, run_id, refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT timeindex AS datetime, total, cash, metadata_json AS metadata
            FROM (
                SELECT id, timeindex, total, cash, metadata_json
                FROM equity
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "datetime")
    finally:
        conn.close()


@st.cache_data
def load_metrics_state(db_path, run_id, refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                timeindex AS datetime,
                total,
                cash,
                metadata_json AS metadata
            FROM (
                SELECT id, timeindex, total, cash, metadata_json
                FROM equity
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        df = _coerce_datetime(df, "datetime")
        if df.empty:
            return df

        benchmark = []
        funding = []
        symbol = []
        for meta in df["metadata"].tolist() if "metadata" in df.columns else []:
            info = _parse_json_dict(meta)
            benchmark.append(info.get("benchmark_price"))
            funding.append(info.get("funding_total"))
            symbol.append(info.get("symbol"))

        if benchmark:
            df["benchmark_price"] = pd.to_numeric(pd.Series(benchmark), errors="coerce")
        if funding:
            df["funding"] = pd.to_numeric(pd.Series(funding), errors="coerce").fillna(0.0)
        if symbol:
            df["event_symbol"] = pd.Series(symbol)
        return df
    finally:
        conn.close()


@st.cache_data
def load_fills_state(db_path, run_id, refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                fill_time AS datetime,
                symbol,
                side AS direction,
                quantity,
                fill_cost,
                commission,
                fill_price AS price,
                status,
                metadata_json AS metadata,
                exchange_order_id,
                client_order_id
            FROM (
                SELECT id, fill_time, symbol, side, quantity, fill_cost, commission,
                       fill_price, status, metadata_json, exchange_order_id, client_order_id
                FROM fills
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        if not df.empty and "direction" in df.columns:
            df["direction"] = (
                df["direction"]
                .fillna("")
                .astype(str)
                .str.upper()
                .map({"BUY": "BUY", "SELL": "SELL", "BUY_LONG": "BUY", "SELL_SHORT": "SELL"})
                .fillna(df["direction"])
            )
        return _coerce_datetime(df, "datetime")
    finally:
        conn.close()


@st.cache_data
def load_orders_state(db_path, run_id, refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT created_at, symbol, side, order_type, quantity, price, status,
                   client_order_id, exchange_order_id, metadata_json AS metadata
            FROM (
                SELECT id, created_at, symbol, side, order_type, quantity, price, status,
                       client_order_id, exchange_order_id, metadata_json
                FROM orders
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "created_at")
    finally:
        conn.close()


@st.cache_data
def load_risk_events_state(db_path, run_id, refresh_counter=0, max_points=5000):
    _ = refresh_counter
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT event_time, reason, details_json AS details
            FROM (
                SELECT id, event_time, reason, details_json
                FROM risk_events
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "event_time")
    finally:
        conn.close()


@st.cache_data
def load_heartbeats_state(db_path, run_id, refresh_counter=0, max_points=5000):
    _ = refresh_counter
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT heartbeat_time, status, details_json AS details
            FROM (
                SELECT id, heartbeat_time, status, details_json
                FROM heartbeats
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "heartbeat_time")
    finally:
        conn.close()


@st.cache_data
def load_order_states_state(db_path, run_id, refresh_counter=0, max_points=10000):
    _ = refresh_counter
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT event_time, symbol, client_order_id, exchange_order_id, state, message, details_json AS details
            FROM (
                SELECT id, event_time, symbol, client_order_id, exchange_order_id, state, message, details_json
                FROM order_state_events
                WHERE run_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[run_id, int(max(1, max_points))],
        )
        return _coerce_datetime(df, "event_time")
    finally:
        conn.close()


@st.cache_data
def load_optimization_results_state(db_path, refresh_counter=0, max_points=10000):
    _ = refresh_counter
    if not _resolve_postgres_dsn(db_path):
        return pd.DataFrame()
    conn = _connect_state_store(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT id, run_id, stage, created_at, params_json, sharpe, cagr, mdd,
                   train_sharpe, robustness_score, extra_json
            FROM (
                SELECT id, run_id, stage, created_at, params_json, sharpe, cagr, mdd,
                       train_sharpe, robustness_score, extra_json
                FROM optimization_results
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            conn,
            params=[int(max(1, max_points))],
        )
        if df.empty:
            return df
        df = _coerce_datetime(df, "created_at")
        df["params"] = df["params_json"].apply(_parse_json_dict)
        df["extra"] = df["extra_json"].apply(_parse_json_dict)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data
def load_market_ohlcv_state(
    db_path,
    symbol,
    timeframe,
    exchange_id,
    refresh_counter=0,
    max_points=DEFAULT_WINDOW_POINTS,
):
    _ = refresh_counter
    root_path = str(db_path or "").strip()
    if not root_path:
        return pd.DataFrame()
    symbol_token = normalize_symbol(symbol)
    timeframe_token, _ = _resolve_dashboard_market_timeframe(timeframe)
    try:
        from lumina_quant.parquet_market_data import ParquetMarketDataRepository

        repo = ParquetMarketDataRepository(root_path)
        interval_ms = max(1, int(timeframe_to_milliseconds(timeframe_token)))
        end_dt = datetime.now(UTC).replace(tzinfo=None)
        start_dt = end_dt - timedelta(milliseconds=interval_ms * int(max(2, max_points) * 2))
        frame = repo.load_ohlcv(
            exchange=str(exchange_id).strip().lower(),
            symbol=symbol_token,
            timeframe=timeframe_token,
            start_date=start_dt,
            end_date=end_dt,
        )
        if frame.is_empty():
            return pd.DataFrame()
        if frame.height > max_points:
            frame = frame.tail(int(max_points))
        return _coerce_datetime(frame.to_pandas(), "datetime")
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_equity_csv(refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    if not os.path.exists("equity.csv"):
        return pd.DataFrame()
    df = pd.read_csv("equity.csv")
    df = _coerce_datetime(df, "datetime")
    if max_points > 0 and len(df) > max_points:
        df = df.tail(max_points).reset_index(drop=True)
    return df


@st.cache_data
def load_trades_csv(refresh_counter=0, max_points=DEFAULT_WINDOW_POINTS):
    _ = refresh_counter
    if not os.path.exists("trades.csv"):
        return pd.DataFrame()
    df = pd.read_csv("trades.csv")
    df = _coerce_datetime(df, "datetime")
    if max_points > 0 and len(df) > max_points:
        df = df.tail(max_points).reset_index(drop=True)
    return df


@st.cache_data
def load_params(strategy_name):
    path = os.path.join("best_optimized_parameters", strategy_name, "best_params.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _strategy_default_params(strategy_name):
    return strategy_registry.get_default_strategy_params(strategy_name)


def _strategy_default_optuna(strategy_name):
    return strategy_registry.get_default_optuna_config(strategy_name)


def _strategy_default_grid(strategy_name):
    return strategy_registry.get_default_grid_config(strategy_name)


def _merged_strategy_params(strategy_name, loaded_params):
    if isinstance(loaded_params, dict):
        return strategy_registry.resolve_strategy_params(strategy_name, loaded_params)
    return _strategy_default_params(strategy_name)


def _save_strategy_params(strategy_name, params):
    resolved = strategy_registry.resolve_strategy_params(strategy_name, params)
    out_dir = os.path.join("best_optimized_parameters", strategy_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "best_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)
    return out_path


def _parse_symbols_csv(raw_symbols):
    return [token.strip() for token in str(raw_symbols).split(",") if token.strip()]




def _run_ghost_cleanup_script(
    *,
    dsn,
    stale_sec,
    startup_grace_sec,
    close_status,
    force_kill_stop_requested_after_sec,
    apply_changes,
):
    return _run_ghost_cleanup_script_data(
        python_executable=sys.executable,
        dsn=dsn,
        stale_sec=stale_sec,
        startup_grace_sec=startup_grace_sec,
        close_status=close_status,
        force_kill_stop_requested_after_sec=force_kill_stop_requested_after_sec,
        apply_changes=apply_changes,
    )


def _parse_json_dict(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _compute_rsi_series(close_series, period):
    try:
        period = max(2, int(period))
    except Exception:
        period = 14
    close = pd.to_numeric(close_series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~((avg_loss <= 0.0) & (avg_gain > 0.0)), 100.0)
    rsi = rsi.where(~((avg_loss <= 0.0) & (avg_gain <= 0.0)), 0.0)
    return rsi


def _build_strategy_indicator_frame(market_df, strategy_name, strategy_params):
    if market_df.empty or "datetime" not in market_df.columns or "close" not in market_df.columns:
        return pd.DataFrame()

    frame = market_df[["datetime", "close"]].copy()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)
    if frame.empty:
        return frame

    if strategy_name == "RsiStrategy":
        period = strategy_params.get("rsi_period", 14)
        frame["rsi"] = _compute_rsi_series(frame["close"], period)
    elif strategy_name == "MovingAverageCrossStrategy":
        try:
            short_window = max(2, int(strategy_params.get("short_window", 10)))
        except Exception:
            short_window = 10
        try:
            long_window = int(strategy_params.get("long_window", 30))
        except Exception:
            long_window = 30
        long_window = max(short_window + 1, long_window)
        frame["short_ma"] = (
            frame["close"]
            .rolling(
                window=short_window,
                min_periods=short_window,
            )
            .mean()
        )
        frame["long_ma"] = (
            frame["close"]
            .rolling(
                window=long_window,
                min_periods=long_window,
            )
            .mean()
        )

    return frame


def _build_pair_indicator_frame(market_x_df, market_y_df, strategy_params):
    if market_x_df.empty or market_y_df.empty:
        return pd.DataFrame()
    if "datetime" not in market_x_df.columns or "datetime" not in market_y_df.columns:
        return pd.DataFrame()
    if "close" not in market_x_df.columns or "close" not in market_y_df.columns:
        return pd.DataFrame()

    frame_x = market_x_df[["datetime", "close"]].copy()
    frame_x = frame_x.rename(columns={"close": "close_x"})
    frame_x["close_x"] = pd.to_numeric(frame_x["close_x"], errors="coerce")

    frame_y = market_y_df[["datetime", "close"]].copy()
    frame_y = frame_y.rename(columns={"close": "close_y"})
    frame_y["close_y"] = pd.to_numeric(frame_y["close_y"], errors="coerce")

    merged = pd.merge(frame_x, frame_y, on="datetime", how="inner")
    merged = merged.dropna(subset=["close_x", "close_y"]).sort_values("datetime")
    merged = merged.reset_index(drop=True)
    if merged.empty:
        return merged

    lookback = max(10, int(strategy_params.get("lookback_window", 96)))
    hedge_window = max(lookback, int(strategy_params.get("hedge_window", 192)))
    use_log_price = bool(strategy_params.get("use_log_price", True))

    if use_log_price:
        merged["x_price"] = np.log(merged["close_x"])
        merged["y_price"] = np.log(merged["close_y"])
    else:
        merged["x_price"] = merged["close_x"]
        merged["y_price"] = merged["close_y"]

    rolling_cov = (
        merged["x_price"]
        .rolling(window=hedge_window, min_periods=hedge_window)
        .cov(merged["y_price"])
    )
    rolling_var = merged["y_price"].rolling(window=hedge_window, min_periods=hedge_window).var()
    merged["hedge_ratio"] = rolling_cov / rolling_var.replace(0.0, np.nan)
    merged["spread"] = merged["x_price"] - (merged["hedge_ratio"] * merged["y_price"])

    merged["spread_mean"] = merged["spread"].rolling(window=lookback, min_periods=lookback).mean()
    merged["spread_std"] = merged["spread"].rolling(window=lookback, min_periods=lookback).std()
    merged["zscore"] = (merged["spread"] - merged["spread_mean"]) / merged["spread_std"].replace(
        0.0, np.nan
    )
    merged["correlation"] = (
        merged["x_price"].rolling(window=lookback, min_periods=lookback).corr(merged["y_price"])
    )
    return merged


def _latest_running_run_id(runs_df):
    if runs_df.empty or "status" not in runs_df.columns:
        return None
    running = runs_df[runs_df["status"].fillna("").astype(str).str.upper() == "RUNNING"]
    if running.empty:
        return None
    return str(running.iloc[0]["run_id"])


def _latest_run_with_equity(runs_df):
    if runs_df.empty or "equity_rows" not in runs_df.columns:
        return None
    with_data = runs_df[pd.to_numeric(runs_df["equity_rows"], errors="coerce").fillna(0) > 0]
    if with_data.empty:
        return None
    return str(with_data.iloc[0]["run_id"])


def _equity_rows_for_run(runs_df, run_id):
    if runs_df.empty or not run_id:
        return 0
    matched = runs_df.loc[runs_df["run_id"] == run_id, "equity_rows"]
    if matched.empty:
        return 0
    try:
        return int(float(matched.iloc[0]))
    except Exception:
        return 0


def _extract_run_symbols(runs_df, run_id):
    if runs_df.empty or not run_id or "metadata" not in runs_df.columns:
        return list(BaseConfig.SYMBOLS)
    row = runs_df.loc[runs_df["run_id"] == run_id]
    if row.empty:
        return list(BaseConfig.SYMBOLS)
    metadata = _parse_json_dict(row.iloc[0].get("metadata"))
    symbols = metadata.get("symbols")
    if isinstance(symbols, list) and symbols:
        return [str(s) for s in symbols]
    return list(BaseConfig.SYMBOLS)


def _runs_for_strategy(runs_df, strategy_name):
    if runs_df.empty or "run_id" not in runs_df.columns:
        return []
    if "strategy" not in runs_df.columns:
        return []
    selected = str(strategy_name or "").strip()
    if not selected:
        return []
    strat_col = runs_df["strategy"].fillna("").astype(str).str.strip()
    matched = runs_df.loc[strat_col == selected]
    if matched.empty:
        return []
    return matched["run_id"].astype(str).tolist()


def _compute_data_latency_seconds(df):
    if df.empty or "datetime" not in df.columns:
        return None, None
    latest_ts = pd.to_datetime(df["datetime"], errors="coerce").dropna()
    if latest_ts.empty:
        return None, None
    latest = latest_ts.max()
    now = pd.Timestamp.utcnow().tz_localize(None)
    return latest, max(0.0, float((now - latest).total_seconds()))


def _resolve_period_window(
    period_preset, custom_start, custom_end, reference_df, dt_col="datetime"
):
    if period_preset == "All":
        return None, None

    ref_end = None
    if not reference_df.empty and dt_col in reference_df.columns:
        ts = pd.to_datetime(reference_df[dt_col], errors="coerce").dropna()
        if not ts.empty:
            ref_end = ts.max()
    if ref_end is None:
        ref_end = pd.Timestamp.utcnow().tz_localize(None)

    if period_preset == "Custom":
        start_ts = pd.Timestamp(custom_start) if custom_start else None
        end_ts = pd.Timestamp(custom_end) if custom_end else None
        if end_ts is not None:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return start_ts, end_ts

    days_map = {
        "1D": 1,
        "7D": 7,
        "30D": 30,
        "90D": 90,
        "180D": 180,
        "365D": 365,
    }
    days = days_map.get(period_preset)
    if days is None:
        return None, None
    return ref_end - pd.Timedelta(days=days), ref_end


def _apply_period_filter(df, dt_col, start_ts, end_ts):
    if df.empty or dt_col not in df.columns:
        return df
    out = df.copy()
    ts = pd.to_datetime(out[dt_col], errors="coerce")
    mask = ts.notna()
    if start_ts is not None:
        mask = mask & (ts >= start_ts)
    if end_ts is not None:
        mask = mask & (ts <= end_ts)
    return out.loc[mask].reset_index(drop=True)


def _safe_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_div(numerator, denominator, default=0.0):
    den = _safe_float(denominator, 0.0)
    if den == 0.0:
        return float(default)
    return float(_safe_float(numerator, 0.0) / den)


def _format_duration_seconds(value):
    sec = _safe_float(value, 0.0)
    if sec <= 0.0:
        return "0s"
    total = round(sec)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _metric_value(metric_name, performance, summary):
    if isinstance(performance, dict) and metric_name in performance:
        return performance.get(metric_name)

    if not isinstance(summary, dict):
        return None

    mapping = {
        "Total Net Profit": summary.get("total_net_profit"),
        "Open P/L": summary.get("open_pnl"),
        "Total (C+O)": summary.get("total_c_plus_o"),
        "Gross Profit": summary.get("gross_profit"),
        "Gross Loss": summary.get("gross_loss"),
        "Profit Factor": summary.get("profit_factor"),
        "Expected Payoff": summary.get("expected_payoff"),
        "Recovery Factor": summary.get("recovery_factor"),
        "R/MDD": summary.get("r_mdd"),
        "Profit Trades (% of total)": summary.get("profit_trades_text"),
        "Loss Trades (% of total)": summary.get("loss_trades_text"),
        "Avg Profit Trade": summary.get("avg_profit_trade"),
        "Avg Loss Trade": summary.get("avg_loss_trade"),
        "Payoff Ratio": summary.get("payoff_ratio"),
        "AHPR": summary.get("ahpr"),
        "GHPR": summary.get("ghpr"),
        "LR Correlation": summary.get("lr_correlation"),
        "LR Std Error": summary.get("lr_std_error"),
        "Z-Score": summary.get("z_score"),
        "Long Trades (Win %)": summary.get("long_trades_win_pct"),
        "Short Trades (Win %)": summary.get("short_trades_win_pct"),
        "Avg Holding Time": summary.get("holding_time_avg_sec"),
        "Max Holding Time": summary.get("holding_time_max_sec"),
        "Min Holding Time": summary.get("holding_time_min_sec"),
        "Equity Drawdown Absolute": summary.get("equity_drawdown_absolute"),
        "Equity Drawdown Maximal": summary.get("equity_drawdown_maximal"),
        "Equity Drawdown Relative %": summary.get("equity_drawdown_relative_pct"),
        "Balance Drawdown Absolute": summary.get("balance_drawdown_absolute"),
        "Balance Drawdown Maximal": summary.get("balance_drawdown_maximal"),
        "Balance Drawdown Relative %": summary.get("balance_drawdown_relative_pct"),
    }
    return mapping.get(metric_name)


def _drawdown_stats(values, initial_value):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "absolute": 0.0,
            "maximal": 0.0,
            "relative_pct": 0.0,
        }

    peak = np.maximum.accumulate(arr)
    drawdown_abs = peak - arr
    drawdown_rel = np.divide(
        drawdown_abs,
        np.where(peak == 0.0, np.nan, peak),
        dtype=np.float64,
    )
    drawdown_rel = np.nan_to_num(drawdown_rel, nan=0.0, posinf=0.0, neginf=0.0)

    absolute = max(0.0, _safe_float(initial_value, 0.0) - float(np.min(arr)))
    maximal = float(np.max(drawdown_abs))
    relative = float(np.max(drawdown_rel))
    return {
        "absolute": float(absolute),
        "maximal": float(maximal),
        "relative_pct": float(relative),
    }


def _streak_groups(sequence):
    groups = []
    if not sequence:
        return groups
    current = sequence[0]
    count = 1
    for value in sequence[1:]:
        if value == current:
            count += 1
            continue
        groups.append((current, count))
        current = value
        count = 1
    groups.append((current, count))
    return groups


def _runs_test_zscore(binary_outcomes):
    if not binary_outcomes:
        return 0.0
    outcomes = [bool(value) for value in binary_outcomes]
    wins = int(sum(outcomes))
    losses = int(len(outcomes) - wins)
    if wins == 0 or losses == 0:
        return 0.0

    runs = 1
    for idx in range(1, len(outcomes)):
        if outcomes[idx] != outcomes[idx - 1]:
            runs += 1

    total = wins + losses
    expected_runs = 1.0 + (2.0 * wins * losses / float(total))
    variance = (
        2.0
        * wins
        * losses
        * ((2.0 * wins * losses) - wins - losses)
        / float((total**2) * (total - 1))
    )
    if variance <= 0.0:
        return 0.0

    diff = float(runs) - expected_runs
    correction = 0.0
    if diff > 0.0:
        correction = 0.5
    elif diff < 0.0:
        correction = -0.5
    return float((diff - correction) / math.sqrt(variance))


def _linear_regression_diagnostics(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0, 0.0

    x = np.arange(arr.size, dtype=np.float64)
    x_center = x - float(np.mean(x))
    y_center = arr - float(np.mean(arr))
    denom = float(np.sum(x_center**2))
    if denom <= 0.0:
        return 0.0, 0.0

    slope = float(np.sum(x_center * y_center) / denom)
    intercept = float(np.mean(arr) - slope * np.mean(x))
    y_hat = intercept + (slope * x)
    residuals = arr - y_hat
    sse = float(np.sum(residuals**2))
    stderr = math.sqrt(sse / float(max(1, arr.size - 2)))

    y_std = float(np.std(arr))
    if y_std <= 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x, arr)[0, 1])
        if not math.isfinite(corr):
            corr = 0.0
    return corr, float(stderr)


def _build_monthly_returns_table(df_equity, performance):
    return _build_monthly_returns_table_data(df_equity, performance)


def _build_mt5_summary_rows(summary):
    return _build_mt5_summary_rows_data(
        summary,
        format_metric_value=_format_metric_value,
    )


def _format_metric_value(name, value):
    return _format_metric_value_data(
        name,
        value,
        safe_float=_safe_float,
        format_duration_seconds=_format_duration_seconds,
    )


def _downsample_frame(df, target_points):
    if df.empty:
        return df
    target = max(1, int(target_points))
    total = len(df)
    if total <= target:
        return df
    step = max(1, total // target)
    indices = list(range(0, total, step))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    return df.iloc[indices].reset_index(drop=True)


def compute_trade_analytics(df_trades):
    if df_trades.empty:
        return df_trades
    df = df_trades.copy().sort_values("datetime").reset_index(drop=True)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ["quantity", "price", "fill_cost", "commission"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    if "direction" not in df.columns:
        df["direction"] = ""
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"

    positions = {}
    avg_cost = {}
    entry_times = {}
    realized_pnl = []
    realized_return = []
    position_after = []
    avg_cost_after = []
    closed_qty = []
    close_side = []
    holding_sec = []

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

            entry_basis = abs(avg * closes)
            exit_basis = abs(price * closes)
            basis = max(entry_basis, exit_basis)
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
    df["realized_return_pct"] = pd.Series(realized_return) * 100.0
    df["position_after"] = position_after
    df["avg_cost_after"] = avg_cost_after
    df["closed_qty"] = closed_qty
    df["close_side"] = close_side
    df["holding_sec"] = holding_sec
    df["cum_realized_pnl"] = df["realized_pnl"].cumsum()
    df["notional"] = df["quantity"] * df["price"]
    return df


def build_summary(df_equity, df_trades):
    out = {
        "period_start": None,
        "period_end": None,
        "period_days": 0.0,
        "bars": len(df_equity),
        "fills": len(df_trades),
        "buy_fills": 0,
        "sell_fills": 0,
        "fills_per_day": 0.0,
        "avg_qty": 0.0,
        "avg_notional": 0.0,
        "total_commission": 0.0,
        "realized_pnl": 0.0,
        "win_rate": 0.0,
        "avg_trade_return_pct": 0.0,
        "best_trade_pnl": 0.0,
        "worst_trade_pnl": 0.0,
        "initial_equity": 0.0,
        "final_equity": 0.0,
        "total_net_profit": 0.0,
        "open_pnl": 0.0,
        "total_c_plus_o": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": 0.0,
        "expected_payoff": 0.0,
        "recovery_factor": 0.0,
        "r_mdd": 0.0,
        "profit_trades_count": 0,
        "loss_trades_count": 0,
        "profit_trades_pct": 0.0,
        "loss_trades_pct": 0.0,
        "profit_trades_text": "0 (0.00%)",
        "loss_trades_text": "0 (0.00%)",
        "avg_profit_trade": 0.0,
        "avg_loss_trade": 0.0,
        "payoff_ratio": 0.0,
        "ahpr": 0.0,
        "ghpr": 0.0,
        "lr_correlation": 0.0,
        "lr_std_error": 0.0,
        "z_score": 0.0,
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "long_trades": 0,
        "short_trades": 0,
        "long_win_rate": 0.0,
        "short_win_rate": 0.0,
        "long_trades_win_pct": "0 (0.00%)",
        "short_trades_win_pct": "0 (0.00%)",
        "win_streak_max": 0,
        "loss_streak_max": 0,
        "win_streak_avg": 0.0,
        "loss_streak_avg": 0.0,
        "max_consecutive_profit_count": 0,
        "max_consecutive_loss_count": 0,
        "max_consecutive_profit_amount": 0.0,
        "max_consecutive_loss_amount": 0.0,
        "holding_time_min_sec": 0.0,
        "holding_time_avg_sec": 0.0,
        "holding_time_max_sec": 0.0,
        "equity_drawdown_absolute": 0.0,
        "equity_drawdown_maximal": 0.0,
        "equity_drawdown_relative_pct": 0.0,
        "balance_drawdown_absolute": 0.0,
        "balance_drawdown_maximal": 0.0,
        "balance_drawdown_relative_pct": 0.0,
    }
    totals_arr = np.array([], dtype=np.float64)
    if not df_equity.empty and "datetime" in df_equity.columns:
        dt = pd.to_datetime(df_equity["datetime"], errors="coerce").dropna()
        if not dt.empty:
            start = dt.min()
            end = dt.max()
            out["period_start"] = str(start)
            out["period_end"] = str(end)
            seconds = max(0.0, float((end - start).total_seconds()))
            out["period_days"] = seconds / 86400.0
    if not df_equity.empty and "total" in df_equity.columns:
        totals = pd.to_numeric(df_equity["total"], errors="coerce").dropna()
        if not totals.empty:
            out["initial_equity"] = float(totals.iloc[0])
            out["final_equity"] = float(totals.iloc[-1])
            out["total_c_plus_o"] = float(out["final_equity"] - out["initial_equity"])
            totals_arr = totals.to_numpy(dtype=np.float64)
            dd_stats = _drawdown_stats(totals_arr, out["initial_equity"])
            out["equity_drawdown_absolute"] = float(dd_stats["absolute"])
            out["equity_drawdown_maximal"] = float(dd_stats["maximal"])
            out["equity_drawdown_relative_pct"] = float(dd_stats["relative_pct"])
            corr, stderr = _linear_regression_diagnostics(totals_arr)
            out["lr_correlation"] = float(corr)
            out["lr_std_error"] = float(stderr)

    if not df_trades.empty:
        out["buy_fills"] = (
            int((df_trades["direction"] == "BUY").sum()) if "direction" in df_trades.columns else 0
        )
        out["sell_fills"] = (
            int((df_trades["direction"] == "SELL").sum()) if "direction" in df_trades.columns else 0
        )
        out["avg_qty"] = (
            float(df_trades["quantity"].mean()) if "quantity" in df_trades.columns else 0.0
        )
        out["avg_notional"] = (
            float(df_trades["notional"].mean()) if "notional" in df_trades.columns else 0.0
        )
        out["total_commission"] = (
            float(df_trades["commission"].sum()) if "commission" in df_trades.columns else 0.0
        )
        out["realized_pnl"] = (
            float(df_trades["realized_pnl"].sum()) if "realized_pnl" in df_trades.columns else 0.0
        )
        out["total_net_profit"] = float(out["realized_pnl"])
        closed = (
            df_trades[df_trades["closed_qty"] > 0]
            if "closed_qty" in df_trades.columns
            else pd.DataFrame()
        )
        if not closed.empty:
            pnl_series = pd.to_numeric(closed.get("realized_pnl"), errors="coerce").fillna(0.0)
            out["closed_trades"] = len(closed)
            out["wins"] = int((pnl_series > 0.0).sum())
            out["losses"] = int((pnl_series < 0.0).sum())
            out["profit_trades_count"] = int((pnl_series > 0.0).sum())
            out["loss_trades_count"] = int((pnl_series < 0.0).sum())
            denom = max(1, out["wins"] + out["losses"])
            out["win_rate"] = float(out["wins"] / float(denom))
            out["profit_trades_pct"] = float(
                out["profit_trades_count"] / float(out["closed_trades"])
            )
            out["loss_trades_pct"] = float(out["loss_trades_count"] / float(out["closed_trades"]))
            out["profit_trades_text"] = (
                f"{out['profit_trades_count']} ({out['profit_trades_pct']:.2%})"
            )
            out["loss_trades_text"] = f"{out['loss_trades_count']} ({out['loss_trades_pct']:.2%})"
            if "realized_return_pct" in closed.columns:
                out["avg_trade_return_pct"] = float(
                    pd.to_numeric(closed["realized_return_pct"], errors="coerce").mean()
                )
            out["best_trade_pnl"] = float(pnl_series.max())
            out["worst_trade_pnl"] = float(pnl_series.min())

            out["gross_profit"] = float(pnl_series[pnl_series > 0.0].sum())
            out["gross_loss"] = float(pnl_series[pnl_series < 0.0].sum())
            profit_only = pnl_series[pnl_series > 0.0]
            loss_only = pnl_series[pnl_series < 0.0]
            out["avg_profit_trade"] = float(profit_only.mean()) if not profit_only.empty else 0.0
            out["avg_loss_trade"] = float(loss_only.mean()) if not loss_only.empty else 0.0
            if out["avg_loss_trade"] < 0.0:
                out["payoff_ratio"] = float(out["avg_profit_trade"] / abs(out["avg_loss_trade"]))
            out["expected_payoff"] = _safe_div(out["total_net_profit"], len(closed), 0.0)
            if out["gross_loss"] < 0.0:
                out["profit_factor"] = float(out["gross_profit"] / abs(out["gross_loss"]))
            elif out["gross_profit"] > 0.0:
                out["profit_factor"] = float("inf")

            trade_returns = (
                pd.to_numeric(closed.get("realized_return_pct"), errors="coerce") / 100.0
            )
            trade_returns = trade_returns[np.isfinite(trade_returns)]
            trade_returns = trade_returns[trade_returns > -1.0]
            if not trade_returns.empty:
                growth = 1.0 + trade_returns.to_numpy(dtype=np.float64)
                out["ahpr"] = float(np.mean(growth))
                if np.all(growth > 0.0):
                    out["ghpr"] = float(np.prod(growth) ** (1.0 / float(growth.size)))

            decisive = closed[
                pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(0.0) != 0.0
            ]
            if not decisive.empty:
                outcomes = list((decisive["realized_pnl"] > 0.0).to_numpy(dtype=bool))
                out["z_score"] = float(_runs_test_zscore(outcomes))
                streaks = _streak_groups(outcomes)
                win_lengths = [length for flag, length in streaks if flag]
                loss_lengths = [length for flag, length in streaks if not flag]
                out["win_streak_max"] = int(max(win_lengths) if win_lengths else 0)
                out["loss_streak_max"] = int(max(loss_lengths) if loss_lengths else 0)
                out["win_streak_avg"] = float(np.mean(win_lengths)) if win_lengths else 0.0
                out["loss_streak_avg"] = float(np.mean(loss_lengths)) if loss_lengths else 0.0

                current_sign = 0
                current_sum = 0.0
                current_count = 0
                best_profit_sum = float("-inf")
                best_profit_count = 0
                best_loss_sum = float("inf")
                best_loss_count = 0

                for value in (
                    pd.to_numeric(decisive["realized_pnl"], errors="coerce").fillna(0.0).to_list()
                ):
                    sign = 1 if value > 0.0 else -1
                    if current_sign == sign:
                        current_sum += float(value)
                        current_count += 1
                    else:
                        if current_sign == 1:
                            best_profit_sum = max(best_profit_sum, current_sum)
                            best_profit_count = max(best_profit_count, current_count)
                        elif current_sign == -1:
                            best_loss_sum = min(best_loss_sum, current_sum)
                            best_loss_count = max(best_loss_count, current_count)
                        current_sign = sign
                        current_sum = float(value)
                        current_count = 1

                if current_sign == 1:
                    best_profit_sum = max(best_profit_sum, current_sum)
                    best_profit_count = max(best_profit_count, current_count)
                elif current_sign == -1:
                    best_loss_sum = min(best_loss_sum, current_sum)
                    best_loss_count = max(best_loss_count, current_count)

                out["max_consecutive_profit_amount"] = (
                    float(best_profit_sum) if math.isfinite(best_profit_sum) else 0.0
                )
                out["max_consecutive_loss_amount"] = (
                    float(best_loss_sum) if math.isfinite(best_loss_sum) else 0.0
                )
                out["max_consecutive_profit_count"] = int(best_profit_count)
                out["max_consecutive_loss_count"] = int(best_loss_count)

            if "close_side" in closed.columns:
                long_closed = closed[closed["close_side"] == "LONG"]
                short_closed = closed[closed["close_side"] == "SHORT"]
                out["long_trades"] = len(long_closed)
                out["short_trades"] = len(short_closed)
                if len(long_closed) > 0:
                    long_wins = int((long_closed["realized_pnl"] > 0.0).sum())
                    out["long_win_rate"] = float(long_wins / float(len(long_closed)))
                if len(short_closed) > 0:
                    short_wins = int((short_closed["realized_pnl"] > 0.0).sum())
                    out["short_win_rate"] = float(short_wins / float(len(short_closed)))
                out["long_trades_win_pct"] = f"{out['long_trades']} ({out['long_win_rate']:.2%})"
                out["short_trades_win_pct"] = f"{out['short_trades']} ({out['short_win_rate']:.2%})"

            if "holding_sec" in closed.columns:
                hold = pd.to_numeric(closed["holding_sec"], errors="coerce")
                hold = hold[np.isfinite(hold)]
                hold = hold[hold >= 0.0]
                if not hold.empty:
                    out["holding_time_min_sec"] = float(hold.min())
                    out["holding_time_avg_sec"] = float(hold.mean())
                    out["holding_time_max_sec"] = float(hold.max())

            if out["initial_equity"] > 0.0:
                balance_curve = np.array(
                    [
                        out["initial_equity"],
                        *(out["initial_equity"] + pnl_series.cumsum()).tolist(),
                    ],
                    dtype=np.float64,
                )
                bal_dd = _drawdown_stats(balance_curve, out["initial_equity"])
                out["balance_drawdown_absolute"] = float(bal_dd["absolute"])
                out["balance_drawdown_maximal"] = float(bal_dd["maximal"])
                out["balance_drawdown_relative_pct"] = float(bal_dd["relative_pct"])

            dd_denom = out["balance_drawdown_maximal"]
            if dd_denom <= 0.0:
                dd_denom = out["equity_drawdown_maximal"]
            out["recovery_factor"] = _safe_div(out["total_net_profit"], dd_denom, 0.0)

    if (
        out["balance_drawdown_maximal"] <= 0.0
        and out["initial_equity"] > 0.0
        and totals_arr.size > 0
    ):
        out["balance_drawdown_absolute"] = out["equity_drawdown_absolute"]
        out["balance_drawdown_maximal"] = out["equity_drawdown_maximal"]
        out["balance_drawdown_relative_pct"] = out["equity_drawdown_relative_pct"]

    if out["period_days"] > 0:
        out["fills_per_day"] = out["fills"] / out["period_days"]

    out["open_pnl"] = float(out["total_c_plus_o"] - out["total_net_profit"])
    out["r_mdd"] = _safe_div(out["total_c_plus_o"], out["equity_drawdown_maximal"], 0.0)
    return out


def build_performance_metrics(df_equity):
    if df_equity.empty or "total" not in df_equity.columns:
        return {}

    totals = pd.to_numeric(df_equity["total"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if totals.size < 2:
        return {}

    periods = 252
    prev = totals[:-1]
    nxt = totals[1:]
    returns = np.divide(nxt - prev, np.where(prev == 0.0, 1.0, prev), dtype=float)
    if returns.size > 0 and not math.isfinite(returns[0]):
        returns[0] = 0.0

    benchmark_returns = np.array([], dtype=float)
    benchmark_series = None
    if "benchmark_price" in df_equity.columns:
        benchmark_values = pd.to_numeric(df_equity["benchmark_price"], errors="coerce").fillna(0.0)
        benchmark_series = benchmark_values.to_numpy(dtype=float)
        if benchmark_series.size >= 2:
            b_prev = benchmark_series[:-1]
            b_next = benchmark_series[1:]
            benchmark_returns = np.divide(
                b_next - b_prev,
                np.where(b_prev == 0.0, 1.0, b_prev),
                dtype=float,
            )

    total_return = _safe_float((totals[-1] - totals[0]) / totals[0] if totals[0] else 0.0)
    cumulative_returns = pd.Series(returns).fillna(0.0).add(1.0).cumprod().sub(1.0)
    cumulative_return = _safe_float(
        cumulative_returns.iloc[-1] if not cumulative_returns.empty else 0.0
    )
    cagr = _safe_float(create_cagr(totals[-1], totals[0], len(totals), periods))
    vol = _safe_float(create_annualized_volatility(returns, periods))
    sharpe = _safe_float(create_sharpe_ratio(returns, periods=periods))
    sortino = _safe_float(create_sortino_ratio(returns, periods=periods))
    drawdown, dd_duration = create_drawdowns(totals)
    max_dd = _safe_float(max(drawdown) if drawdown else 0.0)
    calmar = _safe_float(create_calmar_ratio(cagr, max_dd))

    alpha = 0.0
    beta = 0.0
    info_ratio = 0.0
    if benchmark_returns.size > 1 and returns.size > 1:
        min_len = min(len(returns), len(benchmark_returns))
        alpha, beta = create_alpha_beta(
            returns[:min_len], benchmark_returns[:min_len], periods=periods
        )
        alpha = _safe_float(alpha)
        beta = _safe_float(beta)
        info_ratio = _safe_float(
            create_information_ratio(returns[:min_len], benchmark_returns[:min_len])
        )

    funding_net = 0.0
    if "funding" in df_equity.columns:
        funding_net = _safe_float(
            pd.to_numeric(df_equity["funding"], errors="coerce").fillna(0.0).iloc[-1]
        )

    return {
        "Total Return": total_return,
        "Cumulative Return": cumulative_return,
        "CAGR": cagr,
        "Ann. Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_dd,
        "DD Duration": int(dd_duration),
        "Alpha": alpha,
        "Beta": beta,
        "Information Ratio": info_ratio,
        "Funding (Net)": funding_net,
        "benchmark_series": benchmark_series,
        "return_series": returns,
        "cum_return_series": cumulative_returns,
    }


def _format_signed_dollar(value, digits=2):
    parsed = _safe_float(value, 0.0)
    prefix = "+" if parsed > 0 else ""
    return f"{prefix}${parsed:,.{digits}f}"


def _tone_class(value, invert=False):
    parsed = _safe_float(value, 0.0)
    if invert:
        parsed = -parsed
    if abs(parsed) < 1e-12:
        return ""
    return "up" if parsed > 0 else "down"


def _build_balance_equity_frame(df_equity, trade_analytics, initial_equity):
    empty = pd.DataFrame(
        columns=[
            "datetime",
            "equity",
            "balance",
            "open_pnl",
            "drawdown_abs",
            "drawdown_signed",
            "cum_realized_pnl",
            "cum_total_pnl",
        ]
    )
    if df_equity.empty or "datetime" not in df_equity.columns or "total" not in df_equity.columns:
        return empty

    frame = df_equity[["datetime", "total"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["equity"] = pd.to_numeric(frame["total"], errors="coerce")
    frame = (
        frame.dropna(subset=["datetime", "equity"]).sort_values("datetime").reset_index(drop=True)
    )
    if frame.empty:
        return empty

    baseline = _safe_float(initial_equity, 0.0)
    if baseline <= 0.0:
        baseline = _safe_float(frame["equity"].iloc[0], 0.0)

    frame["cum_realized_pnl"] = 0.0
    if (
        not trade_analytics.empty
        and "datetime" in trade_analytics.columns
        and "realized_pnl" in trade_analytics.columns
    ):
        closed = trade_analytics.copy()
        closed["datetime"] = pd.to_datetime(closed["datetime"], errors="coerce")
        closed = closed.dropna(subset=["datetime"])
        if "closed_qty" in closed.columns:
            closed_qty = pd.to_numeric(closed["closed_qty"], errors="coerce").fillna(0.0)
            closed = closed.loc[closed_qty > 0.0]
        if not closed.empty:
            closed["realized_pnl"] = pd.to_numeric(closed["realized_pnl"], errors="coerce").fillna(
                0.0
            )
            closed = (
                closed.groupby("datetime", as_index=False)["realized_pnl"]
                .sum()
                .sort_values("datetime")
            )
            closed["cum_realized_pnl"] = closed["realized_pnl"].cumsum()
            aligned = pd.merge_asof(
                frame[["datetime"]],
                closed[["datetime", "cum_realized_pnl"]],
                on="datetime",
                direction="backward",
            )
            frame["cum_realized_pnl"] = (
                pd.to_numeric(aligned["cum_realized_pnl"], errors="coerce").fillna(0.0).to_numpy()
            )

    frame["balance"] = baseline + frame["cum_realized_pnl"]
    frame["open_pnl"] = frame["equity"] - frame["balance"]
    frame["cum_total_pnl"] = frame["equity"] - baseline
    equity_peak = frame["equity"].cummax()
    frame["drawdown_abs"] = equity_peak - frame["equity"]
    frame["drawdown_signed"] = -frame["drawdown_abs"]
    return frame


def _build_mirror_snapshot(summary, balance_equity_df):
    return _build_mirror_snapshot_data(summary, balance_equity_df, safe_float=_safe_float, safe_div=_safe_div)


def _render_mirror_cards(snapshot):
    return _render_mirror_cards_data(
        snapshot,
        safe_float=_safe_float,
        tone_class=_tone_class,
        format_signed_dollar=_format_signed_dollar,
        st_module=st,
    )


def _apply_mirror_figure_style(fig):
    return _apply_mirror_figure_style_data(fig)


def _build_mirror_equity_curve_figure(balance_equity_df):
    return _build_mirror_equity_curve_figure_data(
        balance_equity_df,
        go_module=go,
        apply_figure_style=_apply_mirror_figure_style,
    )


def _build_mirror_balance_equity_figure(balance_equity_df, snapshot):
    return _build_mirror_balance_equity_figure_data(
        balance_equity_df,
        snapshot,
        go_module=go,
        make_subplots_fn=make_subplots,
        apply_figure_style=_apply_mirror_figure_style,
        safe_float=_safe_float,
    )


def _serialize_balance_equity_frame(df_balance_equity, limit=1500):
    return _serialize_balance_equity_frame_data(
        df_balance_equity,
        limit=limit,
        downsample_frame=_downsample_frame,
        safe_float=_safe_float,
    )


def build_report_payload(
    summary,
    performance,
    run_id,
    source,
    strategy_name,
    period_preset,
    df_equity,
    df_trades,
    df_risk,
    df_hb,
    runtime_overrides,
    strategy_params,
    mirror_snapshot=None,
    balance_equity_series=None,
):
    return _build_report_payload_data(
        summary=summary,
        performance=performance,
        run_id=run_id,
        source=source,
        strategy_name=strategy_name,
        period_preset=period_preset,
        df_equity=df_equity,
        df_trades=df_trades,
        df_risk=df_risk,
        df_hb=df_hb,
        runtime_overrides=runtime_overrides,
        strategy_params=strategy_params,
        build_mt5_summary_rows=_build_mt5_summary_rows,
        build_monthly_returns_table=_build_monthly_returns_table,
        mirror_snapshot=mirror_snapshot,
        balance_equity_series=balance_equity_series,
    )


def _build_snapshot_report_markdown(payload):
    return _build_report_markdown_data(
        payload,
        safe_float=_safe_float,
        format_signed_dollar=_format_signed_dollar,
        format_metric_value=_format_metric_value,
        format_duration_seconds=_format_duration_seconds,
    )


def save_report_snapshot(payload):
    return _save_report_snapshot_data(
        payload,
        out_dir=Path(DASHBOARD_REPORT_DIR),
        markdown_builder=_build_snapshot_report_markdown,
    )


def _build_dashboard_report_runtime_overrides(
    *,
    runner_initial_capital,
    runner_leverage,
    runner_symbols,
    runner_timeframe,
    runner_data_source,
    runner_timeout_sec,
):
    return _build_dashboard_report_runtime_overrides_data(
        runner_initial_capital=runner_initial_capital,
        runner_leverage=runner_leverage,
        runner_symbols=runner_symbols,
        runner_timeframe=runner_timeframe,
        runner_data_source=runner_data_source,
        runner_timeout_sec=runner_timeout_sec,
    )


def _render_snapshot_report_section(
    *,
    summary,
    performance,
    active_run_id,
    resolved_source,
    strategy_name,
    period_preset,
    df_equity,
    trade_analytics,
    df_risk,
    df_hb,
    runner_initial_capital,
    runner_leverage,
    runner_symbols,
    runner_timeframe,
    runner_data_source,
    runner_timeout_sec,
    strategy_params,
    mirror_snapshot,
    mirror_balance_equity,
) -> None:
    payload = build_report_payload(
        summary,
        performance,
        active_run_id,
        resolved_source,
        strategy_name,
        period_preset,
        df_equity,
        trade_analytics,
        df_risk,
        df_hb,
        runtime_overrides=_build_dashboard_report_runtime_overrides(
            runner_initial_capital=runner_initial_capital,
            runner_leverage=runner_leverage,
            runner_symbols=runner_symbols,
            runner_timeframe=runner_timeframe,
            runner_data_source=runner_data_source,
            runner_timeout_sec=runner_timeout_sec,
        ),
        strategy_params=strategy_params,
        mirror_snapshot=mirror_snapshot,
        balance_equity_series=_serialize_balance_equity_frame(mirror_balance_equity, limit=1500),
    )

    if st.button("Generate Snapshot Report", type="primary"):
        json_path, md_path, markdown = save_report_snapshot(payload)
        st.success(f"Report saved: {json_path} | {md_path}")
        st.download_button(
            label="Download Markdown Report",
            data=markdown,
            file_name=os.path.basename(md_path),
            mime="text/markdown",
        )
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=os.path.basename(json_path),
            mime="application/json",
        )

    if payload.get("mt5_summary"):
        st.subheader("Snapshot MT5 Summary")
        st.dataframe(pd.DataFrame(payload.get("mt5_summary", [])), use_container_width=True)

    monthly_payload = payload.get("monthly_returns") or {}
    if monthly_payload:
        st.subheader("Snapshot Monthly Returns")
        monthly_df = pd.DataFrame.from_dict(monthly_payload, orient="index")
        monthly_df.index.name = "Year"
        st.dataframe(monthly_df, use_container_width=True)

    st.subheader("Current Snapshot Preview")
    st.json(payload)


def _render_raw_data_tab(
    *,
    active_run_id,
    resolved_source,
    market_symbol,
    market_timeframe,
    market_exchange,
    runs_df,
    df_equity,
    trade_analytics,
    df_orders,
    df_risk,
    df_hb,
    df_order_states,
    df_market,
    df_optimize,
    db_path,
    refresh_counter,
) -> None:
    st.caption(
        f"Run: {active_run_id if active_run_id else 'N/A'} | Source: {resolved_source} | "
        f"Market: {market_symbol} {market_timeframe} ({market_exchange})"
    )
    raw_frames = [
        ("Runs", runs_df),
        ("Equity", df_equity),
        ("Fills (enriched)", trade_analytics),
        ("Orders", df_orders),
        ("Risk Events", df_risk),
        ("Heartbeats", df_hb),
        ("Order State Events", df_order_states),
        ("Market OHLCV", df_market),
        ("Optimization Results", df_optimize),
        ("Workflow Jobs", load_workflow_jobs(db_path, refresh_counter=refresh_counter)),
    ]
    for label, frame in raw_frames:
        with st.expander(label):
            st.dataframe(frame)


def _render_missing_equity_warning(df_equity, runner_initial_capital) -> None:
    if df_equity.empty:
        st.warning(
            "No equity data found for current selection. "
            "Start a backtest job from Report Export tab, or switch source/period. "
            f"Configured initial equity is {runner_initial_capital:.2f}."
        )


def _render_ghost_cleanup_section(*, db_path, run_stale_sec) -> None:
    st.subheader("Ghost Cleanup")
    st.caption(
        "Close stale RUNNING runs and reconcile orphan workflow_jobs. "
        "Use dry-run first, then apply."
    )
    cleanup_col_1, cleanup_col_2, cleanup_col_3, cleanup_col_4 = st.columns(4)
    cleanup_stale_sec = cleanup_col_1.number_input(
        "Stale Sec",
        min_value=60,
        max_value=86400,
        value=max(300, int(run_stale_sec)),
        step=30,
        key="ghost_cleanup_stale_sec",
    )
    cleanup_startup_grace_sec = cleanup_col_2.number_input(
        "Startup Grace Sec",
        min_value=30,
        max_value=7200,
        value=90,
        step=30,
        key="ghost_cleanup_startup_grace_sec",
    )
    cleanup_close_status = cleanup_col_3.selectbox(
        "Close Status",
        ["STOPPED", "FAILED"],
        index=0,
        key="ghost_cleanup_close_status",
    )
    cleanup_force_kill_age = cleanup_col_4.number_input(
        "Force Kill STOP_REQUESTED Age Sec",
        min_value=0,
        max_value=86400,
        value=0,
        step=30,
        key="ghost_cleanup_force_kill_age",
    )

    cleanup_btn_col_1, cleanup_btn_col_2 = st.columns(2)
    if cleanup_btn_col_1.button("Ghost Cleanup Dry-Run", use_container_width=True):
        cleanup_result = _run_ghost_cleanup_script(
            dsn=db_path,
            stale_sec=cleanup_stale_sec,
            startup_grace_sec=cleanup_startup_grace_sec,
            close_status=cleanup_close_status,
            force_kill_stop_requested_after_sec=cleanup_force_kill_age,
            apply_changes=False,
        )
        cleanup_result["mode"] = "dry_run"
        st.session_state["ghost_cleanup_last_result"] = cleanup_result

    if cleanup_btn_col_2.button("Ghost Cleanup Apply", use_container_width=True):
        cleanup_result = _run_ghost_cleanup_script(
            dsn=db_path,
            stale_sec=cleanup_stale_sec,
            startup_grace_sec=cleanup_startup_grace_sec,
            close_status=cleanup_close_status,
            force_kill_stop_requested_after_sec=cleanup_force_kill_age,
            apply_changes=True,
        )
        cleanup_result["mode"] = "apply"
        st.session_state["ghost_cleanup_last_result"] = cleanup_result
        st.cache_data.clear()

    cleanup_result = st.session_state.get("ghost_cleanup_last_result")
    if cleanup_result:
        if cleanup_result.get("ok"):
            st.success(
                f"Ghost cleanup {cleanup_result.get('mode', 'run')} completed in "
                f"{cleanup_result.get('elapsed_sec', 0.0):.2f}s"
            )
        else:
            st.error(f"Ghost cleanup failed (returncode={cleanup_result.get('returncode')})")
        st.caption(f"Command: {' '.join(cleanup_result.get('command', []))}")
        if cleanup_result.get("payload") is not None:
            st.json(cleanup_result["payload"])
        st.text_area(
            "Ghost Cleanup Output",
            value=cleanup_result.get("output", ""),
            height=240,
            key="ghost_cleanup_output_view",
        )


def _render_active_workflow_job_controls(*, db_path, active_jobs) -> None:
    ctrl_job_id = st.selectbox(
        "Control Active Job",
        active_jobs["job_id"].astype(str).tolist(),
        key="control_active_job_id",
    )
    ctrl_row = active_jobs[active_jobs["job_id"].astype(str) == str(ctrl_job_id)].iloc[0]
    ctrl_col_1, ctrl_col_2 = st.columns(2)
    can_stop = bool(ctrl_row.get("stop_file")) and ctrl_row.get("status") != "STOP_REQUESTED"
    if ctrl_col_1.button(
        "Request Graceful Stop",
        use_container_width=True,
        disabled=not can_stop,
        key=f"request_stop_{ctrl_job_id}",
    ):
        if _request_job_stop(db_path, str(ctrl_row.get("stop_file") or "")):
            _update_workflow_job_row(db_path, ctrl_job_id, status="STOP_REQUESTED")
            st.success(f"Stop requested for {ctrl_job_id}")
            st.cache_data.clear()
        else:
            st.error("This job does not expose a graceful stop file.")

    if ctrl_col_2.button(
        "Force Kill Process",
        use_container_width=True,
        key=f"force_kill_{ctrl_job_id}",
    ):
        ok, detail = _terminate_process(ctrl_row.get("pid"))
        if ok:
            _update_workflow_job_row(
                db_path,
                ctrl_job_id,
                status="KILLED",
                ended_at=_utc_now_iso(),
                exit_code=-9,
            )
            if "workflow_processes" in st.session_state:
                st.session_state["workflow_processes"].pop(ctrl_job_id, None)
            st.success(f"Killed {ctrl_job_id}")
            st.cache_data.clear()
        else:
            st.error(f"Kill failed: {detail}")


def _render_workflow_job_log_viewer(workflow_jobs) -> None:
    log_job_id = st.selectbox(
        "Job Log Viewer",
        workflow_jobs["job_id"].astype(str).tolist(),
        key="workflow_log_viewer_job",
    )
    log_row = workflow_jobs[workflow_jobs["job_id"].astype(str) == str(log_job_id)].iloc[0]
    st.caption(f"Log path: {log_row.get('log_path')}")
    st.text_area(
        "Job Log Tail",
        value=_tail_text_file(str(log_row.get("log_path") or ""), max_chars=25000),
        height=260,
        key="workflow_log_tail_view",
    )


def _render_workflow_jobs_section(*, db_path, refresh_counter) -> None:
    st.subheader("Workflow Jobs")
    workflow_jobs = load_workflow_jobs(db_path, refresh_counter=refresh_counter)
    if workflow_jobs.empty:
        st.info("No workflow jobs recorded yet.")
        return

    jobs_view = workflow_jobs.copy()
    jobs_view["command"] = jobs_view["command_json"].fillna("").astype(str).str.slice(0, 120)
    st.dataframe(
        jobs_view[
            [
                "started_at",
                "workflow",
                "status",
                "requested_mode",
                "strategy",
                "pid",
                "run_id",
                "exit_code",
                "command",
            ]
        ],
        use_container_width=True,
    )

    active_jobs = workflow_jobs[workflow_jobs["status"].isin(["RUNNING", "STOP_REQUESTED"])].copy()
    if not active_jobs.empty:
        _render_active_workflow_job_controls(db_path=db_path, active_jobs=active_jobs)

    _render_workflow_job_log_viewer(workflow_jobs)


def _render_optimization_results_tab(df_optimize) -> None:
    st.subheader("Optimization Results")
    if df_optimize.empty:
        st.info("No optimization_results rows found in Postgres yet.")
        return

    opt_run_ids = sorted(df_optimize["run_id"].dropna().astype(str).unique().tolist())
    opt_stages = sorted(df_optimize["stage"].dropna().astype(str).unique().tolist())

    opt_col_1, opt_col_2 = st.columns(2)
    selected_opt_run = opt_col_1.selectbox(
        "Optimization Run ID",
        ["All", *opt_run_ids],
        key="opt_run_id_filter",
    )
    selected_opt_stage = opt_col_2.selectbox(
        "Optimization Stage",
        ["All", *opt_stages],
        key="opt_stage_filter",
    )

    opt_filtered = df_optimize.copy()
    if selected_opt_run != "All":
        opt_filtered = opt_filtered[opt_filtered["run_id"].astype(str) == selected_opt_run]
    if selected_opt_stage != "All":
        opt_filtered = opt_filtered[opt_filtered["stage"].astype(str) == selected_opt_stage]

    if opt_filtered.empty:
        st.warning("No rows matched the optimization filters.")
        return

    sharpe_series = pd.to_numeric(opt_filtered["sharpe"], errors="coerce")
    robust_series = pd.to_numeric(opt_filtered["robustness_score"], errors="coerce")
    train_series = pd.to_numeric(opt_filtered["train_sharpe"], errors="coerce")
    top_idx = sharpe_series.idxmax() if sharpe_series.notna().any() else None
    top_row = opt_filtered.loc[top_idx] if top_idx is not None else None

    optm1, optm2, optm3, optm4 = st.columns(4)
    optm1.metric("Rows", f"{len(opt_filtered)}")
    optm2.metric(
        "Best Sharpe",
        f"{float(sharpe_series.max()):.4f}" if sharpe_series.notna().any() else "N/A",
    )
    optm3.metric(
        "Median Sharpe",
        f"{float(sharpe_series.median()):.4f}" if sharpe_series.notna().any() else "N/A",
    )
    optm4.metric(
        "Median Robustness",
        f"{float(robust_series.median()):.4f}" if robust_series.notna().any() else "N/A",
    )

    if sharpe_series.notna().any() and train_series.notna().any():
        fig_opt_scatter = go.Figure()
        fig_opt_scatter.add_trace(
            go.Scatter(
                x=train_series,
                y=sharpe_series,
                mode="markers",
                marker=dict(size=8, color="#2b6cb0", opacity=0.8),
                text=opt_filtered["stage"].astype(str),
                hovertemplate=(
                    "Stage: %{text}<br>"
                    "Train Sharpe: %{x:.4f}<br>"
                    "Current Sharpe: %{y:.4f}<extra></extra>"
                ),
                name="Candidates",
            )
        )
        fig_opt_scatter.update_layout(
            title="Optimization Candidate Quality",
            xaxis_title="Train Sharpe",
            yaxis_title="Sharpe",
            template="plotly_white",
        )
        st.plotly_chart(fig_opt_scatter, use_container_width=True)

    table_cols = [
        "created_at",
        "run_id",
        "stage",
        "sharpe",
        "train_sharpe",
        "robustness_score",
        "cagr",
        "mdd",
    ]
    if "params" in opt_filtered.columns:
        opt_filtered = opt_filtered.copy()
        opt_filtered["params_view"] = opt_filtered["params"].apply(
            lambda v: json.dumps(v, ensure_ascii=False)
        )
        table_cols.append("params_view")
    st.dataframe(
        opt_filtered[table_cols].sort_values("created_at", ascending=False).head(500),
        use_container_width=True,
    )

    if top_row is not None:
        st.caption("Best row by Sharpe")
        st.json(
            {
                "run_id": top_row.get("run_id"),
                "stage": top_row.get("stage"),
                "sharpe": float(top_row.get("sharpe", 0.0)),
                "params": top_row.get("params", {}),
                "extra": top_row.get("extra", {}),
            }
        )


def _render_report_tab(
    *,
    db_path,
    refresh_counter,
    strategy_options,
    strategy_name,
    strategy_params,
    runner_initial_capital,
    runner_leverage,
    runner_symbols,
    runner_timeframe,
    runner_data_source,
    runner_timeout_sec,
    runner_env_overrides,
    market_db_path,
    market_exchange,
    optimize_folds,
    optimize_trials,
    optimize_workers,
    persist_best_params,
    opt_space_error,
    run_stale_sec,
    summary,
    performance,
    active_run_id,
    resolved_source,
    period_preset,
    df_equity,
    trade_analytics,
    df_risk,
    df_hb,
    mirror_snapshot,
    mirror_balance_equity,
) -> None:
    st.subheader("No-Code Workflow Control")
    st.caption(
        f"Runner overrides -> initial_equity={runner_initial_capital:.2f}, "
        f"leverage={runner_leverage}, timeframe={runner_timeframe}, symbols={runner_symbols}"
    )
    if opt_space_error:
        st.error(opt_space_error)

    workflow_jobs = load_workflow_jobs(db_path, refresh_counter=refresh_counter)
    active_live_jobs = _select_active_live_jobs(workflow_jobs)
    if not active_live_jobs.empty:
        st.warning(
            f"{len(active_live_jobs)} live job(s) already active. "
            "Stop existing live jobs before launching a new one."
        )

    live_runner_selection = _render_live_runner_settings(
        strategy_options=strategy_options,
        strategy_name=strategy_name,
    )
    _render_managed_run_launch_controls(
        db_path=db_path,
        launch_context=_ManagedRunLaunchContext(
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            runner_data_source=runner_data_source,
            market_db_path=market_db_path,
            market_exchange=market_exchange,
            runner_env_overrides=runner_env_overrides,
            optimize_folds=optimize_folds,
            optimize_trials=optimize_trials,
            optimize_workers=optimize_workers,
            persist_best_params=persist_best_params,
            runner_leverage=runner_leverage,
        ),
        live_runner_selection=live_runner_selection,
        active_live_jobs=active_live_jobs,
    )
    _render_workflow_jobs_section(db_path=db_path, refresh_counter=refresh_counter)
    _render_ghost_cleanup_section(db_path=db_path, run_stale_sec=run_stale_sec)
    _render_snapshot_report_section(
        summary=summary,
        performance=performance,
        active_run_id=active_run_id,
        resolved_source=resolved_source,
        strategy_name=strategy_name,
        period_preset=period_preset,
        df_equity=df_equity,
        trade_analytics=trade_analytics,
        df_risk=df_risk,
        df_hb=df_hb,
        runner_initial_capital=runner_initial_capital,
        runner_leverage=runner_leverage,
        runner_symbols=runner_symbols,
        runner_timeframe=runner_timeframe,
        runner_data_source=runner_data_source,
        runner_timeout_sec=runner_timeout_sec,
        strategy_params=strategy_params,
        mirror_snapshot=mirror_snapshot,
        mirror_balance_equity=mirror_balance_equity,
    )


def render_main_dashboard() -> None:
    selection_controls = _render_dashboard_selection_controls()
    data_source = selection_controls.data_source
    db_path = selection_controls.db_path
    market_db_path = selection_controls.market_db_path
    market_exchange = selection_controls.market_exchange
    market_timeframe = selection_controls.market_timeframe
    strategy_options = selection_controls.strategy_options
    strategy_name = selection_controls.strategy_name
    auto_refresh_enabled = selection_controls.auto_refresh_enabled
    refresh_interval_sec = selection_controls.refresh_interval_sec
    max_points = selection_controls.max_points
    auto_downsample = selection_controls.auto_downsample
    downsample_target_points = selection_controls.downsample_target_points
    pin_to_running = selection_controls.pin_to_running
    filter_runs_by_strategy = selection_controls.filter_runs_by_strategy
    run_stale_sec = selection_controls.run_stale_sec
    period_preset = selection_controls.period_preset
    custom_start = selection_controls.custom_start
    custom_end = selection_controls.custom_end

    refresh_counter, refresh_mode = _setup_auto_refresh(
        auto_refresh_enabled,
        refresh_interval_sec,
    )
    params = load_params(strategy_name)
    st.sidebar.caption(
        f"Refresh mode: {refresh_mode} | tick: {refresh_counter} | interval: {_safe_interval_sec(refresh_interval_sec)}s"
    )

    execution_lab_controls = _render_execution_lab_controls()
    runner_initial_capital = execution_lab_controls.runner_initial_capital
    runner_leverage = execution_lab_controls.runner_leverage
    runner_symbols = list(execution_lab_controls.runner_symbols)
    runner_timeframe = execution_lab_controls.runner_timeframe
    runner_timeout_sec = execution_lab_controls.runner_timeout_sec
    runner_data_source = execution_lab_controls.runner_data_source

    strategy_params = _merged_strategy_params(strategy_name, params)
    with st.sidebar.expander("Strategy Parameter Overrides", expanded=False):
        if strategy_name == "RsiStrategy":
            strategy_params["rsi_period"] = int(
                st.number_input(
                    "rsi_period",
                    min_value=2,
                    max_value=200,
                    value=int(strategy_params.get("rsi_period", 14)),
                    step=1,
                )
            )
            strategy_params["oversold"] = float(
                st.number_input(
                    "oversold",
                    min_value=1.0,
                    max_value=99.0,
                    value=float(strategy_params.get("oversold", 30.0)),
                    step=0.5,
                )
            )
            strategy_params["overbought"] = float(
                st.number_input(
                    "overbought",
                    min_value=1.0,
                    max_value=99.0,
                    value=float(strategy_params.get("overbought", 70.0)),
                    step=0.5,
                )
            )
        elif strategy_name == "MovingAverageCrossStrategy":
            strategy_params["short_window"] = int(
                st.number_input(
                    "short_window",
                    min_value=2,
                    max_value=500,
                    value=int(strategy_params.get("short_window", 10)),
                    step=1,
                )
            )
            strategy_params["long_window"] = int(
                st.number_input(
                    "long_window",
                    min_value=3,
                    max_value=1000,
                    value=int(strategy_params.get("long_window", 30)),
                    step=1,
                )
            )
        elif strategy_name == "PairTradingZScoreStrategy":
            strategy_params["lookback_window"] = int(
                st.number_input(
                    "lookback_window",
                    min_value=10,
                    max_value=2000,
                    value=int(strategy_params.get("lookback_window", 96)),
                    step=1,
                )
            )
            strategy_params["hedge_window"] = int(
                st.number_input(
                    "hedge_window",
                    min_value=10,
                    max_value=4000,
                    value=int(strategy_params.get("hedge_window", 192)),
                    step=1,
                )
            )
            strategy_params["entry_z"] = float(
                st.number_input(
                    "entry_z",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(strategy_params.get("entry_z", 2.0)),
                    step=0.05,
                )
            )
            strategy_params["exit_z"] = float(
                st.number_input(
                    "exit_z",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(strategy_params.get("exit_z", 0.35)),
                    step=0.05,
                )
            )
            strategy_params["stop_z"] = float(
                st.number_input(
                    "stop_z",
                    min_value=0.2,
                    max_value=12.0,
                    value=float(strategy_params.get("stop_z", 3.5)),
                    step=0.1,
                )
            )
            strategy_params["min_correlation"] = float(
                st.number_input(
                    "min_correlation",
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(strategy_params.get("min_correlation", 0.15)),
                    step=0.05,
                )
            )
            strategy_params["max_hold_bars"] = int(
                st.number_input(
                    "max_hold_bars",
                    min_value=1,
                    max_value=10000,
                    value=int(strategy_params.get("max_hold_bars", 240)),
                    step=1,
                )
            )
            strategy_params["cooldown_bars"] = int(
                st.number_input(
                    "cooldown_bars",
                    min_value=0,
                    max_value=5000,
                    value=int(strategy_params.get("cooldown_bars", 6)),
                    step=1,
                )
            )
            strategy_params["reentry_z_buffer"] = float(
                st.number_input(
                    "reentry_z_buffer",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(strategy_params.get("reentry_z_buffer", 0.20)),
                    step=0.01,
                )
            )
            strategy_params["min_z_turn"] = float(
                st.number_input(
                    "min_z_turn",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(strategy_params.get("min_z_turn", 0.05)),
                    step=0.01,
                )
            )
            strategy_params["stop_loss_pct"] = float(
                st.number_input(
                    "stop_loss_pct",
                    min_value=0.001,
                    max_value=0.50,
                    value=float(strategy_params.get("stop_loss_pct", 0.04)),
                    step=0.001,
                )
            )
            strategy_params["min_abs_beta"] = float(
                st.number_input(
                    "min_abs_beta",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(strategy_params.get("min_abs_beta", 0.02)),
                    step=0.01,
                )
            )
            strategy_params["max_abs_beta"] = float(
                st.number_input(
                    "max_abs_beta",
                    min_value=0.1,
                    max_value=30.0,
                    value=float(strategy_params.get("max_abs_beta", 6.0)),
                    step=0.1,
                )
            )
            strategy_params["min_volume_window"] = int(
                st.number_input(
                    "min_volume_window",
                    min_value=1,
                    max_value=5000,
                    value=int(strategy_params.get("min_volume_window", 24)),
                    step=1,
                )
            )
            strategy_params["min_volume_ratio"] = float(
                st.number_input(
                    "min_volume_ratio",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(strategy_params.get("min_volume_ratio", 0.0)),
                    step=0.01,
                )
            )
            strategy_params["use_log_price"] = bool(
                st.checkbox(
                    "use_log_price",
                    value=bool(strategy_params.get("use_log_price", True)),
                )
            )

            default_symbol_x = str(
                strategy_params.get("symbol_x") or (runner_symbols[0] if runner_symbols else "")
            )
            default_symbol_y = str(
                strategy_params.get("symbol_y")
                or (runner_symbols[1] if len(runner_symbols) > 1 else default_symbol_x)
            )
            strategy_params["symbol_x"] = st.text_input("symbol_x", value=default_symbol_x)
            strategy_params["symbol_y"] = st.text_input("symbol_y", value=default_symbol_y)

    with st.sidebar.expander("Optimization Search Space", expanded=False):
        default_optuna_cfg = _strategy_default_optuna(strategy_name)
        default_grid_cfg = _strategy_default_grid(strategy_name)
        if strategy_name == str(OptimizationConfig.STRATEGY_NAME):
            default_optuna_cfg = strategy_registry.resolve_optuna_config(
                strategy_name,
                OptimizationConfig.OPTUNA_CONFIG if isinstance(OptimizationConfig.OPTUNA_CONFIG, dict) else {},
            )
            default_grid_cfg = strategy_registry.resolve_grid_config(
                strategy_name,
                OptimizationConfig.GRID_CONFIG if isinstance(OptimizationConfig.GRID_CONFIG, dict) else {},
            )

        default_optuna_json = json.dumps(default_optuna_cfg, indent=2, ensure_ascii=False)
        default_grid_json = json.dumps(default_grid_cfg, indent=2, ensure_ascii=False)
        optuna_json_raw = st.text_area("OPTUNA config JSON", value=default_optuna_json, height=140)
        grid_json_raw = st.text_area("GRID config JSON", value=default_grid_json, height=140)
        optimize_folds = st.number_input(
            "Walk-forward folds",
            min_value=1,
            max_value=20,
            value=int(OptimizationConfig.WALK_FORWARD_FOLDS),
            step=1,
        )
        optimize_trials = st.number_input(
            "Optuna trials",
            min_value=1,
            max_value=5000,
            value=int(
                default_optuna_cfg.get("n_trials", OptimizationConfig.OPTUNA_CONFIG.get("n_trials", 20))
            ),
            step=1,
        )
        optimize_workers = st.number_input(
            "Optimization workers",
            min_value=1,
            max_value=max(1, (os.cpu_count() or 4) * 2),
            value=int(OptimizationConfig.MAX_WORKERS),
            step=1,
        )
        persist_best_params = st.checkbox(
            "Persist best params after optimize", value=bool(OptimizationConfig.PERSIST_BEST_PARAMS)
        )

    optuna_config_for_runner = strategy_registry.resolve_optuna_config(strategy_name, default_optuna_cfg)
    grid_config_for_runner = strategy_registry.resolve_grid_config(strategy_name, default_grid_cfg)
    opt_space_error = None
    try:
        parsed_optuna = json.loads(optuna_json_raw)
        parsed_grid = json.loads(grid_json_raw)
        if isinstance(parsed_optuna, dict):
            optuna_config_for_runner = strategy_registry.resolve_optuna_config(
                strategy_name,
                parsed_optuna,
            )
        else:
            opt_space_error = "OPTUNA config JSON must be an object."
        if isinstance(parsed_grid, dict):
            grid_config_for_runner = strategy_registry.resolve_grid_config(
                strategy_name,
                parsed_grid,
            )
        else:
            opt_space_error = "GRID config JSON must be an object."
    except Exception as exc:
        opt_space_error = f"Optimization JSON parse error: {exc}"

    runner_env_overrides = _build_runtime_env_overrides(
        initial_capital=runner_initial_capital,
        leverage=runner_leverage,
        timeframe=runner_timeframe,
        symbols=runner_symbols,
        strategy_name=strategy_name,
        optuna_config=optuna_config_for_runner,
        grid_config=grid_config_for_runner,
    )

    if "runner_last_result" not in st.session_state:
        st.session_state["runner_last_result"] = None

    df_equity = pd.DataFrame()
    df_trades = pd.DataFrame()
    df_orders = pd.DataFrame()
    df_risk = pd.DataFrame()
    df_hb = pd.DataFrame()
    df_order_states = pd.DataFrame()
    df_market = pd.DataFrame()
    df_optimize = pd.DataFrame()
    active_run_id = None
    resolved_source = None
    runs_df = pd.DataFrame()

    _ensure_workflow_jobs_schema(db_path)
    _refresh_workflow_jobs(db_path)

    use_state = data_source == "Postgres" or (data_source == "Auto" and bool(_resolve_postgres_dsn(db_path)))
    if use_state and _resolve_postgres_dsn(db_path):
        try:
            runs_df = load_runs(db_path, refresh_counter=refresh_counter)
            runs_df = _annotate_run_health(runs_df, stale_after_sec=run_stale_sec)
            df_optimize = load_optimization_results_state(db_path, refresh_counter=refresh_counter)
            if not runs_df.empty:
                run_options = runs_df["run_id"].astype(str).tolist()
                strategy_run_options = _runs_for_strategy(runs_df, strategy_name)
                if filter_runs_by_strategy and strategy_run_options:
                    run_options = strategy_run_options
                elif filter_runs_by_strategy and not strategy_run_options:
                    st.sidebar.info(
                        "No Postgres runs tagged with selected strategy yet. Showing all runs instead."
                    )

                default_run = _latest_running_run_id(runs_df) if pin_to_running else None
                if default_run and _equity_rows_for_run(runs_df, default_run) <= 0:
                    default_run = None
                if default_run and default_run not in run_options:
                    default_run = None
                if not default_run:
                    if strategy_run_options:
                        strategy_runs_df = runs_df[runs_df["run_id"].astype(str).isin(run_options)]
                        default_run = _latest_run_with_equity(strategy_runs_df)
                    else:
                        default_run = _latest_run_with_equity(runs_df)
                if not default_run:
                    default_run = run_options[0]

                strategy_changed = st.session_state.get("dashboard_strategy_name") != strategy_name
                st.session_state["dashboard_strategy_name"] = strategy_name

                if "dashboard_run_id" not in st.session_state:
                    st.session_state["dashboard_run_id"] = default_run
                if strategy_changed:
                    st.session_state["dashboard_run_id"] = default_run
                if pin_to_running:
                    st.session_state["dashboard_run_id"] = default_run
                if st.session_state["dashboard_run_id"] not in run_options:
                    st.session_state["dashboard_run_id"] = default_run

                selected_idx = run_options.index(st.session_state["dashboard_run_id"])
                active_run_id = st.sidebar.selectbox("Run ID", run_options, index=selected_idx)
                st.session_state["dashboard_run_id"] = active_run_id

                df_equity = load_equity_state(
                    db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
                )
                df_trades = load_fills_state(
                    db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
                )
                df_orders = load_orders_state(
                    db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
                )
                df_risk = load_risk_events_state(
                    db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
                )
                df_hb = load_heartbeats_state(
                    db_path, active_run_id, refresh_counter=refresh_counter, max_points=max_points
                )
                df_order_states = load_order_states_state(
                    db_path,
                    active_run_id,
                    refresh_counter=refresh_counter,
                    max_points=max_points,
                )
                resolved_source = "Postgres"
        except Exception:
            runs_df = pd.DataFrame()
            df_optimize = pd.DataFrame()

    if resolved_source is None or (df_equity.empty and data_source == "Auto"):
        fallback_equity = load_equity_csv(refresh_counter=refresh_counter, max_points=max_points)
        fallback_trades = load_trades_csv(refresh_counter=refresh_counter, max_points=max_points)
        if not fallback_equity.empty:
            df_equity = fallback_equity
            df_trades = fallback_trades
            active_run_id = None
            resolved_source = "CSV"

    if resolved_source == "CSV" and data_source in {"Auto", "CSV"}:
        st.warning(
            "Dashboard is currently rendering CSV fallback data. In this mode, changing strategy updates "
            "strategy indicators/config controls, but core PnL/equity history remains the same CSV sample until "
            "a Postgres run with equity rows is available."
        )

    if resolved_source is None:
        resolved_source = "None"

    period_start, period_end = _resolve_period_window(
        period_preset,
        custom_start,
        custom_end,
        df_equity,
        dt_col="datetime",
    )

    df_equity = _apply_period_filter(df_equity, "datetime", period_start, period_end)
    df_trades = _apply_period_filter(df_trades, "datetime", period_start, period_end)
    df_orders = _apply_period_filter(df_orders, "created_at", period_start, period_end)
    df_risk = _apply_period_filter(df_risk, "event_time", period_start, period_end)
    df_hb = _apply_period_filter(df_hb, "heartbeat_time", period_start, period_end)
    df_order_states = _apply_period_filter(df_order_states, "event_time", period_start, period_end)

    if auto_downsample and not df_equity.empty:
        plot_equity = _downsample_frame(df_equity, downsample_target_points)
    else:
        plot_equity = df_equity

    if auto_downsample and not df_trades.empty:
        plot_trades = _downsample_frame(df_trades, downsample_target_points)
    else:
        plot_trades = df_trades

    if not runs_df.empty and active_run_id:
        symbols = _extract_run_symbols(runs_df, active_run_id)
    else:
        symbols = list(BaseConfig.SYMBOLS)
    market_symbol = st.sidebar.selectbox(
        "Market Symbol", symbols if symbols else list(BaseConfig.SYMBOLS)
    )

    if str(market_db_path).strip():
        df_market = load_market_ohlcv_state(
            market_db_path,
            market_symbol,
            market_timeframe,
            market_exchange,
            refresh_counter=refresh_counter,
            max_points=max_points,
        )
    df_market = _apply_period_filter(df_market, "datetime", period_start, period_end)
    if auto_downsample and not df_market.empty:
        plot_market = _downsample_frame(df_market, downsample_target_points)
    else:
        plot_market = df_market

    trade_analytics = compute_trade_analytics(df_trades)
    summary = build_summary(df_equity, trade_analytics)
    performance = build_performance_metrics(df_equity)
    latest_data_ts, latency_sec = _compute_data_latency_seconds(df_equity)
    mirror_balance_equity = _build_balance_equity_frame(
        df_equity,
        trade_analytics,
        initial_equity=summary.get("initial_equity", 0.0),
    )
    mirror_snapshot = _build_mirror_snapshot(summary, mirror_balance_equity)

    st.subheader("PC Mirror Snapshot")
    st.caption("Reference-style KPI strip + equity/balance/drawdown narrative")
    _render_mirror_cards(mirror_snapshot)
    mirror_left, mirror_right = st.columns(2)
    if not mirror_balance_equity.empty:
        with mirror_left:
            st.plotly_chart(
                _build_mirror_equity_curve_figure(mirror_balance_equity),
                use_container_width=True,
                key="mirror_equity_curve",
            )
        with mirror_right:
            st.plotly_chart(
                _build_mirror_balance_equity_figure(mirror_balance_equity, mirror_snapshot),
                use_container_width=True,
                key="mirror_balance_equity",
            )
    else:
        with mirror_left:
            st.info("No equity data for mirror-style curve yet.")
        with mirror_right:
            st.info("No balance/equity timeline for mirror-style chart yet.")

    st.header("Overview")
    st.caption("Performance summary is grouped by Context, Equity, Risk, and Trade Quality.")

    ctx1, ctx2, ctx3, ctx4, ctx5, ctx6 = st.columns(6)
    ctx1.metric("Source", resolved_source)
    ctx2.metric("Bars", f"{summary['bars']}")
    ctx3.metric("Fills", f"{summary['fills']}")
    ctx4.metric("Avg Fills/Day", f"{summary['fills_per_day']:.2f}")
    ctx5.metric("Closed PnL", _format_signed_dollar(summary.get("total_net_profit"), digits=2))
    ctx6.metric("Win Rate", f"{summary['win_rate']:.2%}")

    st.markdown("**Equity Context**")
    eq1, eq2, eq3, eq4 = st.columns(4)
    eq1.metric("Initial Equity", f"{summary['initial_equity']:.2f}")
    eq2.metric("Final Equity", f"{summary['final_equity']:.2f}")
    eq3.metric("Configured Initial Equity", f"{runner_initial_capital:.2f}")
    eq4.metric("Configured Leverage", f"{int(runner_leverage)}x")

    st.markdown("**PnL / Drawdown**")
    mx1, mx2, mx3, mx4 = st.columns(4)
    mx1.metric("Open P/L", _format_signed_dollar(summary.get("open_pnl"), digits=2))
    mx2.metric("Total (C+O)", _format_signed_dollar(summary.get("total_c_plus_o"), digits=2))
    mx3.metric("R/MDD", f"{_safe_float(summary.get('r_mdd'), 0.0):.2f}x")
    mx4.metric(
        "Equity MDD",
        (
            f"${_safe_float(summary.get('equity_drawdown_maximal'), 0.0):,.2f} "
            f"({_safe_float(summary.get('equity_drawdown_relative_pct'), 0.0):.2%})"
        ),
    )

    if performance:
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Total Return", f"{performance.get('Total Return', 0.0):.2%}")
        p2.metric("Cumulative Return", f"{performance.get('Cumulative Return', 0.0):.2%}")
        p3.metric("Sharpe", f"{performance.get('Sharpe Ratio', 0.0):.3f}")
        p4.metric("Sortino", f"{performance.get('Sortino Ratio', 0.0):.3f}")
        p5.metric("Max Drawdown", f"{performance.get('Max Drawdown', 0.0):.2%}")
        p6.metric("Funding (Net)", f"{performance.get('Funding (Net)', 0.0):.4f}")

    st.markdown("**Trade Quality**")
    pf1, pf2, pf3, pf4, pf5, pf6 = st.columns(6)
    pf1.metric("Profit Factor", _format_metric_value("Profit Factor", summary.get("profit_factor")))
    pf2.metric("Recovery Factor", f"{_safe_float(summary.get('recovery_factor')):.3f}")
    pf3.metric("Expected Payoff", f"{_safe_float(summary.get('expected_payoff')):.4f}")
    pf4.metric("Profit Trades", str(summary.get("profit_trades_text", "0 (0.00%)")))
    pf5.metric("Loss Trades", str(summary.get("loss_trades_text", "0 (0.00%)")))
    pf6.metric("Payoff Ratio", f"{_safe_float(summary.get('payoff_ratio')):.3f}")

    dir1, dir2, dir3, dir4 = st.columns(4)
    dir1.metric("Long Trades (Win %)", str(summary.get("long_trades_win_pct", "0 (0.00%)")))
    dir2.metric("Short Trades (Win %)", str(summary.get("short_trades_win_pct", "0 (0.00%)")))
    dir3.metric("Avg Holding", _format_duration_seconds(summary.get("holding_time_avg_sec", 0.0)))
    dir4.metric(
        "Avg Profit / Loss",
        (
            f"{_safe_float(summary.get('avg_profit_trade'), 0.0):.2f} / "
            f"{_safe_float(summary.get('avg_loss_trade'), 0.0):.2f}"
        ),
    )

    if latest_data_ts is not None and latency_sec is not None:
        st.caption(
            f"Data timestamp: {latest_data_ts} | latency: {latency_sec:.1f}s | "
            f"period preset: {period_preset} | range: {summary['period_start']} -> {summary['period_end']}"
        )

    if not runs_df.empty and "effective_status" in runs_df.columns:
        healthy_runs = int((runs_df["effective_status"] == "RUNNING_HEALTHY").sum())
        stale_runs = int((runs_df["effective_status"] == "RUNNING_STALE").sum())
        no_telemetry_runs = int((runs_df["effective_status"] == "RUNNING_NO_TELEMETRY").sum())
        st.caption(
            f"Live run health -> healthy: {healthy_runs}, stale: {stale_runs}, "
            f"no-telemetry: {no_telemetry_runs} (threshold={int(run_stale_sec)}s)"
        )

    st.subheader("Strategy Parameters")
    st.json(strategy_params)

    tab_overview, tab_exec, tab_risk, tab_market, tab_opt, tab_report, tab_raw = st.tabs(
        [
            "Performance & Price",
            "Execution Analytics",
            "Risk & Health",
            "Market Data",
            "Optimization Insights",
            "Report Export",
            "Raw Data",
        ]
    )

    with tab_overview:
        with st.expander("Metric Definitions", expanded=False):
            metric_rows = []
            for metric_name, definition in METRIC_DEFINITIONS.items():
                metric_rows.append(
                    {
                        "Metric": metric_name,
                        "Value": _format_metric_value(
                            metric_name,
                            _metric_value(metric_name, performance, summary),
                        ),
                        "Definition": definition,
                    }
                )
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

        mt5_rows = _build_mt5_summary_rows(summary)
        if not mt5_rows.empty:
            st.subheader("MT5-Style Summary Grid")
            st.dataframe(mt5_rows, use_container_width=True, hide_index=True)

        st.caption(
            f"Data context: run={active_run_id if active_run_id else 'N/A'} | "
            f"source={resolved_source} | symbol={market_symbol} | timeframe={market_timeframe} | "
            f"exchange={market_exchange} | market_db={market_db_path}"
        )

        if not plot_equity.empty:
            st.plotly_chart(
                _build_equity_curve_figure_data(plot_equity),
                use_container_width=True,
            )

            benchmark_figure = _build_benchmark_price_figure_data(plot_equity)
            if benchmark_figure is not None:
                st.plotly_chart(benchmark_figure, use_container_width=True)

            funding_figure = _build_funding_figure_data(plot_equity)
            if funding_figure is not None:
                st.plotly_chart(funding_figure, use_container_width=True)

            cumulative_return_figure = _build_cumulative_return_figure_data(df_equity, performance)
            if cumulative_return_figure is not None:
                st.plotly_chart(cumulative_return_figure, use_container_width=True)

            st.plotly_chart(
                _build_drawdown_figure_data(plot_equity),
                use_container_width=True,
            )

            monthly_table = _build_monthly_returns_table(df_equity, performance)
            if not monthly_table.empty:
                st.plotly_chart(
                    _build_monthly_returns_heatmap_data(
                        monthly_table,
                        safe_float=_safe_float,
                    ),
                    use_container_width=True,
                )

        if not plot_market.empty:
            fig_price = go.Figure()
            fig_price.add_trace(
                go.Candlestick(
                    x=plot_market["datetime"],
                    open=plot_market["open"],
                    high=plot_market["high"],
                    low=plot_market["low"],
                    close=plot_market["close"],
                    name=f"{market_symbol} price",
                )
            )

            if not plot_trades.empty:
                symbol_trades = plot_trades[plot_trades["symbol"] == market_symbol]
                if not symbol_trades.empty:
                    enriched = compute_trade_analytics(symbol_trades)
                    for direction, color, symbol_shape in [
                        ("BUY", "#0db39e", "triangle-up"),
                        ("SELL", "#f05a66", "triangle-down"),
                    ]:
                        part = enriched[enriched["direction"] == direction]
                        if part.empty:
                            continue
                        custom_cols = [
                            part["quantity"],
                            part["position_after"],
                            part["realized_pnl"],
                            part["realized_return_pct"],
                        ]
                        custom_data = pd.concat(custom_cols, axis=1).to_numpy()
                        fig_price.add_trace(
                            go.Scatter(
                                x=part["datetime"],
                                y=part["price"],
                                mode="markers",
                                name=direction,
                                marker=dict(symbol=symbol_shape, size=10, color=color),
                                customdata=custom_data,
                                hovertemplate=(
                                    "%{x}<br>"
                                    f"{direction} @ %{{y:.4f}}<br>"
                                    "Qty: %{customdata[0]:.4f}<br>"
                                    "Position after: %{customdata[1]:.4f}<br>"
                                    "Realized PnL: %{customdata[2]:.4f}<br>"
                                    "Trade Return: %{customdata[3]:.4f}%<extra></extra>"
                                ),
                            )
                        )

            fig_price.update_layout(
                title=f"{market_symbol} Price + Buy/Sell Markers",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                xaxis_rangeslider_visible=True,
            )
            st.plotly_chart(fig_price, use_container_width=True)

    with tab_exec:
        execution_metric_rows = _build_execution_metric_rows_data(
            summary,
            format_duration_seconds=_format_duration_seconds,
        )
        for metric_row in execution_metric_rows:
            metric_columns = st.columns(len(metric_row))
            for metric_column, (label, value) in zip(metric_columns, metric_row, strict=False):
                metric_column.metric(label, value)

        direction_table = _build_direction_table_data(
            summary,
            safe_float=_safe_float,
        )
        st.dataframe(direction_table, use_container_width=True, hide_index=True)

        if not trade_analytics.empty:
            closed = _filter_closed_trade_analytics_data(trade_analytics)
            if not closed.empty:
                st.plotly_chart(
                    _build_trade_pnl_figure_data(closed),
                    use_container_width=True,
                )

                st.plotly_chart(
                    _build_cumulative_realized_pnl_figure_data(closed),
                    use_container_width=True,
                )

                streak_figure = _build_streak_distribution_figure_data(
                    closed,
                    streak_groups=_streak_groups,
                )
                if streak_figure is not None:
                    st.plotly_chart(streak_figure, use_container_width=True)

        if not df_orders.empty:
            st.plotly_chart(
                _build_order_status_figure_data(df_orders),
                use_container_width=True,
            )

    with tab_risk:
        if not df_risk.empty:
            st.plotly_chart(
                _build_risk_reason_figure_data(df_risk),
                use_container_width=True,
            )
        else:
            st.info("No risk events recorded for selected run/data source.")

        if not df_hb.empty:
            hb, avg_hb = _prepare_heartbeat_interval_frame_data(df_hb)
            st.metric("Avg Heartbeat Interval (sec)", f"{avg_hb:.2f}")
            st.plotly_chart(
                _build_heartbeat_interval_figure_data(hb),
                use_container_width=True,
            )
        else:
            st.info("No heartbeats recorded for selected run/data source.")

        if not df_order_states.empty:
            st.plotly_chart(
                _build_order_state_figure_data(df_order_states),
                use_container_width=True,
            )

        trace_df = _build_strategy_process_trace_frame_data(
            df_orders=df_orders,
            df_risk=df_risk,
            df_hb=df_hb,
            df_order_states=df_order_states,
        )
        if not trace_df.empty:
            st.subheader("Strategy Process Trace")
            st.dataframe(trace_df, use_container_width=True)

    with tab_market:
        if not plot_market.empty:
            colm1, colm2, colm3, colm4 = st.columns(4)
            market_metrics = _build_market_summary_metrics_data(plot_market)
            colm1.metric("Market Bars", market_metrics["market_bars"])
            colm2.metric("First Price", market_metrics["first_price"])
            colm3.metric("Last Price", market_metrics["last_price"])
            colm4.metric("Range", market_metrics["range"])

            st.plotly_chart(
                _build_market_close_figure_data(
                    plot_market,
                    market_symbol=market_symbol,
                    market_timeframe=market_timeframe,
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                _build_market_volume_figure_data(plot_market),
                use_container_width=True,
            )

            st.subheader("Strategy Indicator View")
            if strategy_name == "PairTradingZScoreStrategy":
                pair_symbol_x = str(
                    strategy_params.get("symbol_x") or (runner_symbols[0] if runner_symbols else "")
                )
                pair_symbol_y = str(
                    strategy_params.get("symbol_y")
                    or (runner_symbols[1] if len(runner_symbols) > 1 else pair_symbol_x)
                )
                if pair_symbol_x == pair_symbol_y or not pair_symbol_x:
                    st.warning("Pair strategy needs two different symbols (symbol_x / symbol_y).")
                else:
                    pair_x_df = pd.DataFrame()
                    pair_y_df = pd.DataFrame()
                    if _resolve_postgres_dsn(db_path):
                        pair_x_df = load_market_ohlcv_state(
                            db_path,
                            pair_symbol_x,
                            market_timeframe,
                            market_exchange,
                            refresh_counter=refresh_counter,
                            max_points=max_points,
                        )
                        pair_y_df = load_market_ohlcv_state(
                            db_path,
                            pair_symbol_y,
                            market_timeframe,
                            market_exchange,
                            refresh_counter=refresh_counter,
                            max_points=max_points,
                        )

                    pair_x_df = _apply_period_filter(pair_x_df, "datetime", period_start, period_end)
                    pair_y_df = _apply_period_filter(pair_y_df, "datetime", period_start, period_end)
                    pair_indicator_df = _build_pair_indicator_frame(
                        pair_x_df, pair_y_df, strategy_params
                    )
                    if pair_indicator_df.empty:
                        st.info("Not enough aligned market bars to compute pair indicators.")
                    else:
                        entry_z = float(strategy_params.get("entry_z", 2.0))
                        exit_z = float(strategy_params.get("exit_z", 0.35))
                        stop_z = float(strategy_params.get("stop_z", 3.5))
                        pm1, pm2, pm3, pm4 = st.columns(4)
                        pair_metrics = _build_pair_indicator_summary_data(
                            pair_indicator_df,
                            pair_symbol_x=pair_symbol_x,
                            pair_symbol_y=pair_symbol_y,
                        )
                        pm1.metric("Pair", pair_metrics["pair"])
                        pm2.metric("Latest Z", pair_metrics["latest_z"])
                        pm3.metric("Hedge Ratio", pair_metrics["hedge_ratio"])
                        pm4.metric("Correlation", pair_metrics["correlation"])

                        pair_plot_df = (
                            _downsample_frame(pair_indicator_df, downsample_target_points)
                            if auto_downsample
                            else pair_indicator_df
                        )

                        st.plotly_chart(
                            _build_pair_price_inputs_figure_data(
                                pair_plot_df,
                                pair_symbol_x=pair_symbol_x,
                                pair_symbol_y=pair_symbol_y,
                            ),
                            use_container_width=True,
                        )

                        st.plotly_chart(
                            _build_pair_zscore_figure_data(
                                pair_plot_df,
                                entry_z=entry_z,
                                exit_z=exit_z,
                                stop_z=stop_z,
                            ),
                            use_container_width=True,
                        )

                        st.plotly_chart(
                            _build_pair_spread_figure_data(pair_plot_df),
                            use_container_width=True,
                        )
            else:
                indicator_df = _build_strategy_indicator_frame(
                    plot_market, strategy_name, strategy_params
                )
                if indicator_df.empty:
                    st.info("Not enough market bars to compute strategy indicators.")
                elif strategy_name == "RsiStrategy":
                    rsi_period = max(2, int(strategy_params.get("rsi_period", 14)))
                    oversold = float(strategy_params.get("oversold", 30.0))
                    overbought = float(strategy_params.get("overbought", 70.0))

                    rm1, rm2, rm3 = st.columns(3)
                    rsi_metrics = _build_rsi_summary_metrics_data(
                        indicator_df,
                        rsi_period=rsi_period,
                        oversold=oversold,
                        overbought=overbought,
                    )
                    rm1.metric("RSI Period", rsi_metrics["rsi_period"])
                    rm2.metric("Latest RSI", rsi_metrics["latest_rsi"])
                    rm3.metric("RSI Zone", rsi_metrics["rsi_zone"])

                    st.plotly_chart(
                        _build_rsi_figure_data(
                            indicator_df,
                            rsi_period=rsi_period,
                            oversold=oversold,
                            overbought=overbought,
                        ),
                        use_container_width=True,
                    )

                    st.plotly_chart(
                        _build_rsi_signal_figure_data(
                            indicator_df,
                            oversold=oversold,
                            overbought=overbought,
                        ),
                        use_container_width=True,
                    )

                elif strategy_name == "MovingAverageCrossStrategy":
                    short_window = max(2, int(strategy_params.get("short_window", 10)))
                    long_window = max(short_window + 1, int(strategy_params.get("long_window", 30)))

                    mm1, mm2 = st.columns(2)
                    ma_metrics = _build_moving_average_summary_metrics_data(
                        short_window=short_window,
                        long_window=long_window,
                    )
                    mm1.metric("Short Window", ma_metrics["short_window"])
                    mm2.metric("Long Window", ma_metrics["long_window"])

                    st.plotly_chart(
                        _build_moving_average_figure_data(
                            indicator_df,
                            short_window=short_window,
                            long_window=long_window,
                        ),
                        use_container_width=True,
                    )
        else:
            st.info("No market_ohlcv rows available for selected symbol/timeframe/exchange.")

    with tab_opt:
        _render_optimization_results_tab(df_optimize)

    with tab_report:
        _render_report_tab(
            db_path=db_path,
            refresh_counter=refresh_counter,
            strategy_options=strategy_options,
            summary=summary,
            performance=performance,
            active_run_id=active_run_id,
            resolved_source=resolved_source,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            period_preset=period_preset,
            df_equity=df_equity,
            trade_analytics=trade_analytics,
            df_risk=df_risk,
            df_hb=df_hb,
            runner_initial_capital=runner_initial_capital,
            runner_leverage=runner_leverage,
            runner_symbols=runner_symbols,
            runner_timeframe=runner_timeframe,
            runner_data_source=runner_data_source,
            runner_timeout_sec=runner_timeout_sec,
            runner_env_overrides=runner_env_overrides,
            market_db_path=market_db_path,
            market_exchange=market_exchange,
            optimize_folds=optimize_folds,
            optimize_trials=optimize_trials,
            optimize_workers=optimize_workers,
            persist_best_params=persist_best_params,
            opt_space_error=opt_space_error,
            run_stale_sec=run_stale_sec,
            mirror_snapshot=mirror_snapshot,
            mirror_balance_equity=mirror_balance_equity,
        )

    with tab_raw:
        _render_raw_data_tab(
            active_run_id=active_run_id,
            resolved_source=resolved_source,
            market_symbol=market_symbol,
            market_timeframe=market_timeframe,
            market_exchange=market_exchange,
            runs_df=runs_df,
            df_equity=df_equity,
            trade_analytics=trade_analytics,
            df_orders=df_orders,
            df_risk=df_risk,
            df_hb=df_hb,
            df_order_states=df_order_states,
            df_market=df_market,
            df_optimize=df_optimize,
            db_path=db_path,
            refresh_counter=refresh_counter,
        )

    _render_missing_equity_warning(df_equity, runner_initial_capital)



def main() -> None:
    _render_dashboard_page_shell()
    _route_dashboard_view()
    render_main_dashboard()


main()
