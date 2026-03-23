"""Dashboard workflow-job launch helpers."""

from __future__ import annotations

import logging
import json
import os
import subprocess
import uuid
from pathlib import Path
from collections.abc import Callable, MutableMapping

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_RUNTIME_ROOT = PROJECT_ROOT / "var" / "dashboard"
WORKFLOW_LOG_DIR = WORKFLOW_RUNTIME_ROOT / "workflow_jobs"
WORKFLOW_CONTROL_DIR = WORKFLOW_RUNTIME_ROOT / "control"

DEFAULT_LQ_COMMAND = ("uv", "run", "lq")
WORKFLOW_JOBS_QUERY = """
SELECT job_id, workflow, status, requested_mode, strategy, command_json,
       env_json, pid, run_id, started_at, ended_at, exit_code,
       log_path, stop_file, metadata_json, last_updated
FROM workflow_jobs
ORDER BY COALESCE(started_at, last_updated) DESC
LIMIT %s
"""
WORKFLOW_JOB_DATETIME_COLUMNS = ("started_at", "ended_at", "last_updated")


def build_lq_command(subcommand: str, *args: str) -> tuple[str, ...]:
    return (*DEFAULT_LQ_COMMAND, str(subcommand), *tuple(str(arg) for arg in args))


def build_stop_file_path(
    job_id: str,
    *,
    control_dir: str | Path | None = None,
) -> str:
    resolved_control_dir = Path(control_dir) if control_dir is not None else WORKFLOW_CONTROL_DIR
    resolved_control_dir.mkdir(parents=True, exist_ok=True)
    return str(resolved_control_dir / f"{job_id}.stop")


def build_runtime_env_overrides(
    *,
    initial_capital: float,
    leverage: int,
    timeframe: str,
    symbols: tuple[str, ...] | list[str],
    strategy_name: str,
    optuna_config: dict[str, object],
    grid_config: dict[str, object],
) -> dict[str, str]:
    return {
        "LQ__TRADING__INITIAL_CAPITAL": str(float(initial_capital)),
        "LQ__BACKTEST__LEVERAGE": str(int(leverage)),
        "LQ__TRADING__TIMEFRAME": str(timeframe),
        "LQ__TRADING__SYMBOLS": json.dumps(list(symbols), ensure_ascii=True),
        "LQ__OPTIMIZATION__STRATEGY": str(strategy_name),
        "LQ__OPTIMIZATION__OPTUNA": json.dumps(optuna_config, ensure_ascii=True),
        "LQ__OPTIMIZATION__GRID": json.dumps(grid_config, ensure_ascii=True),
    }


def launch_managed_job(
    *,
    db_path: str,
    workflow: str,
    command: tuple[str, ...] | list[str],
    env_overrides: dict[str, str] | None,
    workflow_log_dir: str,
    session_state: MutableMapping[str, object],
    insert_workflow_job_row: Callable[[str, dict[str, object]], None],
    utc_now_iso: Callable[[], str],
    requested_mode: str | None = None,
    strategy: str | None = None,
    run_id: str | None = None,
    stop_file: str | None = None,
    metadata: dict[str, object] | None = None,
    cwd: str | None = None,
) -> str:
    workflow_processes = session_state.setdefault("workflow_processes", {})
    if not isinstance(workflow_processes, dict):
        workflow_processes = {}
        session_state["workflow_processes"] = workflow_processes

    resolved_workflow_log_dir = Path(workflow_log_dir)
    resolved_workflow_log_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    log_path = str(resolved_workflow_log_dir / f"{workflow}_{job_id}.log")
    normalized_command = [str(part) for part in command]
    normalized_env = {str(key): str(value) for key, value in (env_overrides or {}).items()}
    env = os.environ.copy()
    env.update(normalized_env)
    resolved_cwd = cwd or str(PROJECT_ROOT)
    normalized_metadata = dict(metadata or {})
    if not cwd:
        logger.warning(
            "launch_managed_job defaulted cwd to project root for workflow=%s job_id=%s",
            workflow,
            job_id,
        )
        normalized_metadata.setdefault("runtime_cwd", resolved_cwd)
        normalized_metadata.setdefault("runtime_cwd_source", "project_root_fallback")

    started_at = utc_now_iso()
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            normalized_command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=resolved_cwd,
        )

    insert_workflow_job_row(
        db_path,
        {
            "job_id": job_id,
            "workflow": str(workflow),
            "status": "RUNNING",
            "requested_mode": requested_mode,
            "strategy": strategy,
            "command_json": json.dumps(normalized_command, ensure_ascii=True),
            "env_json": json.dumps(normalized_env, ensure_ascii=True),
            "pid": int(process.pid),
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": None,
            "exit_code": None,
            "log_path": log_path,
            "stop_file": stop_file,
            "metadata_json": json.dumps(normalized_metadata, ensure_ascii=False),
            "last_updated": started_at,
        },
    )

    workflow_processes[job_id] = {
        "process": process,
        "log_path": log_path,
        "stop_file": stop_file,
    }
    return job_id


def request_job_stop(stop_file: str | None, *, timestamp: str) -> bool:
    if not stop_file:
        return False
    parent = os.path.dirname(stop_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(stop_file, "w", encoding="utf-8") as f:
        f.write(str(timestamp))
    return True


def refresh_workflow_jobs(
    *,
    db_path: str,
    session_state: MutableMapping[str, object],
    resolve_postgres_dsn: Callable[[str], str],
    ensure_workflow_jobs_schema: Callable[[str], None],
    connect_state_store: Callable[[str], object],
    is_process_running: Callable[[object], bool],
    update_workflow_job_row: Callable[..., None],
    utc_now_iso: Callable[[], str],
) -> None:
    if not resolve_postgres_dsn(db_path):
        logger.warning(
            "Skipping workflow job refresh because no Postgres DSN resolved for db_path=%s",
            db_path,
        )
        return
    ensure_workflow_jobs_schema(db_path)
    workflow_processes = session_state.setdefault("workflow_processes", {})
    if not isinstance(workflow_processes, dict):
        workflow_processes = {}
        session_state["workflow_processes"] = workflow_processes

    try:
        conn = connect_state_store(db_path)
    except Exception:
        logger.warning(
            "Unable to connect to workflow job store for db_path=%s; skipping refresh",
            db_path,
            exc_info=True,
        )
        return
    try:
        rows = conn.execute(
            """
            SELECT job_id, status, pid
            FROM workflow_jobs
            WHERE status IN ('RUNNING', 'STOP_REQUESTED')
            """
        ).fetchall()
    finally:
        conn.close()

    for job_id, status, pid in rows:
                entry = workflow_processes.get(job_id)
                if entry is not None:
                    proc = entry.get("process") if isinstance(entry, dict) else None
                    if proc is None:
                        continue
            exit_code = proc.poll()
            if exit_code is None:
                continue
            final_status = "COMPLETED" if exit_code == 0 else "FAILED"
            if status == "STOP_REQUESTED":
                final_status = "STOPPED" if exit_code == 0 else "FAILED"
            update_workflow_job_row(
                db_path,
                job_id,
                status=final_status,
                ended_at=utc_now_iso(),
                exit_code=int(exit_code),
            )
            workflow_processes.pop(job_id, None)
            continue

        if not is_process_running(pid):
            final_status = "STOPPED" if status == "STOP_REQUESTED" else "EXITED"
            update_workflow_job_row(
                db_path,
                job_id,
                status=final_status,
                ended_at=utc_now_iso(),
            )


def load_workflow_jobs_frame(
    *,
    db_path: str,
    limit: int,
    resolve_postgres_dsn: Callable[[str], str],
    ensure_workflow_jobs_schema: Callable[[str], None],
    connect_state_store: Callable[[str], object],
    coerce_datetime: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> pd.DataFrame:
    if not resolve_postgres_dsn(db_path):
        logger.warning(
            "Skipping workflow job load because no Postgres DSN resolved for db_path=%s",
            db_path,
        )
        return pd.DataFrame()
    ensure_workflow_jobs_schema(db_path)
    try:
        conn = connect_state_store(db_path)
    except Exception:
        logger.warning(
            "Unable to connect to workflow job store for db_path=%s; returning empty frame",
            db_path,
            exc_info=True,
        )
        return pd.DataFrame()
    try:
        try:
            df = pd.read_sql_query(
                WORKFLOW_JOBS_QUERY,
                conn,
                params=[int(max(1, limit))],
            )
        except Exception:
            logger.warning(
                "Unable to query workflow jobs for db_path=%s; returning empty frame",
                db_path,
                exc_info=True,
            )
            return pd.DataFrame()
    finally:
        conn.close()

    for column in WORKFLOW_JOB_DATETIME_COLUMNS:
        df = coerce_datetime(df, column)
    return df


def build_backtest_job_launch_spec(
    *,
    runner_data_source: str,
    market_db_path: str,
    market_exchange: str,
    runner_env_overrides: dict[str, str],
    strategy_name: str,
    backtest_run_id: str,
    strategy_params_path: str,
) -> dict[str, object]:
    return {
        "workflow": "backtest",
        "command": build_lq_command(
            "backtest",
            "--data-source",
            runner_data_source,
            "--market-db-path",
            market_db_path,
            "--market-exchange",
            market_exchange,
            "--run-id",
            backtest_run_id,
        ),
        "env_overrides": dict(runner_env_overrides),
        "requested_mode": "backtest",
        "strategy": strategy_name,
        "run_id": backtest_run_id,
        "metadata": {"strategy_params_path": strategy_params_path},
        "stop_file": None,
    }


def build_optimize_job_launch_spec(
    *,
    optimize_folds: int,
    optimize_trials: int,
    optimize_workers: int,
    runner_data_source: str,
    market_db_path: str,
    market_exchange: str,
    persist_best_params: bool,
    runner_env_overrides: dict[str, str],
    strategy_name: str,
    optimize_run_id: str,
) -> dict[str, object]:
    optimize_args: list[str] = [
        "--folds",
        str(int(optimize_folds)),
        "--n-trials",
        str(int(optimize_trials)),
        "--max-workers",
        str(int(optimize_workers)),
        "--data-source",
        runner_data_source,
        "--market-db-path",
        market_db_path,
        "--market-exchange",
        market_exchange,
        "--run-id",
        optimize_run_id,
    ]
    if persist_best_params:
        optimize_args.append("--save-best-params")
    return {
        "workflow": "optimize",
        "command": build_lq_command("optimize", *optimize_args),
        "env_overrides": dict(runner_env_overrides),
        "requested_mode": "optimize",
        "strategy": strategy_name,
        "run_id": optimize_run_id,
        "metadata": {
            "folds": int(optimize_folds),
            "n_trials": int(optimize_trials),
            "max_workers": int(optimize_workers),
        },
        "stop_file": None,
    }


def build_live_job_launch_spec(
    *,
    runner_env_overrides: dict[str, str],
    live_mode: str,
    market_exchange: str,
    runner_leverage: int,
    live_runner_kind: str,
    live_strategy_name: str,
    live_run_id: str,
    stop_file: str,
) -> dict[str, object]:
    transport = "ws" if "WebSocket" in live_runner_kind else "poll"
    live_args: list[str] = [
        "--transport",
        transport,
        "--strategy",
        live_strategy_name,
        "--run-id",
        live_run_id,
        "--stop-file",
        stop_file,
    ]
    live_env = dict(runner_env_overrides)
    live_env["LQ__LIVE__MODE"] = str(live_mode)
    live_env["LQ__LIVE__EXCHANGE__NAME"] = str(market_exchange).lower()
    live_env["LQ__LIVE__EXCHANGE__LEVERAGE"] = str(int(runner_leverage))
    if live_mode == "real":
        live_args.append("--enable-live-real")
        live_env["LUMINA_ENABLE_LIVE_REAL"] = "true"
    return {
        "workflow": "live_ws" if transport == "ws" else "live",
        "command": build_lq_command("live", *live_args),
        "env_overrides": live_env,
        "requested_mode": live_mode,
        "strategy": live_strategy_name,
        "run_id": live_run_id,
        "stop_file": stop_file,
        "metadata": {
            "runner_kind": live_runner_kind,
            "transport": transport,
        },
    }
