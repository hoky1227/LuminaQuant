"""Dashboard workflow-job launch helpers."""

from __future__ import annotations

import os

DEFAULT_LQ_COMMAND = ("uv", "run", "lq")


def build_lq_command(subcommand: str, *args: str) -> tuple[str, ...]:
    return (*DEFAULT_LQ_COMMAND, str(subcommand), *tuple(str(arg) for arg in args))


def build_stop_file_path(job_id: str, *, control_dir: str = os.path.join("logs", "control")) -> str:
    os.makedirs(control_dir, exist_ok=True)
    return os.path.join(control_dir, f"{job_id}.stop")


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
