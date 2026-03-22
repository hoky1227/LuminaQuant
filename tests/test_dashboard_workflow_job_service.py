from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "workflow_jobs.py"
SPEC = importlib.util.spec_from_file_location("dashboard_workflow_jobs", MODULE_PATH)
workflow_jobs = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(workflow_jobs)


def test_build_runtime_env_overrides_serializes_runner_payloads() -> None:
    env = workflow_jobs.build_runtime_env_overrides(
        initial_capital=12_500.5,
        leverage=3,
        timeframe="5m",
        symbols=("BTC/USDT", "ETH/USDT"),
        strategy_name="BreakoutStrategy",
        optuna_config={"n_trials": 25},
        grid_config={"window": [10, 20]},
    )

    assert env == {
        "LQ__TRADING__INITIAL_CAPITAL": "12500.5",
        "LQ__BACKTEST__LEVERAGE": "3",
        "LQ__TRADING__TIMEFRAME": "5m",
        "LQ__TRADING__SYMBOLS": "[\"BTC/USDT\", \"ETH/USDT\"]",
        "LQ__OPTIMIZATION__STRATEGY": "BreakoutStrategy",
        "LQ__OPTIMIZATION__OPTUNA": "{\"n_trials\": 25}",
        "LQ__OPTIMIZATION__GRID": "{\"window\": [10, 20]}",
    }


def test_launch_managed_job_records_process_and_row(tmp_path: Path, monkeypatch) -> None:
    launched: dict[str, object] = {}
    inserted: dict[str, object] = {}
    session_state: dict[str, object] = {}

    class _DummyProcess:
        pid = 43210

    def _fake_popen(command, *, stdout, stderr, text, env, cwd):
        launched.update(
            {
                "command": list(command),
                "stderr": stderr,
                "text": text,
                "env": dict(env),
                "cwd": cwd,
                "log_path": stdout.name,
            }
        )
        return _DummyProcess()

    monkeypatch.setattr(workflow_jobs.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(workflow_jobs.uuid, "uuid4", lambda: "job-123")
    monkeypatch.setenv("EXISTING_FLAG", "kept")

    job_id = workflow_jobs.launch_managed_job(
        db_path="postgres://lumina",
        workflow="backtest",
        command=("uv", "run", "lq", "backtest"),
        env_overrides={"LQ__TRADING__TIMEFRAME": "15m"},
        workflow_log_dir=str(tmp_path / "logs"),
        session_state=session_state,
        insert_workflow_job_row=lambda db_path, row: inserted.update({"db_path": db_path, "row": row}),
        utc_now_iso=lambda: "2026-03-22T00:00:00+00:00",
        requested_mode="backtest",
        strategy="RsiStrategy",
        run_id="run-123",
        stop_file=None,
        metadata={"source": "dashboard"},
        cwd="/repo/root",
    )

    assert job_id == "job-123"
    assert launched["command"] == ["uv", "run", "lq", "backtest"]
    assert launched["cwd"] == "/repo/root"
    assert launched["env"]["EXISTING_FLAG"] == "kept"
    assert launched["env"]["LQ__TRADING__TIMEFRAME"] == "15m"
    assert Path(launched["log_path"]).is_file()

    assert inserted["db_path"] == "postgres://lumina"
    assert inserted["row"]["job_id"] == "job-123"
    assert inserted["row"]["workflow"] == "backtest"
    assert inserted["row"]["command_json"] == json.dumps(
        ["uv", "run", "lq", "backtest"],
        ensure_ascii=True,
    )
    assert inserted["row"]["env_json"] == json.dumps(
        {"LQ__TRADING__TIMEFRAME": "15m"},
        ensure_ascii=True,
    )
    assert inserted["row"]["metadata_json"] == json.dumps({"source": "dashboard"}, ensure_ascii=False)
    assert inserted["row"]["pid"] == 43210

    assert session_state["workflow_processes"]["job-123"]["process"].pid == 43210
    assert session_state["workflow_processes"]["job-123"]["log_path"] == launched["log_path"]


def test_launch_managed_job_defaults_cwd_to_project_root(tmp_path: Path, monkeypatch) -> None:
    launched: dict[str, object] = {}
    session_state: dict[str, object] = {}

    class _DummyProcess:
        pid = 54321

    def _fake_popen(command, *, stdout, stderr, text, env, cwd):
        launched.update(
            {
                "command": list(command),
                "stderr": stderr,
                "text": text,
                "env": dict(env),
                "cwd": cwd,
                "log_path": stdout.name,
            }
        )
        return _DummyProcess()

    monkeypatch.setattr(workflow_jobs.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(workflow_jobs.uuid, "uuid4", lambda: "job-456")

    job_id = workflow_jobs.launch_managed_job(
        db_path="postgres://lumina",
        workflow="optimize",
        command=("uv", "run", "lq", "optimize"),
        env_overrides={},
        workflow_log_dir=str(tmp_path / "logs"),
        session_state=session_state,
        insert_workflow_job_row=lambda *_args, **_kwargs: None,
        utc_now_iso=lambda: "2026-03-22T00:00:00+00:00",
        cwd=None,
    )

    assert job_id == "job-456"
    assert launched["cwd"] == str(workflow_jobs.PROJECT_ROOT)
    assert Path(launched["log_path"]).is_file()


def test_build_stop_file_path_defaults_under_project_var_control(monkeypatch) -> None:
    monkeypatch.setattr(workflow_jobs, "WORKFLOW_CONTROL_DIR", Path("/tmp/lumina-root/var/dashboard/control"))

    stop_file = workflow_jobs.build_stop_file_path("job-789")

    assert stop_file == "/tmp/lumina-root/var/dashboard/control/job-789.stop"


def test_request_job_stop_writes_timestamp(tmp_path: Path) -> None:
    stop_file = tmp_path / "control" / "run.stop"

    assert workflow_jobs.request_job_stop(str(stop_file), timestamp="2026-03-22T00:00:00Z") is True
    assert stop_file.read_text(encoding="utf-8") == "2026-03-22T00:00:00Z"


def test_refresh_workflow_jobs_updates_finished_process_and_exited_pid() -> None:
    updated: list[tuple[str, str, dict[str, object]]] = []
    session_state = {
        "workflow_processes": {
            "job-1": {"process": type("_Proc", (), {"poll": lambda self: 0})()},
        }
    }

    class _Conn:
        def execute(self, _query: str):
            return self

        def fetchall(self):
            return [
                ("job-1", "RUNNING", 101),
                ("job-2", "STOP_REQUESTED", 202),
            ]

        def close(self):
            return None

    workflow_jobs.refresh_workflow_jobs(
        db_path="postgres://lumina",
        session_state=session_state,
        resolve_postgres_dsn=lambda db_path: db_path,
        ensure_workflow_jobs_schema=lambda _db_path: None,
        connect_state_store=lambda _db_path: _Conn(),
        is_process_running=lambda pid: int(pid) == 101,
        update_workflow_job_row=lambda db_path, job_id, **fields: updated.append((db_path, job_id, fields)),
        utc_now_iso=lambda: "2026-03-22T00:00:00Z",
    )

    assert session_state["workflow_processes"] == {}
    assert updated == [
        (
            "postgres://lumina",
            "job-1",
            {"status": "COMPLETED", "ended_at": "2026-03-22T00:00:00Z", "exit_code": 0},
        ),
        (
            "postgres://lumina",
            "job-2",
            {"status": "STOPPED", "ended_at": "2026-03-22T00:00:00Z"},
        ),
    ]


def test_load_workflow_jobs_frame_returns_query_results_with_datetime_coercion(monkeypatch) -> None:
    called: dict[str, object] = {}
    frame = pd.DataFrame(
        [
            {
                "job_id": "job-1",
                "workflow": "backtest",
                "status": "RUNNING",
                "requested_mode": "backtest",
                "strategy": "RsiStrategy",
                "command_json": "[]",
                "env_json": "{}",
                "pid": 123,
                "run_id": "run-1",
                "started_at": "2026-03-22T00:00:00Z",
                "ended_at": None,
                "exit_code": None,
                "log_path": "/tmp/job.log",
                "stop_file": None,
                "metadata_json": "{}",
                "last_updated": "2026-03-22T00:00:01Z",
            }
        ]
    )

    def _fake_read_sql_query(query, conn, params):
        called["query"] = query
        called["conn"] = conn
        called["params"] = list(params)
        return frame.copy()

    class _Conn:
        closed = False

        def close(self):
            self.closed = True

    conn = _Conn()
    monkeypatch.setattr(workflow_jobs.pd, "read_sql_query", _fake_read_sql_query)

    result = workflow_jobs.load_workflow_jobs_frame(
        db_path="postgres://lumina",
        limit=25,
        resolve_postgres_dsn=lambda db_path: db_path,
        ensure_workflow_jobs_schema=lambda _db_path: called.setdefault("schema", True),
        connect_state_store=lambda _db_path: conn,
        coerce_datetime=lambda df, col: df.assign(**{col: f"coerced:{col}"}),
    )

    assert called["query"] == workflow_jobs.WORKFLOW_JOBS_QUERY
    assert called["params"] == [25]
    assert conn.closed is True
    assert list(result["started_at"]) == ["coerced:started_at"]
    assert list(result["ended_at"]) == ["coerced:ended_at"]
    assert list(result["last_updated"]) == ["coerced:last_updated"]


def test_load_workflow_jobs_frame_returns_empty_when_connection_or_query_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow_jobs.pd,
        "read_sql_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("query failed")),
    )

    class _Conn:
        closed = False

        def close(self):
            self.closed = True

    conn = _Conn()
    query_result = workflow_jobs.load_workflow_jobs_frame(
        db_path="postgres://lumina",
        limit=10,
        resolve_postgres_dsn=lambda db_path: db_path,
        ensure_workflow_jobs_schema=lambda _db_path: None,
        connect_state_store=lambda _db_path: conn,
        coerce_datetime=lambda df, _col: df,
    )
    assert query_result.empty
    assert conn.closed is True

    connect_result = workflow_jobs.load_workflow_jobs_frame(
        db_path="postgres://lumina",
        limit=10,
        resolve_postgres_dsn=lambda db_path: db_path,
        ensure_workflow_jobs_schema=lambda _db_path: None,
        connect_state_store=lambda _db_path: (_ for _ in ()).throw(RuntimeError("connect failed")),
        coerce_datetime=lambda df, _col: df,
    )
    assert connect_result.empty
