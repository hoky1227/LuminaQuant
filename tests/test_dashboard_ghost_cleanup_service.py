from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "ghost_cleanup.py"
SPEC = importlib.util.spec_from_file_location("dashboard_ghost_cleanup", MODULE_PATH)
ghost_cleanup = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(ghost_cleanup)


def test_build_ghost_cleanup_command_uses_dsn_flag() -> None:
    command = ghost_cleanup.build_ghost_cleanup_command(
        python_executable="/usr/bin/python3",
        dsn="postgres://lumina",
        stale_sec=300,
        startup_grace_sec=90,
        close_status="STOPPED",
        force_kill_stop_requested_after_sec=0,
        apply_changes=False,
    )

    assert "--dsn" in command
    assert "--db" not in command
    assert command[0] == "/usr/bin/python3"
    assert Path(command[1]).is_absolute()
    assert command[1].endswith("scripts/cleanup_ghost_runs.py")


def test_parse_ghost_cleanup_output_merges_json_and_stderr() -> None:
    payload = ghost_cleanup.parse_ghost_cleanup_output(
        returncode=0,
        command=["python", "cleanup"],
        stdout_text='{"closed_runs": 2}',
        stderr_text="warning",
        elapsed_sec=1.25,
    )

    assert payload["ok"] is True
    assert payload["payload"] == {"closed_runs": 2}
    assert payload["output"] == '{"closed_runs": 2}\nwarning'


def test_run_ghost_cleanup_script_uses_project_root_as_cwd(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command, *, capture_output, text, check, cwd):
        captured["command"] = list(command)
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        captured["cwd"] = cwd
        return _Completed()

    monkeypatch.setattr(ghost_cleanup.subprocess, "run", _fake_run)

    payload = ghost_cleanup.run_ghost_cleanup_script(
        python_executable="/usr/bin/python3",
        dsn="postgres://lumina",
        stale_sec=300,
        startup_grace_sec=90,
        close_status="STOPPED",
        force_kill_stop_requested_after_sec=0,
        apply_changes=False,
    )

    assert payload["ok"] is True
    assert captured["cwd"] == str(ghost_cleanup.PROJECT_ROOT)
