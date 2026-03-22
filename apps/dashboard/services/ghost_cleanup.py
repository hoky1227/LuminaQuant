"""Dashboard ghost-run cleanup helpers."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_ghost_cleanup_command(
    *,
    python_executable: str,
    dsn: str,
    stale_sec: int,
    startup_grace_sec: int,
    close_status: str,
    force_kill_stop_requested_after_sec: int,
    apply_changes: bool,
) -> list[str]:
    command = [
        str(python_executable),
        str(PROJECT_ROOT / "scripts" / "cleanup_ghost_runs.py"),
        "--dsn",
        str(dsn),
        "--stale-sec",
        str(int(stale_sec)),
        "--startup-grace-sec",
        str(int(startup_grace_sec)),
        "--close-status",
        str(close_status),
        "--force-kill-stop-requested-after-sec",
        str(int(force_kill_stop_requested_after_sec)),
    ]
    if apply_changes:
        command.append("--apply")
    return command


def parse_ghost_cleanup_output(
    *,
    returncode: int,
    command: Sequence[str],
    stdout_text: str,
    stderr_text: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    payload = None
    stripped_stdout = str(stdout_text or "").strip()
    stripped_stderr = str(stderr_text or "").strip()
    if stripped_stdout:
        try:
            payload = json.loads(stripped_stdout)
        except json.JSONDecodeError:
            payload = None
    output = stripped_stdout
    if stripped_stderr:
        output = (output + "\n" + stripped_stderr).strip()
    return {
        "ok": int(returncode) == 0,
        "returncode": int(returncode),
        "elapsed_sec": float(elapsed_sec),
        "command": [str(part) for part in command],
        "output": output,
        "payload": payload,
    }


def run_ghost_cleanup_script(
    *,
    python_executable: str,
    dsn: str,
    stale_sec: int,
    startup_grace_sec: int,
    close_status: str,
    force_kill_stop_requested_after_sec: int,
    apply_changes: bool,
) -> dict[str, Any]:
    command = build_ghost_cleanup_command(
        python_executable=python_executable,
        dsn=dsn,
        stale_sec=stale_sec,
        startup_grace_sec=startup_grace_sec,
        close_status=close_status,
        force_kill_stop_requested_after_sec=force_kill_stop_requested_after_sec,
        apply_changes=apply_changes,
    )
    started = time.time()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PROJECT_ROOT),
    )
    return parse_ghost_cleanup_output(
        returncode=completed.returncode,
        command=command,
        stdout_text=completed.stdout or "",
        stderr_text=completed.stderr or "",
        elapsed_sec=time.time() - started,
    )
