"""Dashboard process-control helpers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def is_process_running(pid: object) -> bool:
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or "").upper()
        return str(pid) in output and "NO TASKS" not in output
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_process(pid: object) -> tuple[bool, str]:
    try:
        pid = int(pid)
    except Exception:
        return False, "invalid pid"
    if pid <= 0:
        return False, "invalid pid"
    if sys.platform.startswith("win"):
        proc = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        ok = proc.returncode == 0
        detail = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return ok, detail.strip()
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        return False, str(exc)
    return True, "terminated"


def tail_text_file(path: str | os.PathLike[str] | None, max_chars: int = 20000) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        max_bytes = max(4096, int(max_chars) * 3)
    except Exception:
        max_bytes = 60000
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-int(max_chars) :]
