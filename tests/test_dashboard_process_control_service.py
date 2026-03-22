from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "apps" / "dashboard" / "services" / "process_control.py"
SPEC = importlib.util.spec_from_file_location("dashboard_process_control", MODULE_PATH)
process_control = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(process_control)


def test_is_process_running_rejects_invalid_pid_tokens() -> None:
    assert process_control.is_process_running(None) is False
    assert process_control.is_process_running("abc") is False
    assert process_control.is_process_running(-1) is False


def test_terminate_process_rejects_invalid_pid_tokens() -> None:
    assert process_control.terminate_process(None) == (False, "invalid pid")
    assert process_control.terminate_process("abc") == (False, "invalid pid")
    assert process_control.terminate_process(-1) == (False, "invalid pid")


def test_tail_text_file_returns_tail_slice(tmp_path: Path) -> None:
    path = tmp_path / "sample.log"
    path.write_text("0123456789" * 10, encoding="utf-8")

    assert process_control.tail_text_file(path, max_chars=12) == "890123456789"
