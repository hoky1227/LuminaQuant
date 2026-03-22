from __future__ import annotations

from pathlib import Path


def _extract_function_block(source: str, func_name: str) -> str:
    marker = f"def {func_name}("
    start = source.find(marker)
    if start < 0:
        raise AssertionError(f"{func_name} not found")
    tail = source[start:]
    next_def = tail.find("\ndef ")
    if next_def < 0:
        return tail
    return tail[:next_def]


def test_dashboard_ghost_cleanup_uses_dsn_flag():
    service_path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "dashboard"
        / "services"
        / "ghost_cleanup.py"
    )
    source = service_path.read_text(encoding="utf-8")
    block = _extract_function_block(source, "build_ghost_cleanup_command")

    assert "--dsn" in block
    assert "--db" not in block
