from __future__ import annotations

from pathlib import Path


def test_dashboard_modules_do_not_mutate_sys_path() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "src" / "lumina_quant" / "dashboard" / "retired_stub.py",
        root / "src" / "lumina_quant" / "dashboard" / "exact_window_bundle.py",
        root / "src" / "lumina_quant" / "dashboard" / "state_store_service.py",
    ]

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in source, path.as_posix()
