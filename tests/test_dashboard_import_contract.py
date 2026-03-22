from __future__ import annotations

from pathlib import Path


def test_dashboard_modules_do_not_mutate_sys_path() -> None:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "apps" / "dashboard" / "app.py",
        root / "apps" / "dashboard" / "exact_window_suite.py",
        root / "apps" / "dashboard" / "services" / "execution_dashboard.py",
        root / "apps" / "dashboard" / "services" / "market_dashboard.py",
        root / "apps" / "dashboard" / "services" / "mirror_dashboard.py",
        root / "apps" / "dashboard" / "services" / "overview_dashboard.py",
        root / "apps" / "dashboard" / "services" / "exact_window_panels.py",
        root / "apps" / "dashboard" / "services" / "risk_dashboard.py",
        root / "apps" / "dashboard" / "services" / "exact_window.py",
    ]

    for path in targets:
        source = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in source, path.as_posix()
