from __future__ import annotations

import os
from pathlib import Path

from lumina_quant.cli import dashboard


def test_build_dashboard_command_uses_absolute_app_path() -> None:
    command = dashboard.build_dashboard_command()

    assert command[:2] == ["streamlit", "run"]
    assert Path(command[2]).is_absolute()
    assert command[2].endswith("apps/dashboard/app.py")


def test_dashboard_env_includes_repo_root_and_src() -> None:
    env = dashboard._dashboard_env()
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)

    assert str(dashboard.REPO_ROOT) in pythonpath_entries
    assert str(dashboard.REPO_ROOT / "src") in pythonpath_entries
