from __future__ import annotations

import json
import os

from lumina_quant.cli import dashboard


def test_build_dashboard_command_uses_absolute_app_path() -> None:
    command = dashboard.build_dashboard_command()

    assert command == ["npm", "run", "dev"]


def test_dashboard_env_includes_repo_root_and_src() -> None:
    env = dashboard._dashboard_env()
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)

    assert str(dashboard.REPO_ROOT) in pythonpath_entries
    assert str(dashboard.REPO_ROOT / "src") in pythonpath_entries


def test_dashboard_env_exposes_launch_mode_and_compatibility_path() -> None:
    contract = dashboard.build_dashboard_contract(compat_path="/api/python/dashboard/overview")
    env = dashboard._dashboard_env(contract)

    assert env["LQ_DASHBOARD_LAUNCH_MODE"] == "next"
    assert env["LQ_DASHBOARD_COMPAT_PATH"] == "/api/python/dashboard/overview"
    assert env["LQ_DASHBOARD_FRONTEND_TARGET"].endswith("apps/dashboard_web")
    assert contract.retired_stub_path.endswith("src/lumina_quant/dashboard/retired_stub.py")


def test_dashboard_main_run_uses_dashboard_web_cwd_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_call(command, *, env, cwd):
        captured["command"] = list(command)
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(dashboard.subprocess, "call", _fake_call)

    exit_code = dashboard.main(["--run"])

    assert exit_code == 0
    assert captured["command"] == ["npm", "run", "dev"]
    assert captured["cwd"] == str(dashboard.DASHBOARD_NEXT_APP_DIR)


def test_dashboard_main_print_contract_emits_json(capsys) -> None:
    exit_code = dashboard.main(["--print-contract"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["launch_mode"] == "next"
    assert payload["compatibility_path"] == "/api/python/dashboard/overview"
    assert payload["frontend_target"].endswith("apps/dashboard_web")
