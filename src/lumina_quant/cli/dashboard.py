from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from lumina_quant.dashboard.bridge import (
    DEFAULT_DASHBOARD_COMPAT_PATH,
    DashboardBridgeContract,
    resolve_dashboard_bridge_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_APP_PATH = REPO_ROOT / "apps" / "dashboard" / "app.py"
DASHBOARD_NEXT_APP_DIR = REPO_ROOT / "apps" / "dashboard_web"


def _dashboard_pythonpath() -> str:
    entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing = str(os.environ.get("PYTHONPATH", "") or "").strip()
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _dashboard_env(contract: DashboardBridgeContract | None = None) -> dict[str, str]:
    resolved_contract = contract or resolve_dashboard_bridge_contract(
        launch_mode="streamlit",
        streamlit_app_path=DASHBOARD_APP_PATH,
        next_app_dir=DASHBOARD_NEXT_APP_DIR,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _dashboard_pythonpath()
    env["LQ_DASHBOARD_LAUNCH_MODE"] = resolved_contract.launch_mode
    env["LQ_DASHBOARD_COMPAT_PATH"] = resolved_contract.compatibility_path
    env["LQ_DASHBOARD_FRONTEND_TARGET"] = resolved_contract.frontend_target
    return env


def build_dashboard_contract(
    *,
    mode: str | None = None,
    path: str | Path | None = None,
    compat_path: str | None = None,
) -> DashboardBridgeContract:
    streamlit_target = Path(path).resolve() if path is not None else DASHBOARD_APP_PATH.resolve()
    return resolve_dashboard_bridge_contract(
        launch_mode=mode,
        streamlit_app_path=streamlit_target,
        next_app_dir=DASHBOARD_NEXT_APP_DIR,
        compatibility_path=compat_path,
    )


def build_dashboard_command(
    path: str | Path | None = None,
    *,
    mode: str | None = None,
    compat_path: str | None = None,
) -> list[str]:
    contract = build_dashboard_contract(mode=mode, path=path, compat_path=compat_path)
    if contract.launch_mode == "next":
        return ["npm", "run", "dev"]
    return ["streamlit", "run", contract.streamlit_app_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or print dashboard launch command.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch the dashboard Python bridge instead of printing the command.",
    )
    parser.add_argument(
        "--path",
        default=str(DASHBOARD_APP_PATH),
        help="Dashboard Streamlit app path (default: apps/dashboard/app.py).",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        help="Dashboard launch mode: auto, streamlit, or next.",
    )
    parser.add_argument(
        "--compat-path",
        default=DEFAULT_DASHBOARD_COMPAT_PATH,
        help="Stable compatibility JSON path for the first migrated dashboard slice.",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="Print the resolved dashboard migration compatibility contract as JSON.",
    )
    args = parser.parse_args(argv)

    contract = build_dashboard_contract(
        mode=args.mode,
        path=args.path,
        compat_path=args.compat_path,
    )
    command = build_dashboard_command(
        path=args.path,
        mode=args.mode,
        compat_path=args.compat_path,
    )

    if args.print_contract:
        print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.run:
        target_cwd = DASHBOARD_NEXT_APP_DIR if contract.launch_mode == "next" else REPO_ROOT
        return int(subprocess.call(command, env=_dashboard_env(contract), cwd=str(target_cwd)))
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
