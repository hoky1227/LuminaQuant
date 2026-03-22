from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_APP_PATH = REPO_ROOT / "apps" / "dashboard" / "app.py"


def _dashboard_pythonpath() -> str:
    entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    existing = str(os.environ.get("PYTHONPATH", "") or "").strip()
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _dashboard_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _dashboard_pythonpath()
    return env


def build_dashboard_command(path: str | Path | None = None) -> list[str]:
    target = Path(path).resolve() if path is not None else DASHBOARD_APP_PATH.resolve()
    return ["streamlit", "run", str(target)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or print dashboard launch command.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch streamlit dashboard instead of printing the command.",
    )
    parser.add_argument(
        "--path",
        default=str(DASHBOARD_APP_PATH),
        help="Dashboard app path (default: apps/dashboard/app.py).",
    )
    args = parser.parse_args(argv)

    command = build_dashboard_command(args.path)
    if args.run:
        return int(subprocess.call(command, env=_dashboard_env()))
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
