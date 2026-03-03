from __future__ import annotations

import argparse
import subprocess


DASHBOARD_APP_PATH = "apps/dashboard/app.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or print dashboard launch command.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch streamlit dashboard instead of printing the command.",
    )
    parser.add_argument(
        "--path",
        default=DASHBOARD_APP_PATH,
        help="Dashboard app path (default: apps/dashboard/app.py).",
    )
    args = parser.parse_args(argv)

    command = ["streamlit", "run", args.path]
    if args.run:
        return int(subprocess.call(command))
    print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
