from __future__ import annotations

import runpy
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "apps" / "dashboard" / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(APP_PATH), run_name="__main__")
