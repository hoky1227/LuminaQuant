from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
NEXT_DASHBOARD_DIR = REPO_ROOT / "apps" / "dashboard_web"
NEXT_DASHBOARD_ENTRY = "uv run lq dashboard --run"

RETIREMENT_MESSAGE = f"""\
LuminaQuant Dashboard Runtime Retired

The legacy dashboard runtime has been retired.

Use the Next.js dashboard instead:
  {NEXT_DASHBOARD_ENTRY}

Working directory:
  {NEXT_DASHBOARD_DIR}

If you need a readiness check without starting the dev server, run:
  uv run lq dashboard --print-contract
  cd {NEXT_DASHBOARD_DIR} && npm run build
"""


def main() -> int:
    print(RETIREMENT_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
