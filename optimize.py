from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_quant.cli.optimize import main as _main


def main(argv: list[str] | None = None) -> int:
    return int(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
