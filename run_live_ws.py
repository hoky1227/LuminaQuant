from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lumina_quant.cli.live import main as _live_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    if not any(arg == "--transport" or arg.startswith("--transport=") for arg in args):
        args = ["--transport", "ws", *args]
    return int(_live_main(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
