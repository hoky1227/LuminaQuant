from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from importlib import import_module

Handler = Callable[[list[str] | None], int]


def _load_handler(module_name: str, attr: str = "main") -> Handler:
    module = import_module(f"lumina_quant.cli.{module_name}")
    handler = getattr(module, attr)
    return handler


def _run_handler(handler: Handler, argv: list[str]) -> int:
    try:
        return int(handler(argv))
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return int(code)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    commands = {
        "backtest": "backtest",
        "optimize": "optimize",
        "live": "live",
        "dashboard": "dashboard",
        "data": "data",
        "exact-window": "exact_window",
    }

    parser = argparse.ArgumentParser(
        prog="lq",
        description="LuminaQuant command-line interface.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(commands.keys()))
    parser.add_argument("command_args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    handler = _load_handler(commands[parsed.command])
    return _run_handler(handler, list(parsed.command_args or []))


if __name__ == "__main__":
    raise SystemExit(main())
