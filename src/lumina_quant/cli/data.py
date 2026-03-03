from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Market-data helper commands.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["collect", "materialize", "compact"],
        help="Optional helper command.",
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    print(f"Data command '{args.command}' is available via scripts/ops (planned CLI wiring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
