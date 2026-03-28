from __future__ import annotations

from lumina_quant.cli.refresh_data_fast import build_refresh_command, build_refresh_env


def test_build_refresh_command_applies_safe_defaults() -> None:
    command = build_refresh_command(["--symbols", "BTC/USDT,ETH/USDT"])
    tokens = command[2:]

    assert "--symbols" in tokens
    assert "--priority-symbols" in tokens
    assert "--max-workers" in tokens
    assert "--memory-budget-bytes" in tokens
    assert "--soft-rss-bytes" in tokens


def test_build_refresh_command_ignores_passthrough_separator() -> None:
    command = build_refresh_command(["--", "--symbols", "BTC/USDT"])
    tokens = command[2:]

    assert "--" not in tokens
    assert tokens[:2] == ["--symbols", "BTC/USDT"]


def test_build_refresh_env_sets_thread_caps_and_auto_backend() -> None:
    env = build_refresh_env()

    assert env["LQ_RAW_FIRST_BACKEND"] == "auto"
    assert env["POLARS_MAX_THREADS"] == "1"
    assert env["RAYON_NUM_THREADS"] == "1"
