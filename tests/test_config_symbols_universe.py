from __future__ import annotations

from pathlib import Path

import yaml

from lumina_quant.configuration.schema import TradingConfig


def _default_trading_symbols() -> list[str]:
    return list(TradingConfig().symbols)


def test_config_symbols_universe_matches_schema_defaults():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    symbols = list(((payload or {}).get("trading") or {}).get("symbols") or [])
    assert symbols == _default_trading_symbols()
