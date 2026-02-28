from __future__ import annotations

from pathlib import Path

import yaml

EXPECTED_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TON/USDT",
    "AVAX/USDT",
    "XAU/USDT",
    "XAG/USDT",
]


def test_config_symbols_universe():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    symbols = list(((payload or {}).get("trading") or {}).get("symbols") or [])
    assert symbols == EXPECTED_SYMBOLS

