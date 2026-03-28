from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_saved_incumbent_portfolio_continuity.py"
SPEC = importlib.util.spec_from_file_location("validate_saved_incumbent_portfolio_continuity", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio_continuity module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_latest_common_complete_time_accepts_refresh_symbol_aliases() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "XAU/USD", "after_ohlcv_max_utc": "2026-03-19T09:30:29Z"},
        ],
        "feature_results": [
            {"symbol": "XAU/USD", "last_timestamp_utc": "2026-03-19T09:00:00Z"},
        ],
    }

    anchored_end, evidence = MODULE._latest_common_complete_time(
        refresh_payload=refresh_payload,
        required_pairs=[("XAU/USDT", "1m")],
        feature_symbols=["XAU/USDT"],
    )

    assert MODULE.iso_utc(anchored_end) == "2026-03-19T09:00:00Z"
    assert evidence[0]["symbol"] == "XAU/USDT"
