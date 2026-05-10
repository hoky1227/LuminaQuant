from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_profit_moonshot_candidate_hybrid.py"
SPEC = importlib.util.spec_from_file_location("run_profit_moonshot_candidate_hybrid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_hybrid_discards_fractional_or_missing_source_leverage() -> None:
    accepted, discarded = MODULE._partition_live_integer_candidate_rows(
        [
            {"name": "integer_ok", "leverage": 5.0, "_candidate_hybrid_source_kind": "liquidation"},
            {"name": "fractional_bad", "leverage": 2.3427334297703024, "_candidate_hybrid_source_kind": "candidate"},
            {"name": "missing_bad", "_candidate_hybrid_source_kind": "candidate"},
        ]
    )

    assert [row["name"] for row in accepted] == ["integer_ok"]
    assert [row["name"] for row in discarded] == ["fractional_bad", "missing_bad"]
    assert {row["reason"] for row in discarded} == {"non_integer_or_missing_live_leverage"}
