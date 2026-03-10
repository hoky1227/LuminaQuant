from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.research.write_exact_window_deployment_combo import _build_scenarios


def test_build_scenarios_falls_back_to_experimental_watchlist_without_strict_anchor(tmp_path: Path):
    details_path = tmp_path / "exact_window_candidate_details_latest.json"
    details_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "cand-topcap",
                    "name": "topcap_tsmom_1h_balanced_16_4_0.015",
                    "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                    "family": "cross_sectional",
                    "strategy_timeframe": "1h",
                    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
                    "train": {"return": -0.10, "sharpe": -0.4, "pbo": 0.125, "trade_count": 1200},
                    "val": {"return": 0.018, "sharpe": 1.64, "pbo": 0.25, "trade_count": 90},
                    "oos": {"return": 0.032, "sharpe": 1.46, "pbo": 0.375, "trade_count": 119},
                    "return_streams": {
                        "train": [{"t": 1, "v": 0.0}],
                        "val": [{"t": 2, "v": 0.018}],
                        "oos": [{"t": 3, "v": 0.032}],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    decision = {
        "timeframe_rows": [],
        "source_batches": [
            {
                "details_path": str(details_path),
            }
        ],
    }

    scenarios = _build_scenarios({}, decision, "2026-03-10T12:00:00+00:00")

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["scenario_id"] == "experimental_research_watchlist"
    assert scenario["components"][0]["name"] == "topcap_tsmom_1h_balanced_16_4_0.015"
    assert "research_only" in scenario["components"][0]["risk_flags"]
