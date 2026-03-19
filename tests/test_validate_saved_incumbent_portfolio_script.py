from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_saved_incumbent_portfolio.py"
SPEC = importlib.util.spec_from_file_location("validate_saved_incumbent_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_validation_split_uses_canonical_boundaries() -> None:
    split = MODULE.build_validation_split(MODULE.parse_utc("2026-03-17T23:59:59Z"))
    assert split["train_start"] == "2025-01-01T00:00:00Z"
    assert split["val_start"] == "2026-01-01T00:00:00Z"
    assert split["oos_start"] == "2026-02-01T00:00:00Z"
    assert split["oos_end"] == "2026-03-17T23:59:59Z"


def test_evaluate_saved_weight_portfolio_aggregates_weighted_streams() -> None:
    rows = [
        {
            "name": "a",
            "strategy_class": "A",
            "strategy_timeframe": "1h",
            "candidate_id": "a",
            "_saved_weight": 0.6,
            "oos": {"total_return": 0.1, "sharpe": 1.0, "max_drawdown": 0.05, "turnover": 0.2},
            "train": {"trade_count": 1, "turnover": 0.1, "benchmark_corr": 0.2},
            "val": {"trade_count": 1, "turnover": 0.1, "benchmark_corr": 0.2},
            "return_streams": {
                "train": [{"t": 1_000, "v": 0.01}],
                "val": [{"t": 2_000, "v": 0.02}],
                "oos": [{"t": 3_000, "v": 0.03}, {"t": 4_000, "v": -0.01}],
            },
            "metadata": {"cost_rate": 0.0005},
        },
        {
            "name": "b",
            "strategy_class": "B",
            "strategy_timeframe": "30m",
            "candidate_id": "b",
            "_saved_weight": 0.4,
            "oos": {"total_return": 0.05, "sharpe": 0.5, "max_drawdown": 0.02, "turnover": 0.1},
            "train": {"trade_count": 2, "turnover": 0.2, "benchmark_corr": 0.1},
            "val": {"trade_count": 2, "turnover": 0.2, "benchmark_corr": 0.1},
            "return_streams": {
                "train": [{"t": 1_000, "v": 0.02}],
                "val": [{"t": 2_000, "v": 0.01}],
                "oos": [{"t": 3_000, "v": 0.01}, {"t": 4_000, "v": 0.00}],
            },
            "metadata": {"cost_rate": 0.001},
        },
    ]

    payload = MODULE.evaluate_saved_weight_portfolio(rows)
    oos = payload["portfolio_metrics"]["oos"]
    assert payload["component_rows"][0]["saved_weight"] == 0.6
    assert len(payload["oos_monthly_returns"]) == 1
    assert oos["total_return"] != 0.0
    assert oos["turnover"] > 0.0


def test_common_validation_end_uses_minimum_refreshed_common_coverage() -> None:
    refresh_payload = {
        "collection_cutoff_utc": "2026-03-19T09:30:29Z",
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-03-19T09:30:29Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-03-19T09:30:10Z"},
        ],
        "feature_results": [
            {"symbol": "BTC/USDT", "last_timestamp_utc": "2026-03-19T09:30:00Z"},
        ],
    }

    end = MODULE._common_validation_end(refresh_payload=refresh_payload, feature_symbols=["BTC/USDT"])
    assert MODULE.iso_utc(end) == "2026-03-19T09:30:00Z"
