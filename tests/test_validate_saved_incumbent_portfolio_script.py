from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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


def test_evaluate_saved_weight_portfolio_separates_weighted_component_summaries() -> None:
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
                "train": [{"datetime": "2026-01-01T00:00:00Z", "v": 0.01}],
                "val": [{"datetime": "2026-02-01T00:00:00Z", "v": 0.02}],
                "oos": [
                    {"datetime": "2026-03-01T00:00:00Z", "v": 0.03},
                    {"datetime": "2026-03-02T00:00:00Z", "v": -0.01},
                ],
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
                "train": [{"datetime": "2026-01-01T00:00:00Z", "v": 0.02}],
                "val": [{"datetime": "2026-02-01T00:00:00Z", "v": 0.01}],
                "oos": [
                    {"datetime": "2026-03-01T00:00:00Z", "v": 0.01},
                    {"datetime": "2026-03-02T00:00:00Z", "v": 0.00},
                ],
            },
            "metadata": {"cost_rate": 0.001},
        },
    ]

    payload = MODULE.evaluate_saved_weight_portfolio(rows)
    oos = payload["portfolio_metrics"]["oos"]
    weighted = payload["weighted_component_summaries"]["oos"]
    assert payload["component_rows"][0]["saved_weight"] == 0.6
    assert len(payload["oos_monthly_returns"]) == 1
    assert oos["total_return"] != 0.0
    assert "turnover" not in oos
    assert weighted["turnover"] > 0.0


def test_latest_common_complete_time_uses_timeframe_complete_buckets() -> None:
    refresh_payload = {
        "ohlcv_results": [
            {"symbol": "BTC/USDT", "after_ohlcv_max_utc": "2026-03-19T09:30:29Z"},
            {"symbol": "ETH/USDT", "after_ohlcv_max_utc": "2026-03-19T09:30:29Z"},
        ],
        "feature_results": [
            {"symbol": "BTC/USDT", "last_timestamp_utc": "2026-03-19T08:59:00Z"},
        ],
    }

    anchored_end, evidence = MODULE._latest_common_complete_time(
        refresh_payload=refresh_payload,
        required_pairs=[("BTC/USDT", "1m"), ("ETH/USDT", "4h")],
        feature_symbols=["BTC/USDT"],
    )

    assert MODULE.iso_utc(anchored_end) == "2026-03-19T04:00:00Z"
    assert any(item["symbol"] == "ETH/USDT" and item["timeframe"] == "4h" for item in evidence)


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


def test_run_strict_research_rejects_synthetic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "run_candidate_research",
        lambda **_kwargs: {"data_sources": {"synthetic": ["BTC/USDT@1h"]}, "candidates": []},
    )

    with pytest.raises(RuntimeError, match="synthetic fallback"):
        MODULE._run_strict_research(
            candidates=[{"candidate_id": "c1", "symbols": ["BTC/USDT"], "strategy_timeframe": "1h"}],
            strategy_timeframes=["1h"],
            symbol_universe=["BTC/USDT"],
            split={
                "train_start": "2025-01-01T00:00:00Z",
                "train_end": "2025-12-31T23:59:59Z",
                "val_start": "2026-01-01T00:00:00Z",
                "val_end": "2026-01-31T23:59:59Z",
                "oos_start": "2026-02-01T00:00:00Z",
                "oos_end": "2026-03-01T00:00:00Z",
            },
            min_bundle_bars=1,
        )



def test_build_latest_anchored_split_trims_from_left_when_latest_anchor_moves_forward() -> None:
    saved_oos_end = MODULE.parse_utc("2026-03-17T23:59:59Z")
    latest_common_end = MODULE.parse_utc("2026-03-19T23:59:59Z")
    assert saved_oos_end is not None and latest_common_end is not None

    shifted = MODULE.build_latest_anchored_split(
        saved_oos_end=saved_oos_end,
        anchored_oos_end=latest_common_end,
    ).as_dict()

    assert shifted["train_start"] == "2025-01-03T00:00:00Z"
    assert shifted["val_start"] == "2026-01-03T00:00:00Z"
    assert shifted["oos_start"] == "2026-02-03T00:00:00Z"
    assert shifted["oos_end"] == "2026-03-19T23:59:59Z"
