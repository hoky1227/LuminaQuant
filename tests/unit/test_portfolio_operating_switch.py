import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "write_portfolio_operating_switch.py"
)
SPEC = importlib.util.spec_from_file_location("write_portfolio_operating_switch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _base_operating_plan() -> dict:
    return {
        "deployment_modes": {
            "core_mode": {"allocation": {"soft_three_way_regime": 1.0}},
            "balanced_overlay_mode": {
                "allocation": {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2},
                "metrics": {"oos_total_return": 0.02, "oos_sharpe": 1.0, "oos_max_drawdown": 0.01},
            },
            "defensive_overlay_mode": {"allocation": {"soft_three_way_regime": 0.7, "pair_fast_exit": 0.3}},
            "aggressive_realized_mode": {"allocation": {"three_way_regime": 1.0}},
            "hybrid_guarded_mode": {"allocation": {"hybrid_online_portfolio": 1.0}},
        }
    }


def _ts(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def test_load_symbol_volume_signal_classifies_strong_volume(tmp_path: Path) -> None:
    date_dir = tmp_path / "BNBUSDT" / "date=2026-03-07"
    date_dir.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "timestamp_ms": [
                1772841600000,
                1772845200000,
                1772755200000,
                1772668800000,
            ],
            "price": [100.0, 100.0, 100.0, 100.0],
            "quantity": [30.0, 30.0, 10.0, 10.0],
            "agg_trade_id": [1, 2, 3, 4],
            "is_buyer_maker": [True, True, False, False],
        }
    )
    frame.write_parquet(date_dir / "part-0000.parquet")

    signal = MODULE._load_symbol_volume_signal(
        raw_aggtrades_root=tmp_path,
        symbol="BNB/USDT",
        as_of_date=MODULE._parse_as_of_date("2026-03-07T00:00:00+00:00"),
        lookback_days=7,
    )

    assert signal.symbol == "BNB/USDT"
    assert signal.latest_available_date == "2026-03-07"
    assert signal.volume_ratio is not None and signal.volume_ratio > 1.5
    assert signal.comparison_mode == "same_time_of_day"
    assert signal.state == "strong"


def test_load_symbol_volume_signal_uses_same_time_baseline_for_partial_current_day(tmp_path: Path) -> None:
    symbol_root = tmp_path / "BNBUSDT"
    daily_payloads = {
        "2026-03-05": {
            "timestamp_ms": [_ts(2026, 3, 5, 8), _ts(2026, 3, 5, 10), _ts(2026, 3, 5, 22)],
            "price": [100.0, 100.0, 100.0],
            "quantity": [0.3, 0.3, 0.4],
        },
        "2026-03-06": {
            "timestamp_ms": [_ts(2026, 3, 6, 8), _ts(2026, 3, 6, 10), _ts(2026, 3, 6, 22)],
            "price": [100.0, 100.0, 100.0],
            "quantity": [0.3, 0.3, 0.4],
        },
        "2026-03-07": {
            "timestamp_ms": [_ts(2026, 3, 7, 8), _ts(2026, 3, 7, 10)],
            "price": [100.0, 100.0],
            "quantity": [0.35, 0.35],
        },
    }
    for day, payload in daily_payloads.items():
        date_dir = symbol_root / f"date={day}"
        date_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                **payload,
                "agg_trade_id": list(range(1, len(payload["timestamp_ms"]) + 1)),
                "is_buyer_maker": [True] * len(payload["timestamp_ms"]),
            }
        ).write_parquet(date_dir / "part-0000.parquet")

    signal = MODULE._load_symbol_volume_signal(
        raw_aggtrades_root=tmp_path,
        symbol="BNB/USDT",
        as_of_date=MODULE._parse_as_of_date("2026-03-07T10:00:00+00:00"),
        lookback_days=7,
    )

    assert signal.comparison_mode == "same_time_of_day"
    assert signal.volume_ratio is not None and signal.volume_ratio > 1.1
    assert signal.state == "strong"


def test_recommend_operating_mode_prefers_balanced_overlay_for_current_like_state() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.73,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.03,
                "btc_trend_gap_336": -0.01,
                "btc_trend_accel": 0.02,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "breadth_delta": 0.0,
                "basket_vol_ratio": 0.45,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.80,
            "effective_autoresearch_exposure": 0.20,
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "incumbent",
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="strong",
    )

    assert decision.mode == "balanced_overlay_mode"
    assert decision.allocation == {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}


def test_recommend_operating_mode_falls_back_to_core_when_pair_liquidity_is_weak() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.80,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.05,
                "btc_trend_gap_336": -0.03,
                "btc_trend_accel": -0.01,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "breadth_delta": -0.1,
                "basket_vol_ratio": 0.90,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.85,
            "effective_autoresearch_exposure": 0.15,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.01, "oos_sharpe": 0.5},
        },
        hard_current_state={"state": "incumbent", "raw_target_state": "incumbent"},
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="weak",
    )

    assert decision.mode == "core_mode"
    assert decision.allocation == {"soft_three_way_regime": 1.0}


def test_recommend_operating_mode_can_select_aggressive_realized_mode() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "autoresearch",
            "confidence": 0.80,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.04,
                "btc_trend_gap_336": 0.03,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.7,
                "breadth_ma192": 0.6,
                "breadth_delta": 0.1,
                "basket_vol_ratio": 0.95,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.40,
            "effective_autoresearch_exposure": 0.60,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.05, "oos_sharpe": 2.0},
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "autoresearch_55_45",
            "_allocator_health": {"healthy": True, "oos_total_return": 0.04, "oos_sharpe": 1.8},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="strong",
    )

    assert decision.mode == "aggressive_realized_mode"
    assert decision.allocation == {"three_way_regime": 1.0}


def test_recommend_operating_mode_blocks_aggressive_when_hard_allocator_is_unhealthy() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "autoresearch",
            "confidence": 0.90,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.02,
                "btc_trend_gap_336": -0.01,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.4,
                "breadth_ma192": 0.2,
                "breadth_delta": 0.2,
                "basket_vol_ratio": 0.8,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.60,
            "effective_autoresearch_exposure": 0.40,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.01, "oos_sharpe": -1.0},
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "autoresearch_55_45",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.05, "oos_sharpe": -3.0},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="normal",
    )

    assert decision.mode == "balanced_overlay_mode"
    assert decision.allocation == {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}


def test_recommend_operating_mode_uses_risk_off_when_all_active_sleeves_are_unhealthy() -> None:
    plan = _base_operating_plan()
    plan["deployment_modes"]["balanced_overlay_mode"]["metrics"] = {
        "oos_total_return": -0.01,
        "oos_sharpe": -1.0,
        "oos_max_drawdown": 0.02,
    }
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "autoresearch",
            "confidence": 0.90,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.02,
                "btc_trend_gap_336": -0.01,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "breadth_delta": 0.0,
                "basket_vol_ratio": 0.8,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.60,
            "effective_autoresearch_exposure": 0.40,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.01, "oos_sharpe": -1.0},
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "autoresearch_55_45",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.05, "oos_sharpe": -3.0},
        },
        operating_plan_payload=plan,
        pair_liquidity_state="normal",
    )

    assert decision.mode == "risk_off_mode"
    assert decision.allocation == {"cash": 1.0}


def test_balanced_strategy_health_prefers_hybrid_source_metrics() -> None:
    health = MODULE._balanced_strategy_health(
        operating_plan_payload=_base_operating_plan(),
        balanced_strategy_payload={
            "portfolio_metrics": {
                "val": {"total_return": -0.02, "sharpe": -1.0, "max_drawdown": 0.03, "volatility": 0.01},
                "oos": {"total_return": -0.01, "sharpe": -1.0, "max_drawdown": 0.02},
            }
        },
        hybrid_source_metrics={
            "balanced_overlay_80_20": {
                "val": {"total_return": 0.01, "sharpe": 1.0},
                "oos": {"total_return": 0.02, "sharpe": 1.5, "max_drawdown": 0.01},
            }
        },
    )

    assert health["healthy"] is True
    assert health["oos_total_return"] == 0.02
    assert health["oos_sharpe"] == 1.5


def test_recommend_operating_mode_uses_risk_off_even_in_bullish_state_when_all_active_sleeves_are_unhealthy() -> None:
    plan = _base_operating_plan()
    plan["deployment_modes"]["balanced_overlay_mode"]["metrics"] = {
        "oos_total_return": -0.01,
        "oos_sharpe": -1.0,
        "oos_max_drawdown": 0.02,
    }
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.95,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.04,
                "btc_trend_gap_336": 0.03,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 1.0,
                "breadth_ma192": 1.0,
                "breadth_delta": 0.2,
                "basket_vol_ratio": 0.5,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.85,
            "effective_autoresearch_exposure": 0.15,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.02, "oos_sharpe": -2.0},
        },
        hard_current_state={
            "state": "incumbent",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.03, "oos_sharpe": -3.0},
        },
        operating_plan_payload=plan,
        pair_liquidity_state="normal",
    )

    assert decision.mode == "risk_off_mode"
    assert decision.allocation == {"cash": 1.0}


def test_recommend_operating_mode_prefers_hybrid_guarded_when_legacy_sleeves_are_unhealthy_but_hybrid_is_healthy() -> None:
    plan = _base_operating_plan()
    plan["deployment_modes"]["balanced_overlay_mode"]["metrics"] = {
        "oos_total_return": -0.01,
        "oos_sharpe": -1.0,
        "oos_max_drawdown": 0.02,
    }
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.95,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.03,
                "btc_trend_gap_336": 0.02,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.9,
                "breadth_ma192": 0.8,
                "breadth_delta": 0.2,
                "basket_vol_ratio": 0.6,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.85,
            "effective_autoresearch_exposure": 0.15,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.02, "oos_sharpe": -2.0},
        },
        hard_current_state={
            "state": "incumbent",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.03, "oos_sharpe": -3.0},
        },
        operating_plan_payload=plan,
        pair_liquidity_state="normal",
        hybrid_health={
            "healthy": True,
            "recommended_stage": "guarded_candidate",
            "oos_total_return": 0.02,
            "oos_sharpe": 2.0,
            "beats_cash_refreshed": True,
            "pair_cap_respected": True,
            "max_rss_under_8gib": True,
        },
    )

    assert decision.mode == "hybrid_guarded_mode"
    assert decision.allocation == {"hybrid_online_portfolio": 1.0}


def test_build_operating_switch_payload_includes_market_states() -> None:
    payload = MODULE.build_operating_switch_payload(
        market_judgement_payload={
            "_path": "/tmp/market.json",
            "current_judgement": {
                "date": "2026-03-07T00:00:00+00:00",
                "favored_group": "incumbent",
                "confidence": 0.73,
                "feature_snapshot": {
                    "btc_above_ma192": False,
                    "btc_above_ma336": False,
                    "btc_trend_gap_192": -0.03,
                    "btc_trend_gap_336": -0.01,
                    "btc_trend_accel": 0.02,
                    "breadth_ma96": 0.2,
                    "breadth_ma192": 0.2,
                    "breadth_delta": 0.0,
                    "basket_vol_ratio": 0.45,
                },
            },
        },
        soft_allocator_payload={"_path": "/tmp/soft.json", "current_state": {"effective_incumbent_exposure": 0.8, "effective_autoresearch_exposure": 0.2}},
        three_way_allocator_payload={"_path": "/tmp/hard.json", "current_state": {"state": "autoresearch_55_45", "raw_target_state": "incumbent"}},
        operating_plan_payload={"_path": "/tmp/plan.json", **_base_operating_plan()},
        pair_volume_signals=[
            MODULE.SymbolVolumeSignal(
                symbol="BNB/USDT",
                as_of_date="2026-03-07",
                latest_available_date="2026-03-07",
                stale_days=0,
                latest_dollar_volume=1000.0,
                lookback_mean_dollar_volume=800.0,
                volume_ratio=1.25,
                comparison_mode="full_day",
                state="strong",
            )
        ],
        feature_lookback_days=21,
    )

    assert payload["current_market_state"]["trend_state"] == "bearish"
    assert payload["current_market_state"]["volatility_state"] == "calm"
    assert payload["recommended_mode"]["mode"] == "balanced_overlay_mode"
    assert payload["recommended_mode"]["allocation"] == {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}
