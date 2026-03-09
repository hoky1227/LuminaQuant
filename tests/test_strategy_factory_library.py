from __future__ import annotations

from lumina_quant.strategies.factory_candidate_set import (
    build_candidate_set,
    summarize_candidate_set,
)
from lumina_quant.strategies.pair_spread_zscore import (
    PairSpreadZScoreStrategy,
    bounded_pair_retune_params,
)
from lumina_quant.strategies.registry import get_strategy_names
from lumina_quant.strategy_factory import (
    build_binance_futures_candidates,
    build_single_asset_portfolio_sets,
    candidate_identity,
    select_diversified_shortlist,
)


def test_factory_candidate_set_is_large_and_diverse():
    candidates = build_candidate_set(
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
        timeframes=["1s", "1m", "1h"],
    )
    summary = summarize_candidate_set(candidates)

    assert len(candidates) >= 100
    assert "RegimeBreakoutCandidateStrategy" in summary["strategies"]
    assert "VolatilityCompressionReversionStrategy" in summary["strategies"]
    assert "trend_breakout" in summary["families"]
    assert "mean_reversion" in summary["families"]


def test_strategy_factory_library_builds_candidates_and_shortlist():
    rows = build_binance_futures_candidates(
        timeframes=["1m"],
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
    )
    assert len(rows) > 20

    mock_candidates = []
    for idx, item in enumerate(rows[:40]):
        as_dict = item.to_dict()
        as_dict["name"] = f"{as_dict['name']}_{idx}"
        as_dict["strategy_timeframe"] = as_dict["timeframe"]
        as_dict["hurdle_fields"] = {"oos": {"pass": True, "score": float(100 - idx)}}
        as_dict["identity"] = candidate_identity(as_dict)
        mock_candidates.append(as_dict)

    shortlist = select_diversified_shortlist(
        mock_candidates,
        mode="oos",
        max_total=12,
        max_per_family=6,
        max_per_timeframe=6,
    )
    assert 1 <= len(shortlist) <= 12
    assert all("shortlist_score" in row for row in shortlist)


def test_mean_reversion_candidate_builder_uses_bounded_5m_15m_slice_only():
    rows = build_binance_futures_candidates(
        timeframes=["1m", "5m", "15m"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "TRX/USDT"],
    )
    volcomp_rows = [
        row
        for row in rows
        if row.strategy_class == "VolCompressionVWAPReversionStrategy"
    ]
    leadlag_rows = [
        row
        for row in rows
        if row.strategy_class == "LeadLagSpilloverStrategy"
    ]

    assert {row.timeframe for row in volcomp_rows} == {"5m", "15m"}
    assert {row.timeframe for row in leadlag_rows} == {"5m", "15m"}
    assert all("volcomp_vwap_rev_guarded_" in row.name for row in volcomp_rows)
    assert all(float(row.params["entry_z"]) >= 2.0 for row in volcomp_rows)
    assert all(float(row.params["compression_vol_ratio"]) <= 0.78 for row in volcomp_rows)
    assert all(int(row.params["max_hold_bars"]) <= 36 for row in volcomp_rows)
    assert all(row.params["allow_short"] is False for row in volcomp_rows)


def test_pair_candidate_builder_prunes_5m_and_focuses_15m_pairs():
    rows = build_binance_futures_candidates(
        timeframes=["5m", "15m"],
        symbols=[
            "BTC/USDT",
            "BNB/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
            "XPT/USDT",
            "XPD/USDT",
        ],
    )
    pair_rows = [
        row
        for row in rows
        if row.strategy_class == "PairSpreadZScoreStrategy"
    ]
    pair_set = {tuple(row.symbols) for row in pair_rows}

    assert {row.timeframe for row in pair_rows} == {"15m"}
    assert pair_set == {("BTC/USDT", "TRX/USDT"), ("BNB/USDT", "TRX/USDT")}
    assert all(float(row.params["entry_z"]) >= 2.6 for row in pair_rows)
    assert all(float(row.params["min_correlation"]) >= 0.25 for row in pair_rows)
    assert all(int(row.params["cooldown_bars"]) >= 10 for row in pair_rows)
    assert all(float(row.params["reentry_z_buffer"]) >= 0.35 for row in pair_rows)


def test_pair_candidate_builder_includes_4h_and_1d_rows():
    rows = build_binance_futures_candidates(
        timeframes=["4h", "1d"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
            "XPT/USDT",
            "XPD/USDT",
        ],
    )
    pair_rows = [
        row
        for row in rows
        if row.strategy_class == "PairSpreadZScoreStrategy"
    ]
    pair_timeframes = {row.timeframe for row in pair_rows}
    pair_set = {(row.timeframe, tuple(row.symbols)) for row in pair_rows}

    assert pair_timeframes == {"4h", "1d"}
    assert ("4h", ("BTC/USDT", "ETH/USDT")) in pair_set
    assert ("4h", ("ETH/USDT", "SOL/USDT")) in pair_set
    assert ("4h", ("XAU/USDT", "XAG/USDT")) in pair_set
    assert ("4h", ("BTC/USDT", "XAU/USDT")) in pair_set
    assert ("4h", ("ETH/USDT", "XAU/USDT")) in pair_set
    assert ("4h", ("BNB/USDT", "XAU/USDT")) in pair_set
    assert ("4h", ("BTC/USDT", "XAG/USDT")) in pair_set
    assert ("1d", ("BTC/USDT", "BNB/USDT")) in pair_set
    assert ("1d", ("BTC/USDT", "TRX/USDT")) in pair_set
    assert ("1d", ("XPT/USDT", "XPD/USDT")) in pair_set
    assert ("1d", ("BTC/USDT", "XAU/USDT")) in pair_set
    assert ("1d", ("ETH/USDT", "XAU/USDT")) in pair_set
    assert ("1d", ("BNB/USDT", "XAU/USDT")) in pair_set
    high_tf_rows = [row for row in pair_rows if row.timeframe == "4h"]
    daily_rows = [row for row in pair_rows if row.timeframe == "1d"]

    assert all(float(row.params["min_correlation"]) >= 0.03 for row in high_tf_rows)
    assert all(int(row.params["cooldown_bars"]) >= 3 for row in high_tf_rows)
    assert all(float(row.params["reentry_z_buffer"]) >= 0.12 for row in high_tf_rows)
    assert all(int(row.params["max_hold_bars"]) <= 144 for row in high_tf_rows)
    assert any(float(row.params["entry_z"]) < 1.8 for row in high_tf_rows)
    assert len({int(row.params["lookback_window"]) for row in high_tf_rows}) >= 2

    assert all(float(row.params["min_correlation"]) >= 0.0 for row in daily_rows)
    assert all(int(row.params["cooldown_bars"]) >= 1 for row in daily_rows)
    assert all(float(row.params["reentry_z_buffer"]) >= 0.08 for row in daily_rows)
    assert all(int(row.params["max_hold_bars"]) <= 36 for row in daily_rows)
    assert any(float(row.params["entry_z"]) < 1.8 for row in daily_rows)
    assert len({int(row.params["lookback_window"]) for row in daily_rows}) >= 2


def test_composite_trend_candidate_builder_uses_explicit_30m_1h_stability_slice():
    rows = build_binance_futures_candidates(
        timeframes=["30m", "1h", "4h", "1d"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XAU/USDT",
            "XAG/USDT",
        ],
    )
    trend_rows = [row for row in rows if row.strategy_class == "CompositeTrendStrategy"]

    assert len(trend_rows) == 6
    assert {row.timeframe for row in trend_rows} == {"30m", "1h"}
    assert all("composite_trend_stable_" in row.name for row in trend_rows)
    assert all("exit_score_cross" in row.params for row in trend_rows)
    assert all("max_signal_strength" in row.params for row in trend_rows)

    one_hour_rows = [row for row in trend_rows if row.timeframe == "1h"]
    assert one_hour_rows
    assert all(row.params["allow_short"] is False for row in one_hour_rows)
    assert all(float(row.params["crowding_block_threshold"]) <= 0.70 for row in one_hour_rows)


def test_pair_spread_default_pair_prefers_xpt_xpd_when_available():
    left, right = PairSpreadZScoreStrategy._resolve_default_pair(
        ["XPT/USDT", "XPD/USDT", "BTC/USDT"]
    )
    assert (left, right) == ("XPT/USDT", "XPD/USDT")


def test_bounded_pair_retune_params_raise_15m_turnover_guards():
    default_params = bounded_pair_retune_params("1h")
    focused_params = bounded_pair_retune_params("15m")

    assert focused_params["lookback_window"] > default_params["lookback_window"]
    assert focused_params["min_correlation"] > default_params["min_correlation"]
    assert focused_params["cooldown_bars"] > default_params["cooldown_bars"]
    assert focused_params["reentry_z_buffer"] > default_params["reentry_z_buffer"]


def test_bounded_pair_retune_params_shorten_high_timeframe_holds_for_oos_realization():
    default_params = bounded_pair_retune_params("1h")
    four_hour = bounded_pair_retune_params("4h")
    daily = bounded_pair_retune_params("1d")

    assert four_hour["lookback_window"] < default_params["lookback_window"]
    assert four_hour["hedge_window"] < default_params["hedge_window"]
    assert four_hour["max_hold_bars"] < default_params["max_hold_bars"]
    assert four_hour["min_correlation"] <= default_params["min_correlation"]

    assert daily["lookback_window"] < four_hour["lookback_window"]
    assert daily["hedge_window"] < four_hour["hedge_window"]
    assert daily["max_hold_bars"] < four_hour["max_hold_bars"]
    assert daily["cooldown_bars"] < four_hour["cooldown_bars"]


def test_pair_candidate_builder_adds_mixed_asset_pairs_only_when_symbols_present():
    rows = build_binance_futures_candidates(
        timeframes=["4h"],
        symbols=["BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT"],
    )
    pair_rows = [row for row in rows if row.strategy_class == "PairSpreadZScoreStrategy"]
    pair_set = {tuple(row.symbols) for row in pair_rows}

    assert ("BTC/USDT", "XAU/USDT") in pair_set
    assert ("ETH/USDT", "XAU/USDT") in pair_set
    assert ("XAU/USDT", "XAG/USDT") in pair_set
    assert ("BTC/USDT", "XAG/USDT") in pair_set


def test_registry_exposes_new_candidate_strategies():
    names = get_strategy_names()
    assert "RegimeBreakoutCandidateStrategy" in names
    assert "VolatilityCompressionReversionStrategy" in names


def test_shortlist_filters_weak_single_and_assigns_weights():
    candidates = [
        {
            "name": "rsi_1m_candidate",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 2.0}},
            "oos": {"return": 0.02, "sharpe": 1.0, "mdd": 0.08, "trades": 19},
            "params": {"rsi_period": 14},
        },
        {
            "name": "pair_z_1m_candidate",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 5.5}},
            "oos": {"return": 0.11, "sharpe": 1.4, "mdd": 0.05, "trades": 30},
            "params": {"entry_z": 2.0},
        },
        {
            "name": "topcap_tsmom_1h_candidate",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "hurdle_fields": {"oos": {"pass": True, "score": 4.2}},
            "oos": {"return": 0.09, "sharpe": 1.1, "mdd": 0.06, "trades": 24},
            "params": {"lookback_bars": 16},
        },
    ]

    shortlist = select_diversified_shortlist(
        candidates,
        mode="oos",
        max_total=5,
        max_per_family=5,
        max_per_timeframe=5,
        single_min_score=0.0,
        single_min_return=0.0,
        single_min_sharpe=0.7,
        single_min_trades=20,
        drop_single_without_metrics=True,
        include_weights=True,
    )

    # Weak single strategy is filtered out; mixed strategies remain and get normalized weights.
    assert len(shortlist) == 2
    assert all(float(row.get("portfolio_weight", 0.0)) > 0.0 for row in shortlist)
    total_weight = sum(float(row.get("portfolio_weight", 0.0)) for row in shortlist)
    assert abs(total_weight - 1.0) < 1e-9


def test_build_single_asset_portfolio_sets_from_shortlist():
    shortlist = [
        {
            "name": "rsi_btc",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "shortlist_score": 5.0,
            "oos": {"return": 0.1, "sharpe": 1.2},
        },
        {
            "name": "ma_btc_alt",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT"],
            "shortlist_score": 4.5,
            "oos": {"return": 0.08, "sharpe": 1.0},
        },
        {
            "name": "rsi_eth",
            "strategy_timeframe": "1m",
            "symbols": ["ETH/USDT"],
            "shortlist_score": 4.8,
            "oos": {"return": 0.09, "sharpe": 1.1},
        },
        {
            "name": "pair_btc_eth",
            "strategy_timeframe": "1m",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "shortlist_score": 6.0,
            "oos": {"return": 0.12, "sharpe": 1.5},
        },
    ]
    sets = build_single_asset_portfolio_sets(shortlist, mode="oos", max_per_asset=2, max_sets=4)
    assert len(sets) >= 1
    top = sets[0]
    assert top["member_count"] == 2
    assert abs(sum(float(row.get("portfolio_weight", 0.0)) for row in top["members"]) - 1.0) < 1e-9


def test_strategy_library_rows_include_tags_and_metadata():
    rows = build_binance_futures_candidates(
        timeframes=["5m", "1h", "4h"],
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XAU/USDT", "TRX/USDT"],
    )
    assert rows
    row_dicts = [row.to_dict() for row in rows]
    assert all(isinstance(item["tags"], list) for item in row_dicts)
    assert all(isinstance(item["metadata"], dict) for item in row_dicts)
    assert any("trend-following" in row["tags"] for row in row_dicts)
    assert any("vol_compression" in row["tags"] for row in row_dicts)
    assert any("pair" in row["tags"] for row in row_dicts)
    assert any("carry" in row["tags"] for row in row_dicts)
    assert any("leadlag" in row["tags"] for row in row_dicts)
    assert any(item["metadata"].get("timeframe") is not None for item in row_dicts)
