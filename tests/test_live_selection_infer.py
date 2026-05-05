from lumina_quant.live_selection import infer_strategy_class_name


def test_infer_strategy_class_name_extended_catalog():
    assert infer_strategy_class_name("topcap_tsmom") == "TopCapTimeSeriesMomentumStrategy"
    assert infer_strategy_class_name("pair_xau_xag") == "PairTradingZScoreStrategy"
    assert infer_strategy_class_name("rolling_breakout_topcap") == "RollingBreakoutStrategy"
    assert infer_strategy_class_name("mean_reversion_std_topcap") == "MeanReversionStdStrategy"
    assert infer_strategy_class_name("vwap_reversion_topcap") == "VwapReversionStrategy"
    assert infer_strategy_class_name("lag_convergence_xau_xag") == "LagConvergenceStrategy"
    assert infer_strategy_class_name("bitcoin_buy_hold") == "BitcoinBuyHoldStrategy"
    assert infer_strategy_class_name("panic_rebound_mr_5m") == "PanicReboundMeanReversionStrategy"
    assert (
        infer_strategy_class_name("session_filtered_pair_carry_1h")
        == "SessionFilteredPairCarryStrategy"
    )
    assert infer_strategy_class_name("profit_moonshot_trend_1h_balanced") == "ProfitMoonshotTrendStrategy"
    assert (
        infer_strategy_class_name("profit_moonshot_breakout_1h_expansion")
        == "ProfitMoonshotBreakoutStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_reversion_1h_shock")
        == "ProfitMoonshotReversionStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_perp_crowding_carry")
        == "PerpCrowdingCarryStrategy"
    )
    assert (
        infer_strategy_class_name("dfse_15m_top5_exhaustion_plus_flow")
        == "DerivativesFlowSqueezeStrategy"
    )
    assert (
        infer_strategy_class_name("derivatives_flow_squeeze_mode")
        == "DerivativesFlowSqueezeStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_derivatives_taker_flow_mode")
        == "DerivativesFlowSqueezeStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_derivatives_taker_flow_sparse_mode")
        == "DerivativesFlowSqueezeStrategy"
    )


def test_infer_strategy_class_name_leadlag_slow_diffusion_mode():
    assert (
        infer_strategy_class_name("profit_moonshot_leadlag_slow_diffusion_mode")
        == "CrossCryptoSlowDiffusionStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_leadlag_slow_diffusion_ensemble_mode")
        == "CrossCryptoSlowDiffusionStrategy"
    )
    assert (
        infer_strategy_class_name("profit_moonshot_leadlag_slow_diffusion_sol_eth_mode")
        == "CrossCryptoSlowDiffusionStrategy"
    )
