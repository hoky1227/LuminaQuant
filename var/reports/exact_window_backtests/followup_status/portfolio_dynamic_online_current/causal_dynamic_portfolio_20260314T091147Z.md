# Causal Dynamic Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- selection_basis: `validation_only_dynamic_search_on_preselected_current_sleeves`
- objective_profile: `balanced_multi_metric`
- validation_objective: `10.898286`

## Best params

```json
{
  "cash_when_no_active": true,
  "correlation_penalty": 1.0,
  "lookback_days": 20,
  "max_family_weight": 1.0,
  "max_trailing_drawdown": 0.15,
  "max_weight": 0.5,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.0,
  "rebalance_days": 1,
  "regime_strength": 1.5,
  "use_regime_features": true
}
```

## Split metrics

- train: {"cagr": -0.05600794363333916, "calmar": -0.5156088665169948, "max_drawdown": 0.10862486522328474, "sharpe": -0.35400209030793306, "sortino": -0.5279865942846778, "total_return": -0.05600794363333916, "volatility": 0.13670454023259873}
- val: {"cagr": 0.899395706732061, "calmar": 47.54108756863033, "max_drawdown": 0.01891828211615254, "sharpe": 3.666116857029011, "sortino": 6.553183867250698, "total_return": 0.055998327044695984, "volatility": 0.17932597894806557}
- oos: {"cagr": 0.3943184548107568, "calmar": 5.048932635439931, "max_drawdown": 0.07809936936827411, "sharpe": 1.1160540853203742, "sortino": 1.9471139154061994, "total_return": 0.03238795555792473, "volatility": 0.3496737634326915}

## Final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=50.00%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=26.17%
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | weight=2.76%
