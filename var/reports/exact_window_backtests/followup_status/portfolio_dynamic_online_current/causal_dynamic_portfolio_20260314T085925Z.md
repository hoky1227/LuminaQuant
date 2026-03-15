# Causal Dynamic Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- selection_basis: `validation_only_dynamic_search_on_preselected_current_sleeves`
- objective_profile: `balanced_multi_metric`
- validation_objective: `10.975854`

## Best params

```json
{
  "cash_when_no_active": true,
  "lookback_days": 20,
  "max_trailing_drawdown": 0.15,
  "max_weight": 0.5,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.0,
  "rebalance_days": 1,
  "regime_strength": 0.5,
  "use_regime_features": true
}
```

## Split metrics

- train: {"cagr": -0.10673147419222995, "calmar": -0.7410147412482713, "max_drawdown": 0.14403421180587606, "sharpe": -0.6678761765701424, "sortino": -0.9057540825465794, "total_return": -0.10673147419222995, "volatility": 0.15185497853603006}
- val: {"cagr": 0.7131025159360145, "calmar": 49.37294950883458, "max_drawdown": 0.014443182411219224, "sharpe": 3.6880701128973974, "sortino": 6.347176308399388, "total_return": 0.046780376281467584, "volatility": 0.14894143181979863}
- oos: {"cagr": 0.28814101829152183, "calmar": 4.2066055810009155, "max_drawdown": 0.06849727476065437, "sharpe": 0.9395900547598108, "sortino": 1.7156908302959193, "total_return": 0.02457660852837562, "volatility": 0.3215160998823203}

## Final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=50.00%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=16.38%
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | weight=13.28%
