# Causal Dynamic Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- selection_basis: `validation_only_dynamic_search_on_preselected_current_sleeves`
- objective_profile: `balanced_multi_metric`
- validation_objective: `15.685594`

## Best params

```json
{
  "cash_when_no_active": true,
  "lookback_days": 5,
  "max_trailing_drawdown": 0.15,
  "max_weight": 0.5,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.5,
  "rebalance_days": 3
}
```

## Split metrics

- train: {"cagr": -0.09700624991024498, "calmar": -0.4708182277626524, "max_drawdown": 0.2060375834878413, "sharpe": -0.6237744812722383, "sortino": -1.000217490630906, "total_return": -0.09700624991024498, "volatility": 0.14658688950326454}
- val: {"cagr": 1.1037245623883494, "calmar": 54.93465467944899, "max_drawdown": 0.02009159006875949, "sharpe": 5.146934955840749, "sortino": 13.559745102210139, "total_return": 0.0652018991574479, "volatility": 0.146635557705558}
- oos: {"cagr": 0.2923678179207414, "calmar": 3.4623180361289805, "max_drawdown": 0.08444279666683108, "sharpe": 0.8292557842743573, "sortino": 1.35205267549338, "total_return": 0.02489851115860109, "volatility": 0.3998290548091287}

## Final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=50.00%
