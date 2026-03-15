# Causal Dynamic Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- selection_basis: `validation_only_dynamic_search`
- validation_objective: `5.288416`

## Best params

```json
{
  "cash_when_no_active": true,
  "lookback_days": 5,
  "max_trailing_drawdown": 0.05,
  "max_weight": 0.4,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.5,
  "rebalance_days": 3
}
```

## Split metrics

- train: {"cagr": -0.14626719742001815, "calmar": -0.5444372785848819, "max_drawdown": 0.26865757209021457, "sharpe": -0.7445278410083921, "sortino": -1.1531894812169827, "total_return": -0.14626719742001815, "volatility": 0.1887981017077212}
- val: {"cagr": 1.262574955419021, "calmar": 39.99169791910051, "max_drawdown": 0.03157092649512139, "sharpe": 4.638825538479358, "sortino": 8.449290778871816, "total_return": 0.07180792780245704, "volatility": 0.17953851406162596}
- oos: {"cagr": 0.3854422209915782, "calmar": 4.932852585370817, "max_drawdown": 0.07813779437372004, "sharpe": 1.0371734222761868, "sortino": 1.808395677957515, "total_return": 0.031755925204412394, "volatility": 0.37971119245119744}

## Final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=100.00%
