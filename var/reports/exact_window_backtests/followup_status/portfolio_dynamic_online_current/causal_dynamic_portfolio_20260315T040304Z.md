# Causal Dynamic Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_incumbent_bundle_latest.json`
- selection_basis: `validation_only_dynamic_search_on_preselected_current_sleeves`
- objective_profile: `balanced_multi_metric`
- validation_objective: `3.029332`
- oos_start: `2026-02-01T00:00:00Z`
- memory_log: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_dynamic_online_current/_memory_guard/causal_dynamic_portfolio_rss_latest.jsonl`

## Best params

```json
{
  "cash_when_no_active": true,
  "correlation_penalty": 0.0,
  "lookback_days": 20,
  "max_family_weight": 0.55,
  "max_trailing_drawdown": 0.1,
  "max_weight": 0.5,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.0,
  "rebalance_days": 3,
  "regime_strength": 0.5,
  "use_regime_features": true
}
```

## Split metrics

- train: {"cagr": -0.06554325751578971, "calmar": -0.5191250822005327, "max_drawdown": 0.12625715798195825, "sharpe": -0.8081760399756627, "sortino": -0.890453293670511, "total_return": -0.06554325751578971, "volatility": 0.07993297175443093}
- val: {"cagr": 0.07072407535175973, "calmar": 12.423551051117775, "max_drawdown": 0.005692742361725678, "sharpe": 1.9880683478612817, "sortino": 0.0, "total_return": 0.005820679890860214, "volatility": 0.03466792001419261}
- oos: {"cagr": 0.4526158380287906, "calmar": 31.84220096304904, "max_drawdown": 0.01421433896965929, "sharpe": 2.9445254139423347, "sortino": 11.155284424011684, "total_return": 0.03645083188181708, "volatility": 0.12959329078232643}

## Final allocation

- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | weight=50.00%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=8.45%
