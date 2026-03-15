# Causal Overlay Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_incumbent_bundle_latest.json`
- backbone_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- selection_basis: `validation_only_overlay_search_on_current_one_shot_backbone`
- objective_profile: `balanced_multi_metric_with_backbone_overlay`
- validation_objective: `3.074284`
- oos_start: `2026-02-01T00:00:00Z`
- memory_log: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_overlay_current/_memory_guard/causal_overlay_portfolio_rss_latest.jsonl`

## Best params

```json
{
  "cash_buffer": 0.0,
  "correlation_penalty": 0.0,
  "lookback_days": 20,
  "max_trailing_drawdown": 0.1,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.0,
  "overlay_strength": 0.5,
  "rebalance_days": 3,
  "regime_strength": 0.5
}
```

## Split metrics

- train: {"cagr": -0.06331027851780513, "calmar": -0.4922442104483199, "max_drawdown": 0.12861558790126593, "sharpe": -0.7055158496869055, "sortino": -0.8290427323540349, "total_return": -0.06331027851780513, "volatility": 0.08731531829828217}
- val: {"cagr": 0.07072407535175973, "calmar": 12.423551051117775, "max_drawdown": 0.005692742361725678, "sharpe": 1.9880683478612817, "sortino": 0.0, "total_return": 0.005820679890860214, "volatility": 0.03466792001419261}
- oos: {"cagr": 0.5868243580350623, "calmar": 19.334467292948883, "max_drawdown": 0.030351203844600994, "sharpe": 2.4672054412727857, "sortino": 8.238849788195571, "total_return": 0.04527074280675625, "volatility": 0.19457205533375166}

## Final allocation

- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=50.00%
- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | weight=46.65%
