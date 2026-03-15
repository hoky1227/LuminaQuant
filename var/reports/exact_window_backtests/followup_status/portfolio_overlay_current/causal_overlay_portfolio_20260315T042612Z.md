# Causal Overlay Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_incumbent_bundle_latest.json`
- backbone_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- selection_basis: `validation_only_overlay_search_on_current_one_shot_backbone`
- objective_profile: `balanced_multi_metric_with_backbone_overlay`
- validation_objective: `9.711425`
- oos_start: `2026-02-01T00:00:00Z`
- memory_log: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_overlay_current/_memory_guard/causal_overlay_portfolio_rss_latest.jsonl`

## Best params

```json
{
  "cash_buffer": 0.0,
  "correlation_penalty": 0.0,
  "lookback_days": 20,
  "max_trailing_drawdown": 0.15,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.0,
  "overlay_strength": 0.5,
  "rebalance_days": 3,
  "regime_strength": 0.5
}
```

## Split metrics

- train: {"cagr": -0.024119742075664585, "calmar": -0.3098569264897007, "max_drawdown": 0.07784154560916778, "sharpe": -0.3595953821080424, "sortino": -0.4537540603057438, "total_return": -0.024119742075664585, "volatility": 0.06249284602640957}
- val: {"cagr": 0.2448436662393625, "calmar": 35.841493531238356, "max_drawdown": 0.006831290834070969, "sharpe": 4.661985354631714, "sortino": 3.7482208308868197, "total_return": 0.018774918653491124, "volatility": 0.04722302918965266}
- oos: {"cagr": 0.7200821058511753, "calmar": 55.98898268422918, "max_drawdown": 0.01286113930507271, "sharpe": 3.4496548450543796, "sortino": 12.164513377393309, "total_return": 0.05338446074194625, "volatility": 0.16091888831080745}

## Final allocation

- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | weight=56.46%
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=25.44%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=14.56%
