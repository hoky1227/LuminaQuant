# Causal Overlay Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_incumbent_bundle_latest.json`
- backbone_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- selection_basis: `validation_only_overlay_search_on_current_one_shot_backbone`
- objective_profile: `balanced_multi_metric_with_backbone_overlay`
- validation_objective: `17.168327`
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

- train: {"cagr": -0.09821323319146169, "calmar": -0.7440658565485265, "max_drawdown": 0.1319953500447395, "sharpe": -0.842677353989956, "sortino": -1.1192513482225537, "total_return": -0.09821323319146169, "volatility": 0.11487054871592432}
- val: {"cagr": 0.2213716174872593, "calmar": 82.22307489774425, "max_drawdown": 0.002692329589504716, "sharpe": 5.76839737009594, "sortino": 8.098610586819188, "total_return": 0.017129186402154106, "volatility": 0.03477793370278213}
- oos: {"cagr": 0.26254137673823785, "calmar": 4.590543652738419, "max_drawdown": 0.05719178306508921, "sharpe": 0.9540071089926616, "sortino": 2.221828079263361, "total_return": 0.022606347417474693, "volatility": 0.28440295396074344}

## Final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=35.00%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=35.00%
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | weight=23.53%
