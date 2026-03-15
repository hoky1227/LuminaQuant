# Causal Dynamic Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_incumbent_bundle_latest.json`
- selection_basis: `validation_only_dynamic_search_on_preselected_current_sleeves`
- objective_profile: `balanced_multi_metric`
- validation_objective: `15.524463`
- oos_start: `2026-02-01T00:00:00Z`
- memory_log: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_dynamic_online_current/_memory_guard/causal_dynamic_portfolio_rss_latest.jsonl`

## Best params

```json
{
  "cash_when_no_active": true,
  "correlation_penalty": 0.0,
  "lookback_days": 20,
  "max_family_weight": 0.55,
  "max_trailing_drawdown": 0.15,
  "max_weight": 0.4,
  "min_trailing_return": 0.0,
  "min_trailing_sharpe": 0.0,
  "rebalance_days": 3,
  "regime_strength": 0.5,
  "use_regime_features": true
}
```

## Split metrics

- train: {"cagr": -0.04506291354773151, "calmar": -0.48039891996064715, "max_drawdown": 0.09380311169605238, "sharpe": -0.6845769641632502, "sortino": -0.9047419350959442, "total_return": -0.04506291354773151, "volatility": 0.06434300124283818}
- val: {"cagr": 0.3201799442772475, "calmar": 70.30441655631694, "max_drawdown": 0.004554193889380609, "sharpe": 5.875273750045433, "sortino": 7.520611551412189, "total_return": 0.023871733925421523, "volatility": 0.04748070589622486}
- oos: {"cagr": 0.3860039356168006, "calmar": 32.576764525421844, "max_drawdown": 0.011849056873514119, "sharpe": 2.718892444240817, "sortino": 8.996887645654638, "total_return": 0.03179603032991585, "volatility": 0.12275996965708974}

## Final allocation

- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | weight=40.00%
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=38.26%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=5.71%
