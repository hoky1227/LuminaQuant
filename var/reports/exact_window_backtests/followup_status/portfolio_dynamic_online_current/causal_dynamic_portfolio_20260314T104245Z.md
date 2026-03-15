# Causal Dynamic Portfolio

- input_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_incumbent_bundle_latest.json`
- selection_basis: `validation_only_dynamic_search_on_preselected_current_sleeves`
- objective_profile: `balanced_multi_metric`
- validation_objective: `17.150987`
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

- train: {"cagr": -0.06415062218390044, "calmar": -0.6205198866037431, "max_drawdown": 0.10338205683465274, "sharpe": -0.6146120834215115, "sortino": -0.8633262098821006, "total_return": -0.06415062218390044, "volatility": 0.0998219468989189}
- val: {"cagr": 0.330145942071572, "calmar": 85.83724680328498, "max_drawdown": 0.0038461851278638326, "sharpe": 5.768397370095939, "sortino": 8.098610586819186, "total_return": 0.024525928052964163, "volatility": 0.04968276243254591}
- oos: {"cagr": 0.13781667344586013, "calmar": 5.985883861822879, "max_drawdown": 0.023023612991363795, "sharpe": 1.1764008956920013, "sortino": 3.130311250245071, "total_return": 0.012457484650929773, "volatility": 0.11517336627567139}

## Final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=50.00%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=11.54%
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | weight=2.58%
