# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_four_sleeve_anchored_bundle_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 25.00%
- trend: 75.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 25.00% | 3.539 | 8.61% | 2.009 | 8.08% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 25.00% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | rolling_breakout_30m_guarded_ls_64_0.002 | RollingBreakoutStrategy | trend | 30m | 25.00% | 2.397 | 3.63% | -2.363 | -5.41% |

## Portfolio metrics

- Fit (val): {"cagr": 0.01169814972008254, "calmar": 0.36772704866549594, "max_drawdown": 0.031812045816416945, "sharpe": 0.5130599889377316, "sortino": 0.54142373221041, "total_return": 0.048555188197246135, "volatility": 0.023188484334050418}
- Report (oos): {"cagr": 0.005275241954893373, "calmar": 0.10661685522394937, "max_drawdown": 0.04947849890913303, "sharpe": 0.16255585772534367, "sortino": 0.17955984154294544, "total_return": 0.024113693640387845, "volatility": 0.03634035179397559}
- Train: {"cagr": -0.0015728851113596676, "calmar": -0.010768789390228871, "max_drawdown": 0.14605960376445237, "sharpe": -0.05584733153066415, "sortino": -0.04683010266311663, "total_return": -0.07277397864237178, "volatility": 0.023339633903198712}
- Val: {"cagr": 0.01169814972008254, "calmar": 0.36772704866549594, "max_drawdown": 0.031812045816416945, "sharpe": 0.5130599889377316, "sortino": 0.54142373221041, "total_return": 0.048555188197246135, "volatility": 0.023188484334050418}
- OOS: {"cagr": 0.005275241954893373, "calmar": 0.10661685522394937, "max_drawdown": 0.04947849890913303, "sharpe": 0.16255585772534367, "sortino": 0.17955984154294544, "total_return": 0.024113693640387845, "volatility": 0.03634035179397559}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": -0.0015607711682594783,
      "calmar": -0.02173105698216462,
      "max_drawdown": 0.07182214696415612,
      "sharpe": -0.02520759154337536,
      "sortino": -0.04846691177325597,
      "total_return": -0.007048929878813559,
      "volatility": 0.03634035179397559
    },
    "x3": {
      "cagr": -0.008350424934841394,
      "calmar": -0.08910049151482935,
      "max_drawdown": 0.09371917924214368,
      "sharpe": -0.2129710408120947,
      "sortino": -0.4105122258776439,
      "total_return": -0.03726386610787058,
      "volatility": 0.03634035179397559
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.004803788252447649,
      "calmar": 0.10780460482730794,
      "max_drawdown": 0.044560139709642566,
      "sharpe": 0.16255585772534376,
      "sortino": 0.17955984154294558,
      "total_return": 0.021940377735706562,
      "volatility": 0.032706316614578027
    },
    "plus_10pct_signal": {
      "cagr": 0.0057344496629681,
      "calmar": 0.10543253713119873,
      "max_drawdown": 0.054389753097113025,
      "sharpe": 0.1625558577253437,
      "sortino": 0.17955984154294544,
      "total_return": 0.026234018306123374,
      "volatility": 0.03997438697337314
    }
  }
}
```
