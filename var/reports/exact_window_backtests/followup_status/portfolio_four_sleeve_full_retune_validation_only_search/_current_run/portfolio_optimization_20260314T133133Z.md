# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_four_sleeve_full_retune_bundle_validation_only_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 30.00%
- trend: 70.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | topcap_tsmom_1h_slow_rebalance_16_6_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 30.00% | 3.148 | 2.92% | 1.292 | 2.20% |
| 2 | rolling_breakout_30m_guarded_ls_64_0.0015 | RollingBreakoutStrategy | trend | 30m | 30.00% | 2.904 | 4.57% | -2.565 | -6.05% |
| 3 | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 28.44% | 5.033 | 6.29% | 1.625 | 2.21% |
| 4 | regime_breakout_1h_tightest_ls_72_0.80 | RegimeBreakoutCandidateStrategy | trend | 1h | 11.56% | 5.293 | 12.55% | -1.738 | -7.35% |

## Portfolio metrics

- Fit (val): {"cagr": 0.013277058945580178, "calmar": 0.675172045582812, "max_drawdown": 0.019664704770351316, "sharpe": 0.860126809268491, "sortino": 1.0994135574977593, "total_return": 0.055242477512001065, "volatility": 0.01547338676390309}
- Report (oos): {"cagr": -0.0027739341295880404, "calmar": -0.05005726741876287, "max_drawdown": 0.055415212867738206, "sharpe": -0.10227384389003721, "sortino": -0.09886733957501666, "total_return": -0.012501160851255522, "volatility": 0.024300119611564308}
- Train: {"cagr": -0.0003508504611026142, "calmar": -0.003208535828262477, "max_drawdown": 0.10934908627547124, "sharpe": -0.011025938388023258, "sortino": -0.009773172038552958, "total_return": -0.016702713838570915, "volatility": 0.017692714104199765}
- Val: {"cagr": 0.013277058945580178, "calmar": 0.675172045582812, "max_drawdown": 0.019664704770351316, "sharpe": 0.860126809268491, "sortino": 1.0994135574977593, "total_return": 0.055242477512001065, "volatility": 0.01547338676390309}
- OOS: {"cagr": -0.0027739341295880404, "calmar": -0.05005726741876287, "max_drawdown": 0.055415212867738206, "sharpe": -0.10227384389003721, "sortino": -0.09886733957501666, "total_return": -0.012501160851255522, "volatility": 0.024300119611564308}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": -0.006907504669694253,
      "calmar": -0.10018427358904636,
      "max_drawdown": 0.06894799375427607,
      "sharpe": -0.2732040623143464,
      "sortino": -0.4390140592975175,
      "total_return": -0.03090343920796157,
      "volatility": 0.024300119611564308
    },
    "x3": {
      "cagr": -0.011023988024231657,
      "calmar": -0.1338185218632235,
      "max_drawdown": 0.08238013595382054,
      "sharpe": -0.4441342807386557,
      "sortino": -0.7156739704983937,
      "total_return": -0.04896299027208306,
      "volatility": 0.024300119611564308
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": -0.002470826958958394,
      "calmar": -0.049449715375874226,
      "max_drawdown": 0.049966454613081024,
      "sharpe": -0.10227384389003707,
      "sortino": -0.09886733957501655,
      "total_return": -0.01114111962044706,
      "volatility": 0.021870107650407877
    },
    "plus_10pct_signal": {
      "cagr": -0.0030826840251860776,
      "calmar": -0.05066591419369186,
      "max_drawdown": 0.06084335147691633,
      "sharpe": -0.10227384389003724,
      "sortino": -0.09886733957501669,
      "total_return": -0.01388502206174358,
      "volatility": 0.02673013157272074
    }
  }
}
```
