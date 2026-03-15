# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_exact_window_freeze_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 30.00%
- trend: 70.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 30.00% | 5.033 | 6.29% | 1.625 | 2.21% |
| 2 | topcap_tsmom_1h_defensive_24_6_0.020 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 30.00% | 2.535 | 2.33% | -2.451 | -4.18% |
| 3 | rolling_breakout_30m_guarded_ls_64_0.002 | RollingBreakoutStrategy | trend | 30m | 30.00% | 3.957 | 10.00% | 1.714 | 7.22% |
| 4 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 10.00% | 3.539 | 8.61% | 2.009 | 8.08% |

## Portfolio metrics

- Fit (val): {"cagr": 0.01583229212704995, "calmar": 0.6567476828347171, "max_drawdown": 0.02410711532123433, "sharpe": 0.8263716074984389, "sortino": 1.1635994479104115, "total_return": 0.06613300996213267, "volatility": 0.019231996753443264}
- Report (oos): {"cagr": 0.005739981865378274, "calmar": 0.08481530890003625, "max_drawdown": 0.06767624783567605, "sharpe": 0.20985287822422158, "sortino": 0.24979371963966698, "total_return": 0.026259583279574894, "volatility": 0.029309132710955532}
- Train: {"cagr": -0.0021128731972431813, "calmar": -0.009959311392015311, "max_drawdown": 0.21215053070206613, "sharpe": -0.08664638328671011, "sortino": -0.08837644338375035, "total_return": -0.09654159371660209, "volatility": 0.021701815344068783}
- Val: {"cagr": 0.01583229212704995, "calmar": 0.6567476828347171, "max_drawdown": 0.02410711532123433, "sharpe": 0.8263716074984389, "sortino": 1.1635994479104115, "total_return": 0.06613300996213267, "volatility": 0.019231996753443264}
- OOS: {"cagr": 0.005739981865378274, "calmar": 0.08481530890003625, "max_drawdown": 0.06767624783567605, "sharpe": 0.20985287822422158, "sortino": 0.24979371963966698, "total_return": 0.026259583279574894, "volatility": 0.029309132710955532}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.0011610638137558027,
      "calmar": 0.014141525502475872,
      "max_drawdown": 0.0821031517110743,
      "sharpe": 0.05416016620442156,
      "sortino": 0.07908360192007341,
      "total_return": 0.005268969897470832,
      "volatility": 0.029309132710955532
    },
    "x3": {
      "cagr": -0.003397064263085392,
      "calmar": -0.03492656769698931,
      "max_drawdown": 0.09726304320988921,
      "sharpe": -0.10153254581537804,
      "sortino": -0.14851792538411865,
      "total_return": -0.015292566119181061,
      "volatility": 0.029309132710955532
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.005202956011385096,
      "calmar": 0.08517992831465733,
      "max_drawdown": 0.06108194869764638,
      "sharpe": 0.20985287822422152,
      "sortino": 0.2497937196396669,
      "total_return": 0.02378023503458948,
      "volatility": 0.026378219439859983
    },
    "plus_10pct_signal": {
      "cagr": 0.006268788198828146,
      "calmar": 0.0844479146139655,
      "max_drawdown": 0.07423259920016367,
      "sharpe": 0.20985287822422155,
      "sortino": 0.24979371963966693,
      "total_return": 0.028705552742712692,
      "volatility": 0.03224004598205108
    }
  }
}
```
