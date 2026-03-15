# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_four_sleeve_full_retune_bundle_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 30.00%
- trend: 70.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | topcap_tsmom_1h_persistence_24_6_020_24_6_0.020 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 30.00% | 3.004 | 2.77% | -4.176 | -5.92% |
| 2 | rolling_breakout_30m_guarded_ls_64_0.0015 | RollingBreakoutStrategy | trend | 30m | 29.70% | 2.904 | 4.57% | -2.565 | -6.05% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 29.21% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | regime_breakout_1h_tightest_ls_72_0.80 | RegimeBreakoutCandidateStrategy | trend | 1h | 11.09% | 5.293 | 12.55% | -1.738 | -7.35% |

## Portfolio metrics

- Fit (val): {"cagr": 0.01220530081973159, "calmar": 0.6140652068122102, "max_drawdown": 0.019876229241342025, "sharpe": 0.8107858487633829, "sortino": 0.987374919562822, "total_return": 0.05069967151763444, "volatility": 0.015102807432369157}
- Report (oos): {"cagr": -0.007657580806361142, "calmar": -0.11168066151952515, "max_drawdown": 0.06856675723596395, "sharpe": -0.3270243033117061, "sortino": -0.27881032404661177, "total_return": -0.03421386548408756, "volatility": 0.022721619792969182}
- Train: {"cagr": -0.000374666334540974, "calmar": -0.0031466719694832374, "max_drawdown": 0.11906749040717568, "sharpe": -0.014085920927723664, "sortino": -0.01233552072635471, "total_return": -0.017826547052993025, "volatility": 0.016717398403801317}
- Val: {"cagr": 0.01220530081973159, "calmar": 0.6140652068122102, "max_drawdown": 0.019876229241342025, "sharpe": 0.8107858487633829, "sortino": 0.987374919562822, "total_return": 0.05069967151763444, "volatility": 0.015102807432369157}
- OOS: {"cagr": -0.007657580806361142, "calmar": -0.11168066151952515, "max_drawdown": 0.06856675723596395, "sharpe": -0.3270243033117061, "sortino": -0.27881032404661177, "total_return": -0.03421386548408756, "volatility": 0.022721619792969182}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": -0.011136574772497965,
      "calmar": -0.13944748671796914,
      "max_drawdown": 0.07986214047028006,
      "sharpe": -0.4815866452380977,
      "sortino": -0.6909447503525227,
      "total_return": -0.049453211024594546,
      "volatility": 0.022721619792969182
    },
    "x3": {
      "cagr": -0.014603405240744771,
      "calmar": -0.16032716344153491,
      "max_drawdown": 0.09108503467080964,
      "sharpe": -0.6361489871644895,
      "sortino": -0.9139903610146158,
      "total_return": -0.06445223485435336,
      "volatility": 0.022721619792969182
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": -0.006871658573138206,
      "calmar": -0.1110410822378445,
      "max_drawdown": 0.06188393011533744,
      "sharpe": -0.3270243033117062,
      "sortino": -0.2788103240466118,
      "total_return": -0.03074501305726063,
      "volatility": 0.020449457813672262
    },
    "plus_10pct_signal": {
      "cagr": -0.0084479114387368,
      "calmar": -0.11232205212865749,
      "max_drawdown": 0.07521151259825876,
      "sharpe": -0.3270243033117062,
      "sortino": -0.2788103240466119,
      "total_return": -0.03769241284967351,
      "volatility": 0.024993781772266098
    }
  }
}
```
