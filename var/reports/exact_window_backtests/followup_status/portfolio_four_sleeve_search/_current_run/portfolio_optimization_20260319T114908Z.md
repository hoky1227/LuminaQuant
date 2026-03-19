# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_four_sleeve_anchored_bundle_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 30.00%
- trend: 70.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 30.00% | 1.641 | 1.84% | 1.464 | 3.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 28.74% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | rolling_breakout_30m_guarded_ls_64_0.002 | RollingBreakoutStrategy | trend | 30m | 28.74% | 2.397 | 3.63% | -2.363 | -5.41% |
| 4 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 12.51% | 3.539 | 8.61% | 2.009 | 8.08% |

## Portfolio metrics

- Fit (val): {"cagr": 0.010084911417625131, "calmar": 0.38684908569158005, "max_drawdown": 0.026069368626259193, "sharpe": 0.5420636417937691, "sortino": 0.5810710280933017, "total_return": 0.04175558675355351, "volatility": 0.018836998903523514}
- Report (oos): {"cagr": 0.003098589737781987, "calmar": 0.07207169894478588, "max_drawdown": 0.04299315519335567, "sharpe": 0.12155966599016298, "sortino": 0.1412301794940194, "total_return": 0.014109710634370165, "volatility": 0.028805453278722228}
- Train: {"cagr": -0.0007035235830785691, "calmar": -0.0058512695547569575, "max_drawdown": 0.12023434854520065, "sharpe": -0.025499797392377344, "sortino": -0.022055607202940284, "total_return": -0.03321680873494026, "volatility": 0.01988537155667895}
- Val: {"cagr": 0.010084911417625131, "calmar": 0.38684908569158005, "max_drawdown": 0.026069368626259193, "sharpe": 0.5420636417937691, "sortino": 0.5810710280933017, "total_return": 0.04175558675355351, "volatility": 0.018836998903523514}
- OOS: {"cagr": 0.003098589737781987, "calmar": 0.07207169894478588, "max_drawdown": 0.04299315519335567, "sharpe": 0.12155966599016298, "sortino": 0.1412301794940194, "total_return": 0.014109710634370165, "volatility": 0.028805453278722228}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": -0.002642253067365208,
      "calmar": -0.0426675263247224,
      "max_drawdown": 0.06192655855547535,
      "sharpe": -0.07769240412480433,
      "sortino": -0.15075924298380805,
      "total_return": -0.011910487381742607,
      "volatility": 0.028805453278722228
    },
    "x3": {
      "cagr": -0.008350329903173415,
      "calmar": -0.10363194043135344,
      "max_drawdown": 0.08057679773645399,
      "sharpe": -0.2769444742397716,
      "sortino": -0.5383326065940464,
      "total_return": -0.037263448279951206,
      "volatility": 0.028805453278722228
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.0028245622615550303,
      "calmar": 0.07294108194732467,
      "max_drawdown": 0.03872388763844803,
      "sharpe": 0.12155966599016305,
      "sortino": 0.14123017949401942,
      "total_return": 0.012855685933361505,
      "volatility": 0.025924907950850003
    },
    "plus_10pct_signal": {
      "cagr": 0.0033647705423147656,
      "calmar": 0.07120386075971127,
      "max_drawdown": 0.047255450847949354,
      "sharpe": 0.12155966599016305,
      "sortino": 0.14123017949401942,
      "total_return": 0.015328984811645574,
      "volatility": 0.03168599860659445
    }
  }
}
```
