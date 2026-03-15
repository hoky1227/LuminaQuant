# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/regime_breakout_1h_trend_ls_48_0.70.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 20.18%
- market_neutral: 35.00%
- trend: 44.82%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 35.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 35.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 20.18% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 9.82% | 3.539 | 8.61% | 2.009 | 8.08% |

## Portfolio metrics

- Fit (val): {"cagr": 0.510633781461624, "calmar": 37.84955956275082, "max_drawdown": 0.013491141967320486, "sharpe": 4.046683666695413, "sortino": 8.009481288228248, "total_return": 0.03565775179184083, "volatility": 0.10326966050774286}
- Report (oos): {"cagr": 0.6863698804460618, "calmar": 24.06034726047788, "max_drawdown": 0.028527014719089694, "sharpe": 2.6661836170669932, "sortino": 8.584888340766478, "total_return": 0.05138699507094091, "volatility": 0.20350997399138515}
- Train: {"cagr": -0.0067476869666511785, "calmar": -0.07003867747117844, "max_drawdown": 0.09634229557558271, "sharpe": -0.008633712778206452, "sortino": -0.01913167182378677, "total_return": -0.0067476869666511785, "volatility": 0.10855697755187961}
- Val: {"cagr": 0.510633781461624, "calmar": 37.84955956275082, "max_drawdown": 0.013491141967320486, "sharpe": 4.046683666695413, "sortino": 8.009481288228248, "total_return": 0.03565775179184083, "volatility": 0.10326966050774286}
- OOS: {"cagr": 0.6863698804460618, "calmar": 24.06034726047788, "max_drawdown": 0.028527014719089694, "sharpe": 2.6661836170669932, "sortino": 8.584888340766478, "total_return": 0.05138699507094091, "volatility": 0.20350997399138515}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.6761623978869089,
      "calmar": 23.608404359433653,
      "max_drawdown": 0.028640749607320326,
      "sharpe": 2.63630963171709,
      "sortino": 8.488696605553304,
      "total_return": 0.05077507474128762,
      "volatility": 0.20350997399138515
    },
    "x3": {
      "cagr": 0.6660165322581679,
      "calmar": 23.162188726450008,
      "max_drawdown": 0.02875447308214063,
      "sharpe": 2.606435646367186,
      "sortino": 8.392504870340126,
      "total_return": 0.05016350038601769,
      "volatility": 0.20350997399138515
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.6033297227143564,
      "calmar": 23.47694993237171,
      "max_drawdown": 0.025698812002935778,
      "sharpe": 2.6661836170669932,
      "sortino": 8.584888340766478,
      "total_return": 0.04630843080342539,
      "volatility": 0.18315897659224664
    },
    "plus_10pct_signal": {
      "cagr": 0.7730346252063247,
      "calmar": 24.658362982468468,
      "max_drawdown": 0.03134979502718549,
      "sharpe": 2.666183617066994,
      "sortino": 8.58488834076648,
      "total_return": 0.05645157799597489,
      "volatility": 0.22386097139052363
    }
  }
}
```
