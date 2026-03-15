# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_residual_topcap_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 33.33%
- market_neutral: 33.33%
- trend: 33.33%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 33.33% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | topcap_tsmom_1h_resid_beta_neutral_24_4_0.008 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 33.33% | 0.394 | 0.41% | 1.898 | 4.30% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 33.33% | 3.880 | 1.61% | 4.557 | 7.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.31403303783843795, "calmar": 23.491730620524063, "max_drawdown": 0.013367811972272325, "sharpe": 3.3023792048355736, "sortino": 5.707470488671383, "total_return": 0.023465978051811387, "volatility": 0.08375478437848409}
- Report (oos): {"cagr": 0.6436662023326494, "calmar": 30.403903128554735, "max_drawdown": 0.021170512207300485, "sharpe": 2.9361597453093102, "sortino": 8.743937803634527, "total_return": 0.048804295051793734, "volatility": 0.17428241704886177}
- Train: {"cagr": -0.06863807479828121, "calmar": -0.4867673278769616, "max_drawdown": 0.14100797417453337, "sharpe": -0.6492027906292361, "sortino": -1.172414754323393, "total_return": -0.06863807479828121, "volatility": 0.10162159835573469}
- Val: {"cagr": 0.31403303783843795, "calmar": 23.491730620524063, "max_drawdown": 0.013367811972272325, "sharpe": 3.3023792048355736, "sortino": 5.707470488671383, "total_return": 0.023465978051811387, "volatility": 0.08375478437848409}
- OOS: {"cagr": 0.6436662023326494, "calmar": 30.403903128554735, "max_drawdown": 0.021170512207300485, "sharpe": 2.9361597453093102, "sortino": 8.743937803634527, "total_return": 0.048804295051793734, "volatility": 0.17428241704886177}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.6346390689619206,
      "calmar": 29.81047482510924,
      "max_drawdown": 0.021289129833898746,
      "sharpe": 2.9045188234761214,
      "sortino": 8.649710555610623,
      "total_return": 0.048250579744743005,
      "volatility": 0.17428241704886177
    },
    "x3": {
      "cagr": 0.6256613781637275,
      "calmar": 29.225949480047593,
      "max_drawdown": 0.02140773488268921,
      "sharpe": 2.8728779016429313,
      "sortino": 8.555483307586716,
      "total_return": 0.04769714842246442,
      "volatility": 0.17428241704886177
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5660278757893378,
      "calmar": 29.68918424045838,
      "max_drawdown": 0.019065120523520274,
      "sharpe": 2.936159745309311,
      "sortino": 8.743937803634529,
      "total_return": 0.04394928937041365,
      "volatility": 0.15685417534397558
    },
    "plus_10pct_signal": {
      "cagr": 0.7246643078381512,
      "calmar": 31.137131936196763,
      "max_drawdown": 0.023273315902153868,
      "sharpe": 2.9361597453093102,
      "sortino": 8.743937803634525,
      "total_return": 0.05365322078920309,
      "volatility": 0.19171065875374793
    }
  }
}
```
