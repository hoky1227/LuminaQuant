# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_triplet_probes/composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 16.38%
- market_neutral: 55.00%
- trend: 28.62%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 55.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 28.62% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 16.38% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.35744612036158063, "calmar": 48.30229051282205, "max_drawdown": 0.0074001898578017755, "sharpe": 4.678051764713497, "sortino": 8.040054210646414, "total_return": 0.02629527740909987, "volatility": 0.06580189300116177}
- Report (oos): {"cagr": 0.72881322049139, "calmar": 47.432416397691526, "max_drawdown": 0.015365298161930885, "sharpe": 3.352955446114801, "sortino": 12.002525335725192, "total_return": 0.05389601077663331, "volatility": 0.1673685585511439}
- Train: {"cagr": 0.030546576467483044, "calmar": 0.4191686702339188, "max_drawdown": 0.07287418797410694, "sharpe": 0.395223562448117, "sortino": 0.6947596298284593, "total_return": 0.030546576467483044, "volatility": 0.08526812512054477}
- Val: {"cagr": 0.35744612036158063, "calmar": 48.30229051282205, "max_drawdown": 0.0074001898578017755, "sharpe": 4.678051764713497, "sortino": 8.040054210646414, "total_return": 0.02629527740909987, "volatility": 0.06580189300116177}
- OOS: {"cagr": 0.72881322049139, "calmar": 47.432416397691526, "max_drawdown": 0.015365298161930885, "sharpe": 3.352955446114801, "sortino": 12.002525335725192, "total_return": 0.05389601077663331, "volatility": 0.1673685585511439}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.7204778439724486,
      "calmar": 46.6120658855917,
      "max_drawdown": 0.015456895769023538,
      "sharpe": 3.324036312130867,
      "sortino": 11.8990039367959,
      "total_return": 0.05340769750519114,
      "volatility": 0.1673685585511439
    },
    "x3": {
      "cagr": 0.7121825464228273,
      "calmar": 45.80398008607491,
      "max_drawdown": 0.015548486072269108,
      "sharpe": 3.295117178146932,
      "sortino": 11.7954825378666,
      "total_return": 0.052919604027339195,
      "volatility": 0.1673685585511439
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.6387037256132553,
      "calmar": 46.16211178182909,
      "max_drawdown": 0.013836102833247543,
      "sharpe": 3.352955446114801,
      "sortino": 12.00252533572519,
      "total_return": 0.048500242983829045,
      "volatility": 0.15063170269602952
    },
    "plus_10pct_signal": {
      "cagr": 0.823396834952159,
      "calmar": 48.74227640999265,
      "max_drawdown": 0.016892867867438266,
      "sharpe": 3.3529554461148017,
      "sortino": 12.002525335725192,
      "total_return": 0.05929276281928608,
      "volatility": 0.1841054144062583
    }
  }
}
```
