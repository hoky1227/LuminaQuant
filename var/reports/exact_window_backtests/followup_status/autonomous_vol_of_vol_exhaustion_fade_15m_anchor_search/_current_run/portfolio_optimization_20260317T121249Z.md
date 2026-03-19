# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_vol_of_vol_exhaustion_fade_15m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 30.00%
- market_neutral: 30.00%
- mean_reversion: 10.00%
- trend: 30.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 30.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 30.00% | 1.641 | 1.84% | 1.464 | 3.24% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 4 | vol_of_vol_exhaustion_fade_15m_balanced_ls_24_1.8 | VolOfVolExhaustionFadeStrategy | mean_reversion | 15m | 10.00% | 0.000 | 0.00% | 0.000 | 0.00% |

## Portfolio metrics

- Fit (val): {"cagr": 0.3448124032462294, "calmar": 33.363880388942164, "max_drawdown": 0.010334901073452807, "sharpe": 3.72976013306876, "sortino": 7.138464135293634, "total_return": 0.0254805616159397, "volatility": 0.08029584051967763}
- Report (oos): {"cagr": 0.5165821147901493, "calmar": 23.03834726089827, "max_drawdown": 0.022422707190759117, "sharpe": 2.6832221906378626, "sortino": 9.758223574528905, "total_return": 0.04074254429287305, "volatility": 0.15983548351776994}
- Train: {"cagr": 0.008703575464519497, "calmar": 0.10649668411646193, "max_drawdown": 0.08172625783354437, "sharpe": 0.13980898803129532, "sortino": 0.2786087339101769, "total_return": 0.008703575464519497, "volatility": 0.0919976678431982}
- Val: {"cagr": 0.3448124032462294, "calmar": 33.363880388942164, "max_drawdown": 0.010334901073452807, "sharpe": 3.72976013306876, "sortino": 7.138464135293634, "total_return": 0.0254805616159397, "volatility": 0.08029584051967763}
- OOS: {"cagr": 0.5165821147901493, "calmar": 23.03834726089827, "max_drawdown": 0.022422707190759117, "sharpe": 2.6832221906378626, "sortino": 9.758223574528905, "total_return": 0.04074254429287305, "volatility": 0.15983548351776994}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5088771689126075,
      "calmar": 22.59807139525394,
      "max_drawdown": 0.02251861054919413,
      "sharpe": 2.6513204568585516,
      "sortino": 9.642204762624392,
      "total_return": 0.04023435951734422,
      "volatility": 0.15983548351776994
    },
    "x3": {
      "cagr": 0.5012112610517621,
      "calmar": 22.1632639036067,
      "max_drawdown": 0.02261450584316682,
      "sharpe": 2.6194187230792396,
      "sortino": 9.526185950719883,
      "total_return": 0.03972641579679359,
      "volatility": 0.15983548351776994
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.45631785923894563,
      "calmar": 22.59209591249007,
      "max_drawdown": 0.02019811977633601,
      "sharpe": 2.6832221906378617,
      "sortino": 9.7582235745289,
      "total_return": 0.036703827030993175,
      "volatility": 0.14385193516599296
    },
    "plus_10pct_signal": {
      "cagr": 0.5789629283770794,
      "calmar": 23.49364596186713,
      "max_drawdown": 0.02464338354790918,
      "sharpe": 2.683222190637862,
      "sortino": 9.7582235745289,
      "total_return": 0.04477306158446659,
      "volatility": 0.17581903186954695
    }
  }
}
```
