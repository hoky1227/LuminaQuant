# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_triplet_probes/composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 32.00%
- market_neutral: 34.00%
- trend: 34.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 34.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 34.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 32.00% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.3923160247919115, "calmar": 34.87025156457156, "max_drawdown": 0.011250736865647037, "sharpe": 3.8141859612171305, "sortino": 7.33405613975802, "total_return": 0.028508463430196862, "volatility": 0.08778655420881679}
- Report (oos): {"cagr": 0.589874305759706, "calmar": 24.408447082287903, "max_drawdown": 0.024166810111723613, "sharpe": 2.7190250242397958, "sortino": 9.930824556094086, "total_return": 0.045463224957480675, "volatility": 0.1760533618379444}
- Train: {"cagr": 0.011456018982367366, "calmar": 0.12846305167596062, "max_drawdown": 0.0891775404126659, "sharpe": 0.16307796449527479, "sortino": 0.32713161194697277, "total_return": 0.011456018982367366, "volatility": 0.10059546780029684}
- Val: {"cagr": 0.3923160247919115, "calmar": 34.87025156457156, "max_drawdown": 0.011250736865647037, "sharpe": 3.8141859612171305, "sortino": 7.33405613975802, "total_return": 0.028508463430196862, "volatility": 0.08778655420881679}
- OOS: {"cagr": 0.589874305759706, "calmar": 24.408447082287903, "max_drawdown": 0.024166810111723613, "sharpe": 2.7190250242397958, "sortino": 9.930824556094086, "total_return": 0.045463224957480675, "volatility": 0.1760533618379444}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5808876902662652,
      "calmar": 23.931049352701713,
      "max_drawdown": 0.024273389841999782,
      "sharpe": 2.686788331978851,
      "sortino": 9.813084950074188,
      "total_return": 0.04489511867733986,
      "volatility": 0.1760533618379444
    },
    "x3": {
      "cagr": 0.5719517323815602,
      "calmar": 23.459913055402897,
      "max_drawdown": 0.02437995959451511,
      "sharpe": 2.654551639717908,
      "sortino": 9.695345344054294,
      "total_return": 0.044327312290692866,
      "volatility": 0.1760533618379444
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5198581081169205,
      "calmar": 23.87892773632264,
      "max_drawdown": 0.021770580063616318,
      "sharpe": 2.7190250242397953,
      "sortino": 9.930824556094084,
      "total_return": 0.040957907671591576,
      "volatility": 0.15844802565414995
    },
    "plus_10pct_signal": {
      "cagr": 0.66263673485496,
      "calmar": 24.95006401382536,
      "max_drawdown": 0.0265585184265571,
      "sharpe": 2.7190250242397953,
      "sortino": 9.930824556094084,
      "total_return": 0.049959024513541506,
      "volatility": 0.19365869802173885
    }
  }
}
```
