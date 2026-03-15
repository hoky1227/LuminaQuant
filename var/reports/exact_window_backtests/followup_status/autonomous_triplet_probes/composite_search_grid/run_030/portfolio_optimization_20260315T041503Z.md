# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_triplet_probes/composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 18.20%
- market_neutral: 50.00%
- trend: 31.80%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 50.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 31.80% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 18.20% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.37501224255982746, "calmar": 50.41243399940642, "max_drawdown": 0.007438883878613023, "sharpe": 4.649976358817286, "sortino": 8.427308369509293, "total_return": 0.02741661565219422, "volatility": 0.0690113873743435}
- Report (oos): {"cagr": 0.693785091962901, "calmar": 42.169609437532245, "max_drawdown": 0.016452253203592893, "sharpe": 3.2493926615716116, "sortino": 11.741192492461074, "total_return": 0.051829427310355225, "volatility": 0.16634349858680117}
- Train: {"cagr": 0.02989635188028772, "calmar": 0.3995984825091484, "max_drawdown": 0.0748159795116421, "sharpe": 0.38521311559792637, "sortino": 0.717534293404377, "total_return": 0.02989635188028772, "volatility": 0.08600236186646883}
- Val: {"cagr": 0.37501224255982746, "calmar": 50.41243399940642, "max_drawdown": 0.007438883878613023, "sharpe": 4.649976358817286, "sortino": 8.427308369509293, "total_return": 0.02741661565219422, "volatility": 0.0690113873743435}
- OOS: {"cagr": 0.693785091962901, "calmar": 42.169609437532245, "max_drawdown": 0.016452253203592893, "sharpe": 3.2493926615716116, "sortino": 11.741192492461074, "total_return": 0.051829427310355225, "volatility": 0.16634349858680117}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.6855072981136834,
      "calmar": 41.43287495424873,
      "max_drawdown": 0.01654500921962665,
      "sharpe": 3.2198994132829606,
      "sortino": 11.634623068125169,
      "total_return": 0.051335414523862255,
      "volatility": 0.16634349858680117
    },
    "x3": {
      "cagr": 0.6772698488409892,
      "calmar": 40.70679832711507,
      "max_drawdown": 0.016637757737627212,
      "sharpe": 3.1904061649943105,
      "sortino": 11.528053643789269,
      "total_return": 0.050841627133719314,
      "volatility": 0.16634349858680117
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.6087632787736093,
      "calmar": 41.089356136298406,
      "max_drawdown": 0.014815595473296472,
      "sharpe": 3.2493926615716116,
      "sortino": 11.741192492461073,
      "total_return": 0.046647924569001686,
      "volatility": 0.14970914872812105
    },
    "plus_10pct_signal": {
      "cagr": 0.7828368249720836,
      "calmar": 43.281709733187775,
      "max_drawdown": 0.01808701250015121,
      "sharpe": 3.249392661571611,
      "sortino": 11.74119249246107,
      "total_return": 0.057010238762165155,
      "volatility": 0.1829778484454813
    }
  }
}
```
