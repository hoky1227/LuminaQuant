# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_triplet_probes/composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 14.56%
- market_neutral: 60.00%
- trend: 25.44%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 60.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.44% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 14.56% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.34003864659038885, "calmar": 43.35649610335891, "max_drawdown": 0.007842853485663603, "sharpe": 4.661185401236607, "sortino": 7.569957740388168, "total_return": 0.025170890133217716, "volatility": 0.06323456277167129}
- Report (oos): {"cagr": 0.7644493706045741, "calmar": 53.54257718910656, "max_drawdown": 0.014277410814661051, "sharpe": 3.4471209004535015, "sortino": 12.272011214344426, "total_return": 0.05595997643375439, "volatility": 0.1687910826754393}
- Train: {"cagr": 0.031128987733148206, "calmar": 0.4387372048542752, "max_drawdown": 0.07095132892476619, "sharpe": 0.40169098218880034, "sortino": 0.6684884761882702, "total_return": 0.031128987733148206, "volatility": 0.08531609458298543}
- Val: {"cagr": 0.34003864659038885, "calmar": 43.35649610335891, "max_drawdown": 0.007842853485663603, "sharpe": 4.661185401236607, "sortino": 7.569957740388168, "total_return": 0.025170890133217716, "volatility": 0.06323456277167129}
- OOS: {"cagr": 0.7644493706045741, "calmar": 53.54257718910656, "max_drawdown": 0.014277410814661051, "sharpe": 3.4471209004535015, "sortino": 12.272011214344426, "total_return": 0.05595997643375439, "volatility": 0.1687910826754393}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.7560722799533763,
      "calmar": 52.62307530261478,
      "max_drawdown": 0.014367694696775124,
      "sharpe": 3.4188835136275526,
      "sortino": 12.17148398080694,
      "total_return": 0.055478205540062,
      "volatility": 0.1687910826754393
    },
    "x3": {
      "cagr": 0.7477348528277457,
      "calmar": 51.71782592733425,
      "max_drawdown": 0.014457971490881016,
      "sharpe": 3.3906461268016024,
      "sortino": 12.070956747269456,
      "total_return": 0.05499664817163996,
      "volatility": 0.1687910826754393
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.6691120370791261,
      "calmar": 52.047262809544065,
      "max_drawdown": 0.012855854486096607,
      "sharpe": 3.447120900453501,
      "sortino": 12.272011214344424,
      "total_return": 0.05035044793573573,
      "volatility": 0.15191197440789536
    },
    "plus_10pct_signal": {
      "cagr": 0.8647310833968871,
      "calmar": 55.08684795864265,
      "max_drawdown": 0.01569759598599829,
      "sharpe": 3.4471209004535015,
      "sortino": 12.272011214344426,
      "total_return": 0.0615721088742649,
      "volatility": 0.18567019094298326
    }
  }
}
```
