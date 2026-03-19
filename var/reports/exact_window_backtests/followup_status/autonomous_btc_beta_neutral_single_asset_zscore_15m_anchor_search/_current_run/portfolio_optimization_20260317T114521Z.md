# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_btc_beta_neutral_single_asset_zscore_15m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 23.25%
- market_neutral: 30.00%
- mean_reversion: 16.75%
- trend: 30.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 30.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 23.25% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | mean_reversion_std_15m_resid_btc_guarded_lo_96_2.40 | MeanReversionStdStrategy | mean_reversion | 15m | 16.75% | -6.202 | -6.89% | -2.604 | -4.00% |

## Portfolio metrics

- Fit (val): {"cagr": 0.15815575404006443, "calmar": 23.21041231083407, "max_drawdown": 0.006814000196206815, "sharpe": 2.9862826987790303, "sortino": 5.529179725797286, "total_return": 0.01254847703961981, "volatility": 0.049575349639153055}
- Report (oos): {"cagr": 0.3841942634715225, "calmar": 23.80726985384792, "max_drawdown": 0.016137686758291858, "sharpe": 2.4133365509201856, "sortino": 8.89228835544682, "total_return": 0.03166677128446982, "volatility": 0.13857794090179273}
- Train: {"cagr": 0.005979683105917344, "calmar": 0.10117509252636467, "max_drawdown": 0.059102324066164114, "sharpe": 0.11504144143141014, "sortino": 0.20725743482956555, "total_return": 0.005979683105917344, "volatility": 0.07826136093170857}
- Val: {"cagr": 0.15815575404006443, "calmar": 23.21041231083407, "max_drawdown": 0.006814000196206815, "sharpe": 2.9862826987790303, "sortino": 5.529179725797286, "total_return": 0.01254847703961981, "volatility": 0.049575349639153055}
- OOS: {"cagr": 0.3841942634715225, "calmar": 23.80726985384792, "max_drawdown": 0.016137686758291858, "sharpe": 2.4133365509201856, "sortino": 8.89228835544682, "total_return": 0.03166677128446982, "volatility": 0.13857794090179273}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.3777609818912806,
      "calmar": 23.281410382671908,
      "max_drawdown": 0.0162258632824257,
      "sharpe": 2.3796911322247607,
      "sortino": 8.768316933075008,
      "total_return": 0.03120602236029768,
      "volatility": 0.13857794090179273
    },
    "x3": {
      "cagr": 0.37135751855541566,
      "calmar": 22.763072614166028,
      "max_drawdown": 0.016314033032794995,
      "sharpe": 2.3460457135293344,
      "sortino": 8.644345510703191,
      "total_return": 0.03074547333254274,
      "volatility": 0.13857794090179273
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.34102193475995635,
      "calmar": 23.46538296553558,
      "max_drawdown": 0.014532979719991235,
      "sharpe": 2.413336550920186,
      "sortino": 8.892288355446821,
      "total_return": 0.028536904512724837,
      "volatility": 0.12472014681161346
    },
    "plus_10pct_signal": {
      "cagr": 0.4284992547612956,
      "calmar": 24.153884434065883,
      "max_drawdown": 0.017740386890191195,
      "sharpe": 2.413336550920186,
      "sortino": 8.892288355446821,
      "total_return": 0.03478829746898393,
      "volatility": 0.152435734991972
    }
  }
}
```
