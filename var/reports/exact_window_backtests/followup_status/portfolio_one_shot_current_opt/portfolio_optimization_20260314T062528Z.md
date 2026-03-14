# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 1

## Sleeve budgets

- cross_sectional: 35.00%
- trend: 65.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 35.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 35.00% | 1.641 | 1.84% | 1.464 | 3.24% |
| 3 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 30.00% | 3.539 | 8.61% | 2.009 | 8.08% |

## Portfolio metrics

- Fit (val): {"cagr": 0.77977446831369, "calmar": 26.466070466415825, "max_drawdown": 0.02946317509821439, "sharpe": 3.0028163085596984, "sortino": 6.19759686323781, "total_return": 0.050180317234329586, "volatility": 0.19841887691699528}
- Report (oos): {"cagr": 0.6289327837877341, "calmar": 10.713604745238294, "max_drawdown": 0.05870412421806637, "sharpe": 1.7067110916177235, "sortino": 4.340534394917453, "total_return": 0.04789913406446966, "volatility": 0.31322550443768743}
- Train: {"cagr": -0.10181746447818285, "calmar": -0.557572458709836, "max_drawdown": 0.18260848951143993, "sharpe": -0.4556062326536862, "sortino": -0.9377162272271935, "total_return": -0.10181746447818285, "volatility": 0.1946140050325783}
- Val: {"cagr": 0.77977446831369, "calmar": 26.466070466415825, "max_drawdown": 0.02946317509821439, "sharpe": 3.0028163085596984, "sortino": 6.19759686323781, "total_return": 0.050180317234329586, "volatility": 0.19841887691699528}
- OOS: {"cagr": 0.6289327837877341, "calmar": 10.713604745238294, "max_drawdown": 0.05870412421806637, "sharpe": 1.7067110916177235, "sortino": 4.340534394917453, "total_return": 0.04789913406446966, "volatility": 0.31322550443768743}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.6154905088903915,
      "calmar": 10.457710039116318,
      "max_drawdown": 0.058855189767950455,
      "sharpe": 1.6802241016158064,
      "sortino": 4.273172266853834,
      "total_return": 0.04706681494232523,
      "volatility": 0.31322550443768743
    },
    "x3": {
      "cagr": 0.6021588608161978,
      "calmar": 10.205004022712103,
      "max_drawdown": 0.059006234537099855,
      "sharpe": 1.6537371116138875,
      "sortino": 4.205810138790208,
      "total_return": 0.04623513803144319,
      "volatility": 0.31322550443768743
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5577235311474988,
      "calmar": 10.534798271043917,
      "max_drawdown": 0.05294107364926626,
      "sharpe": 1.7067110916177244,
      "sortino": 4.340534394917457,
      "total_return": 0.043417176460573526,
      "volatility": 0.2819029539939187
    },
    "plus_10pct_signal": {
      "cagr": 0.7018949057097308,
      "calmar": 10.891630274127456,
      "max_drawdown": 0.06444351194853248,
      "sharpe": 1.7067110916177237,
      "sortino": 4.340534394917457,
      "total_return": 0.05231130287567454,
      "volatility": 0.3445480548814563
    }
  }
}
```
