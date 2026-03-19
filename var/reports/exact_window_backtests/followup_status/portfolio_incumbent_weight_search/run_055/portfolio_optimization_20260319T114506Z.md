# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_incumbent_bundle_latest.json`
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
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 33.33% | 1.641 | 1.84% | 1.464 | 3.24% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 33.33% | 3.880 | 1.61% | 4.557 | 7.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.38926483974233705, "calmar": 33.906282399899105, "max_drawdown": 0.011480611031054688, "sharpe": 3.7297601330687606, "sortino": 7.1384641352936375, "total_return": 0.028316842357536842, "volatility": 0.08921760057741959}
- Report (oos): {"cagr": 0.5860273276766477, "calmar": 23.54479931984585, "max_drawdown": 0.024889884161496623, "sharpe": 2.6832221906378617, "sortino": 9.758223574528898, "total_return": 0.04522038710991727, "volatility": 0.177594981686411}
- Train: {"cagr": 0.00915566006509927, "calmar": 0.10111672542307319, "max_drawdown": 0.09054545651861168, "sharpe": 0.13980898803129524, "sortino": 0.27860873391017676, "total_return": 0.00915566006509927, "volatility": 0.1022196309368869}
- Val: {"cagr": 0.38926483974233705, "calmar": 33.906282399899105, "max_drawdown": 0.011480611031054688, "sharpe": 3.7297601330687606, "sortino": 7.1384641352936375, "total_return": 0.028316842357536842, "volatility": 0.08921760057741959}
- OOS: {"cagr": 0.5860273276766477, "calmar": 23.54479931984585, "max_drawdown": 0.024889884161496623, "sharpe": 2.6832221906378617, "sortino": 9.758223574528898, "total_return": 0.04522038710991727, "volatility": 0.177594981686411}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5769265606864289,
      "calmar": 23.078895296122592,
      "max_drawdown": 0.02499801456196027,
      "sharpe": 2.6507797495063583,
      "sortino": 9.640238342083633,
      "total_return": 0.04464378057290452,
      "volatility": 0.177594981686411
    },
    "x3": {
      "cagr": 0.5678778724496625,
      "calmar": 22.619088106708038,
      "max_drawdown": 0.025106134684591885,
      "sharpe": 2.6183373083748562,
      "sortino": 9.522253109638374,
      "total_return": 0.04406748304263819,
      "volatility": 0.17759498168641102
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5165821147901526,
      "calmar": 23.03834726089842,
      "max_drawdown": 0.022422707190759117,
      "sharpe": 2.6832221906378626,
      "sortino": 9.758223574528902,
      "total_return": 0.04074254429287327,
      "volatility": 0.15983548351776994
    },
    "plus_10pct_signal": {
      "cagr": 0.6581664088223917,
      "calmar": 24.062614930344377,
      "max_drawdown": 0.02735223959356159,
      "sharpe": 2.6832221906378617,
      "sortino": 9.758223574528898,
      "total_return": 0.04968799424432735,
      "volatility": 0.1953544798550521
    }
  }
}
```
