# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_topcap_exec_tightstop_anchor_latest.json`
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
| 2 | topcap_tsmom_1h_exec_tightstop_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 33.33% | 1.641 | 1.84% | 1.480 | 3.28% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 33.33% | 3.880 | 1.61% | 4.557 | 7.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.38926483974233705, "calmar": 33.906282399899105, "max_drawdown": 0.011480611031054688, "sharpe": 3.7297601330687606, "sortino": 7.1384641352936375, "total_return": 0.028316842357536842, "volatility": 0.08921760057741959}
- Report (oos): {"cagr": 0.5880374242065471, "calmar": 23.738516926896494, "max_drawdown": 0.024771447433612925, "sharpe": 2.691070344452474, "sortino": 9.806508896680109, "total_return": 0.045347339420796384, "volatility": 0.17754510863999792}
- Train: {"cagr": 0.008194558407396135, "calmar": 0.0902345072520852, "max_drawdown": 0.09081402067729216, "sharpe": 0.1305287455588169, "sortino": 0.25986501510342846, "total_return": 0.008194558407396135, "volatility": 0.10208055818422056}
- Val: {"cagr": 0.38926483974233705, "calmar": 33.906282399899105, "max_drawdown": 0.011480611031054688, "sharpe": 3.7297601330687606, "sortino": 7.1384641352936375, "total_return": 0.028316842357536842, "volatility": 0.08921760057741959}
- OOS: {"cagr": 0.5880374242065471, "calmar": 23.738516926896494, "max_drawdown": 0.024771447433612925, "sharpe": 2.691070344452474, "sortino": 9.806508896680109, "total_return": 0.045347339420796384, "volatility": 0.17754510863999792}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5789251548458312,
      "calmar": 23.26908024103416,
      "max_drawdown": 0.02487958908770782,
      "sharpe": 2.6586187901249803,
      "sortino": 9.688252435313462,
      "total_return": 0.04477066486338521,
      "volatility": 0.17754510863999792
    },
    "x3": {
      "cagr": 0.5698650298787438,
      "calmar": 22.80580298313876,
      "max_drawdown": 0.024987720463079843,
      "sharpe": 2.626167235797488,
      "sortino": 9.569995973946824,
      "total_return": 0.04419429934809371,
      "volatility": 0.17754510863999792
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5183109194610287,
      "calmar": 23.226077245271302,
      "max_drawdown": 0.022315904402950948,
      "sharpe": 2.6910703444524757,
      "sortino": 9.806508896680112,
      "total_return": 0.04085624804391008,
      "volatility": 0.15979059777599816
    },
    "plus_10pct_signal": {
      "cagr": 0.6604795171444615,
      "calmar": 24.262519154532974,
      "max_drawdown": 0.02722221517632739,
      "sharpe": 2.691070344452474,
      "sortino": 9.806508896680105,
      "total_return": 0.04982831746850436,
      "volatility": 0.19529961950399777
    }
  }
}
```
