# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_topcap_exec_takeprofit_anchor_latest.json`
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
| 2 | topcap_tsmom_1h_exec_takeprofit_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 33.33% | 1.641 | 1.84% | 1.667 | 3.61% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 33.33% | 3.880 | 1.61% | 4.557 | 7.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.38926483974233705, "calmar": 33.906282399899105, "max_drawdown": 0.011480611031054688, "sharpe": 3.7297601330687606, "sortino": 7.1384641352936375, "total_return": 0.028316842357536842, "volatility": 0.08921760057741959}
- Report (oos): {"cagr": 0.6059790574368882, "calmar": 25.556863041205037, "max_drawdown": 0.023711010872495386, "sharpe": 2.816218162656632, "sortino": 10.43520147979752, "total_return": 0.046474093491835955, "volatility": 0.1734102981280905}
- Train: {"cagr": 0.009734620016889961, "calmar": 0.10592565260771411, "max_drawdown": 0.09190049602942951, "sharpe": 0.14528972149222494, "sortino": 0.29130824105694963, "total_return": 0.009734620016889961, "volatility": 0.10251240032377607}
- Val: {"cagr": 0.38926483974233705, "calmar": 33.906282399899105, "max_drawdown": 0.011480611031054688, "sharpe": 3.7297601330687606, "sortino": 7.1384641352936375, "total_return": 0.028316842357536842, "volatility": 0.08921760057741959}
- OOS: {"cagr": 0.6059790574368882, "calmar": 25.556863041205037, "max_drawdown": 0.023711010872495386, "sharpe": 2.816218162656632, "sortino": 10.43520147979752, "total_return": 0.046474093491835955, "volatility": 0.1734102981280905}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5965389165433073,
      "calmar": 25.04161110035864,
      "max_drawdown": 0.023821906432160977,
      "sharpe": 2.782178384304534,
      "sortino": 10.309070645850797,
      "total_return": 0.04588266932936991,
      "volatility": 0.1734102981280905
    },
    "x3": {
      "cagr": 0.5871541148019679,
      "calmar": 24.533457465376824,
      "max_drawdown": 0.023932791194661296,
      "sharpe": 2.748138605952437,
      "sortino": 10.182939811904077,
      "total_return": 0.045291569870101744,
      "volatility": 0.1734102981280905
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5336533349615225,
      "calmar": 24.98386145154447,
      "max_drawdown": 0.021359922123989072,
      "sharpe": 2.8162181626566336,
      "sortino": 10.435201479797522,
      "total_return": 0.04186022386522992,
      "volatility": 0.15606926831528142
    },
    "plus_10pct_signal": {
      "cagr": 0.6812432457615369,
      "calmar": 26.143669467096682,
      "max_drawdown": 0.026057675133130065,
      "sharpe": 2.816218162656632,
      "sortino": 10.435201479797522,
      "total_return": 0.051080082346799616,
      "volatility": 0.19075132794089958
    }
  }
}
```
