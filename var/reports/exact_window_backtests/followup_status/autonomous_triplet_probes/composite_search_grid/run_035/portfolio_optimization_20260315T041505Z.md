# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_triplet_probes/composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 21.84%
- market_neutral: 40.00%
- trend: 38.16%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 40.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 38.16% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 21.84% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.4106217477538836, "calmar": 43.97010871665777, "max_drawdown": 0.009338656640580978, "sharpe": 4.506008168643512, "sortino": 8.897880221544982, "total_return": 0.029650098283141357, "volatility": 0.07702041490513785}
- Report (oos): {"cagr": 0.6255214671958853, "calmar": 33.587997438603615, "max_drawdown": 0.01862336295396272, "sharpe": 3.0160111918870154, "sortino": 11.106020128238706, "total_return": 0.04768850173997663, "volatility": 0.16551245554337365}
- Train: {"cagr": 0.028393153409329575, "calmar": 0.3605213295387117, "max_drawdown": 0.07875582131481296, "sharpe": 0.35652794603695875, "sortino": 0.7308744309383788, "total_return": 0.028393153409329575, "volatility": 0.08972023105829002}
- Val: {"cagr": 0.4106217477538836, "calmar": 43.97010871665777, "max_drawdown": 0.009338656640580978, "sharpe": 4.506008168643512, "sortino": 8.897880221544982, "total_return": 0.029650098283141357, "volatility": 0.07702041490513785}
- OOS: {"cagr": 0.6255214671958853, "calmar": 33.587997438603615, "max_drawdown": 0.01862336295396272, "sharpe": 3.0160111918870154, "sortino": 11.106020128238706, "total_return": 0.04768850173997663, "volatility": 0.16551245554337365}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.6174027775152988,
      "calmar": 32.98449166194245,
      "max_drawdown": 0.018717971580191395,
      "sharpe": 2.985720496412225,
      "sortino": 10.994479072109236,
      "total_return": 0.047185599992411786,
      "volatility": 0.16551245554337365
    },
    "x3": {
      "cagr": 0.6093245261968439,
      "calmar": 32.38921895473251,
      "max_drawdown": 0.01881257238862233,
      "sharpe": 2.955429800937434,
      "sortino": 10.882938015979768,
      "total_return": 0.04668293274874413,
      "volatility": 0.16551245554337365
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5502642023614155,
      "calmar": 32.80789571091822,
      "max_drawdown": 0.016772310153933212,
      "sharpe": 3.016011191887015,
      "sortino": 11.106020128238708,
      "total_return": 0.04293701809541828,
      "volatility": 0.1489612099890363
    },
    "plus_10pct_signal": {
      "cagr": 0.7039955252281314,
      "calmar": 34.388354149379374,
      "max_drawdown": 0.02047191680561533,
      "sharpe": 3.0160111918870167,
      "sortino": 11.106020128238713,
      "total_return": 0.052435780667339005,
      "volatility": 0.18206370109771106
    }
  }
}
```
