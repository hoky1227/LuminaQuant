# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/perp_crowding_carry_30m_0.25_0.08.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- carry: 0.52%
- cross_sectional: 29.48%
- market_neutral: 35.00%
- trend: 35.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 35.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 35.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 29.48% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | perp_crowding_carry_30m_0.25_0.08 | PerpCrowdingCarryStrategy | carry | 30m | 0.52% | 0.000 | 0.00% | 3.438 | 0.06% |

## Portfolio metrics

- Fit (val): {"cagr": 0.2579619693967443, "calmar": 23.91400033820193, "max_drawdown": 0.010787068903091779, "sharpe": 3.2880735574832696, "sortino": 5.300704553152284, "total_return": 0.028697715626928577, "volatility": 0.07055493942923734}
- Report (oos): {"cagr": 0.5929981072239023, "calmar": 26.047277438264167, "max_drawdown": 0.022766222252187163, "sharpe": 2.783347493839723, "sortino": 10.227123299392456, "total_return": 0.04566002226043753, "volatility": 0.17247880912164612}
- Train: {"cagr": 0.01428926624765503, "calmar": 0.16554217879964925, "max_drawdown": 0.08631797860380286, "sharpe": 0.19802899458684492, "sortino": 0.38559566916416127, "total_return": 0.015472767825372102, "volatility": 0.09354890102432595}
- Val: {"cagr": 0.2579619693967443, "calmar": 23.91400033820193, "max_drawdown": 0.010787068903091779, "sharpe": 3.2880735574832696, "sortino": 5.300704553152284, "total_return": 0.028697715626928577, "volatility": 0.07055493942923734}
- OOS: {"cagr": 0.5929981072239023, "calmar": 26.047277438264167, "max_drawdown": 0.022766222252187163, "sharpe": 2.783347493839723, "sortino": 10.227123299392456, "total_return": 0.04566002226043753, "volatility": 0.17247880912164612}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5842894111789276,
      "calmar": 25.548937150512163,
      "max_drawdown": 0.02286942144547155,
      "sharpe": 2.7515255863924004,
      "sortino": 10.110196982500327,
      "total_return": 0.045110507650378606,
      "volatility": 0.17247880912164612
    },
    "x3": {
      "cagr": 0.5756281946128952,
      "calmar": 25.05715119464389,
      "max_drawdown": 0.02297261129732653,
      "sharpe": 2.7197036789450797,
      "sortino": 9.9932706656082,
      "total_return": 0.04456127357362094,
      "volatility": 0.17247880912164612
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5224667198993833,
      "calmar": 25.476809063806467,
      "max_drawdown": 0.020507541528881013,
      "sharpe": 2.7833474938397234,
      "sortino": 10.227123299392455,
      "total_return": 0.04112909752847527,
      "volatility": 0.1552309282094815
    },
    "plus_10pct_signal": {
      "cagr": 0.666335330769587,
      "calmar": 26.631112860901535,
      "max_drawdown": 0.025020934508068082,
      "sharpe": 2.7833474938397234,
      "sortino": 10.227123299392455,
      "total_return": 0.05018276815867173,
      "volatility": 0.18972669003381074
    }
  }
}
```
