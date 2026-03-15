# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/regime_breakout_30m_trend_ls_64_0.72.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 19.99%
- market_neutral: 35.00%
- trend: 45.01%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 35.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 34.92% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 19.99% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | regime_breakout_30m_trend_ls_64_0.72 | RegimeBreakoutCandidateStrategy | trend | 30m | 10.09% | 1.477 | 2.00% | 1.070 | 1.27% |

## Portfolio metrics

- Fit (val): {"cagr": 0.26700402119620814, "calmar": 30.139599639809145, "max_drawdown": 0.008858910681863952, "sharpe": 3.4806023941894453, "sortino": 7.876630126792643, "total_return": 0.029606461582011523, "volatility": 0.06867460824883484}
- Report (oos): {"cagr": 0.5657528056732466, "calmar": 33.34695512506299, "max_drawdown": 0.0169656510932249, "sharpe": 2.9758487889923244, "sortino": 11.065956253723574, "total_return": 0.04393170475024655, "volatility": 0.15459479353870048}
- Train: {"cagr": 0.026995942922836136, "calmar": 0.35342445648364584, "max_drawdown": 0.07638391296241642, "sharpe": 0.37790845119880234, "sortino": 0.7687366108463473, "total_return": 0.029246934806941294, "volatility": 0.07859758522326096}
- Val: {"cagr": 0.26700402119620814, "calmar": 30.139599639809145, "max_drawdown": 0.008858910681863952, "sharpe": 3.4806023941894453, "sortino": 7.876630126792643, "total_return": 0.029606461582011523, "volatility": 0.06867460824883484}
- OOS: {"cagr": 0.5657528056732466, "calmar": 33.34695512506299, "max_drawdown": 0.0169656510932249, "sharpe": 2.9758487889923244, "sortino": 11.065956253723574, "total_return": 0.04393170475024655, "volatility": 0.15459479353870048}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5573668698136993,
      "calmar": 32.65708119585838,
      "max_drawdown": 0.017067259210060248,
      "sharpe": 2.9410698734382987,
      "sortino": 10.93662778801107,
      "total_return": 0.043394265475315885,
      "volatility": 0.15459479353870048
    },
    "x3": {
      "cagr": 0.5490257253603488,
      "calmar": 31.97799847702903,
      "max_drawdown": 0.017168858324723923,
      "sharpe": 2.9062909578842726,
      "sortino": 10.807299322298562,
      "total_return": 0.04285709498425505,
      "volatility": 0.15459479353870048
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.4986499831553637,
      "calmar": 32.63743987884455,
      "max_drawdown": 0.015278465008482067,
      "sharpe": 2.975848788992324,
      "sortino": 11.06595625372357,
      "total_return": 0.03955618256750615,
      "volatility": 0.13913531418483044
    },
    "plus_10pct_signal": {
      "cagr": 0.6354914160111442,
      "calmar": 34.07321934617064,
      "max_drawdown": 0.01865075940006722,
      "sharpe": 2.9758487889923244,
      "sortino": 11.065956253723574,
      "total_return": 0.04830297989483645,
      "volatility": 0.17005427289257055
    }
  }
}
```
