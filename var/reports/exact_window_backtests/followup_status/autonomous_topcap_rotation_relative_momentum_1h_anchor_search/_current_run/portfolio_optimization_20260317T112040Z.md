# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_topcap_rotation_relative_momentum_1h_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 42.34%
- market_neutral: 30.00%
- trend: 27.66%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 27.66% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_defensive_24_6_0.020 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 26.51% | 2.535 | 2.33% | -2.451 | -4.18% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 15.83% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.3821122239500103, "calmar": 37.05515062123353, "max_drawdown": 0.010311986796541328, "sharpe": 4.09846624658193, "sortino": 8.44413478526932, "total_return": 0.02786612859065074, "volatility": 0.0797428536678236}
- Report (oos): {"cagr": 0.27604081128916347, "calmar": 9.05694565278524, "max_drawdown": 0.030478355714133487, "sharpe": 1.5879176110681714, "sortino": 4.232813763645834, "total_return": 0.023649778038482783, "volatility": 0.16141647562424052}
- Train: {"cagr": -0.017201428626277382, "calmar": -0.19588590616413407, "max_drawdown": 0.08781350819524603, "sharpe": -0.13119969501776732, "sortino": -0.2471589387398093, "total_return": -0.017201428626277382, "volatility": 0.09681314920999819}
- Val: {"cagr": 0.3821122239500103, "calmar": 37.05515062123353, "max_drawdown": 0.010311986796541328, "sharpe": 4.09846624658193, "sortino": 8.44413478526932, "total_return": 0.02786612859065074, "volatility": 0.0797428536678236}
- OOS: {"cagr": 0.27604081128916347, "calmar": 9.05694565278524, "max_drawdown": 0.030478355714133487, "sharpe": 1.5879176110681714, "sortino": 4.232813763645834, "total_return": 0.023649778038482783, "volatility": 0.16141647562424052}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.269568246891738,
      "calmar": 8.813192869201057,
      "max_drawdown": 0.030586899764077802,
      "sharpe": 1.5563936974199133,
      "sortino": 4.148782416777295,
      "total_return": 0.023150736958656326,
      "volatility": 0.16141647562424047
    },
    "x3": {
      "cagr": 0.2631284243035261,
      "calmar": 8.572233620362875,
      "max_drawdown": 0.030695433180738196,
      "sharpe": 1.524869783771655,
      "sortino": 4.064751069908757,
      "total_return": 0.02265193221885009,
      "volatility": 0.16141647562424052
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.2467007741618108,
      "calmar": 8.983842005686187,
      "max_drawdown": 0.027460497858896593,
      "sharpe": 1.5879176110681719,
      "sortino": 4.232813763645833,
      "total_return": 0.021369017035062443,
      "volatility": 0.1452748280618164
    },
    "plus_10pct_signal": {
      "cagr": 0.30575304161429817,
      "calmar": 9.129794251814035,
      "max_drawdown": 0.03348958729859075,
      "sharpe": 1.5879176110681723,
      "sortino": 4.232813763645836,
      "total_return": 0.025911654121279826,
      "volatility": 0.17755812318666453
    }
  }
}
```
