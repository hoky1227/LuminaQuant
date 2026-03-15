# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/rolling_breakout_30m_guarded_ls_64_0.002.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 30.00%
- market_neutral: 35.00%
- trend: 35.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 35.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 35.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 30.00% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.39689564287984336, "calmar": 36.39272390184336, "max_drawdown": 0.010905906464993675, "sharpe": 3.9442853978167314, "sortino": 7.63789659684106, "total_return": 0.02879535344798123, "volatility": 0.08567971072826848}
- Report (oos): {"cagr": 0.5956439308571313, "calmar": 25.806197196162444, "max_drawdown": 0.023081429872422565, "sharpe": 2.7732278231783765, "sortino": 10.18217745245587, "total_return": 0.045826434574331554, "volatility": 0.17378433295630907}
- Train: {"cagr": 0.01490248610773226, "calmar": 0.1710425492907276, "max_drawdown": 0.08712736198992177, "sharpe": 0.19924314792908948, "sortino": 0.40332681119889685, "total_return": 0.01490248610773226, "volatility": 0.09824892741327707}
- Val: {"cagr": 0.39689564287984336, "calmar": 36.39272390184336, "max_drawdown": 0.010905906464993675, "sharpe": 3.9442853978167314, "sortino": 7.63789659684106, "total_return": 0.02879535344798123, "volatility": 0.08567971072826848}
- OOS: {"cagr": 0.5956439308571313, "calmar": 25.806197196162444, "max_drawdown": 0.023081429872422565, "sharpe": 2.7732278231783765, "sortino": 10.18217745245587, "total_return": 0.045826434574331554, "volatility": 0.17378433295630907}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.586831883047144,
      "calmar": 25.310127336928822,
      "max_drawdown": 0.023185655103003966,
      "sharpe": 2.7413221314031513,
      "sortino": 10.065032581528394,
      "total_return": 0.04527121812161039,
      "volatility": 0.17378433295630905
    },
    "x3": {
      "cagr": 0.5780683677422189,
      "calmar": 24.820591434217153,
      "max_drawdown": 0.023289870802405854,
      "sharpe": 2.709416439627926,
      "sortino": 9.947887710600925,
      "total_return": 0.044716288008569416,
      "volatility": 0.17378433295630907
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5247711507835096,
      "calmar": 25.239374503173888,
      "max_drawdown": 0.020791765291866438,
      "sharpe": 2.7732278231783773,
      "sortino": 10.18217745245587,
      "total_return": 0.04128010508013924,
      "volatility": 0.15640589966067817
    },
    "plus_10pct_signal": {
      "cagr": 0.6693416432315282,
      "calmar": 26.38630676144622,
      "max_drawdown": 0.025367007565057276,
      "sharpe": 2.7732278231783773,
      "sortino": 10.18217745245587,
      "total_return": 0.05036430210315257,
      "volatility": 0.19116276625193998
    }
  }
}
```
