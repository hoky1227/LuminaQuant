# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/regime_breakout_1h_trend_ls_48_0.70.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 15.46%
- market_neutral: 50.00%
- trend: 34.54%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 50.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 27.01% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 15.46% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 7.52% | 3.539 | 8.61% | 2.009 | 8.08% |

## Portfolio metrics

- Fit (val): {"cagr": 0.4358763741165683, "calmar": 42.00953068143358, "max_drawdown": 0.01037565445379296, "sharpe": 4.3156109289398525, "sortino": 7.863328975299037, "total_return": 0.03120304742932789, "volatility": 0.08467255695855017}
- Report (oos): {"cagr": 0.7706222643672882, "calmar": 33.227865807590156, "max_drawdown": 0.023192048169138113, "sharpe": 3.045079692337286, "sortino": 9.825325478375213, "total_return": 0.05631366122077064, "volatility": 0.19361405831160264}
- Train: {"cagr": 0.0038255924352317283, "calmar": 0.04763696852128701, "max_drawdown": 0.08030721840585275, "sharpe": 0.08744785325104394, "sortino": 0.17514849993012693, "total_return": 0.0038255924352317283, "volatility": 0.09687332710774875}
- Val: {"cagr": 0.4358763741165683, "calmar": 42.00953068143358, "max_drawdown": 0.01037565445379296, "sharpe": 4.3156109289398525, "sortino": 7.863328975299037, "total_return": 0.03120304742932789, "volatility": 0.08467255695855017}
- OOS: {"cagr": 0.7706222643672882, "calmar": 33.227865807590156, "max_drawdown": 0.023192048169138113, "sharpe": 3.045079692337286, "sortino": 9.825325478375213, "total_return": 0.05631366122077064, "volatility": 0.19361405831160264}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.7604970380719809,
      "calmar": 32.63935698319053,
      "max_drawdown": 0.023300000623898365,
      "sharpe": 3.015414829917091,
      "sortino": 9.72960813827305,
      "total_return": 0.055732933975360055,
      "volatility": 0.19361405831160264
    },
    "x3": {
      "cagr": 0.7504295547006095,
      "calmar": 32.05875712491688,
      "max_drawdown": 0.023407942852449404,
      "sharpe": 2.9857499674968966,
      "sortino": 9.633890798170887,
      "total_return": 0.055152516877252644,
      "volatility": 0.1936140583116026
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.6749856534552932,
      "calmar": 32.31365701586437,
      "max_drawdown": 0.02088855659772304,
      "sharpe": 3.0450796923372865,
      "sortino": 9.825325478375214,
      "total_return": 0.0507043145710111,
      "volatility": 0.1742526524804424
    },
    "plus_10pct_signal": {
      "cagr": 0.8710683224376499,
      "calmar": 34.17018332663325,
      "max_drawdown": 0.0254920587961468,
      "sharpe": 3.0450796923372865,
      "sortino": 9.825325478375213,
      "total_return": 0.061917524737106655,
      "volatility": 0.21297546414276292
    }
  }
}
```
