# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_session_gated_residual_basket_reversion_5m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 44.56%
- market_neutral: 30.00%
- trend: 25.44%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | session_gated_residual_basket_reversion_5m_resid_btc_guarded_lo_80_2.00 | SessionGatedResidualBasketReversionStrategy | cross_sectional | 5m | 30.00% | 2.635 | 1.00% | -1.348 | -0.84% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.44% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 14.56% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.31217148096392977, "calmar": 70.49544835153175, "max_drawdown": 0.004428250167404557, "sharpe": 5.491680313882841, "sortino": 9.522365980774026, "total_return": 0.023342754441814106, "volatility": 0.04970759402893009}
- Report (oos): {"cagr": 0.38505668761991196, "calmar": 30.63690693357752, "max_drawdown": 0.012568393031801017, "sharpe": 2.845438475179226, "sortino": 12.44811723032853, "total_return": 0.03172839050448606, "volatility": 0.11682563271530529}
- Train: {"cagr": -0.029472077050105572, "calmar": -0.34623167999875437, "max_drawdown": 0.08512241586388514, "sharpe": -0.44454645072084653, "sortino": -0.8131235845262501, "total_return": -0.029472077050105572, "volatility": 0.06287145770687834}
- Val: {"cagr": 0.31217148096392977, "calmar": 70.49544835153175, "max_drawdown": 0.004428250167404557, "sharpe": 5.491680313882841, "sortino": 9.522365980774026, "total_return": 0.023342754441814106, "volatility": 0.04970759402893009}
- OOS: {"cagr": 0.38505668761991196, "calmar": 30.63690693357752, "max_drawdown": 0.012568393031801017, "sharpe": 2.845438475179226, "sortino": 12.44811723032853, "total_return": 0.03172839050448606, "volatility": 0.11682563271530529}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.38027881081903514,
      "calmar": 30.09962577223501,
      "max_drawdown": 0.01263400461177222,
      "sharpe": 2.8158340067028673,
      "sortino": 12.318604714999212,
      "total_return": 0.03138657940850553,
      "volatility": 0.11682563271530529
    },
    "x3": {
      "cagr": 0.3755173707295256,
      "calmar": 29.569199222784015,
      "max_drawdown": 0.012699612454847187,
      "sharpe": 2.7862295382265083,
      "sortino": 12.189092199669895,
      "total_return": 0.031044878319815616,
      "volatility": 0.11682563271530529
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.3414619414102511,
      "calmar": 30.17275074492417,
      "max_drawdown": 0.011316897961903383,
      "sharpe": 2.8454384751792268,
      "sortino": 12.448117230328531,
      "total_return": 0.028569260451859257,
      "volatility": 0.10514306944377474
    },
    "plus_10pct_signal": {
      "cagr": 0.4298827726546408,
      "calmar": 31.108763411053,
      "max_drawdown": 0.01381870333366908,
      "sharpe": 2.8454384751792268,
      "sortino": 12.448117230328531,
      "total_return": 0.03488435720370897,
      "volatility": 0.1285081959868358
    }
  }
}
```
