# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_cross_sectional_1h_validation_filtered_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 50.00%
- market_neutral: 50.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | market_neutral | 1h | 50.00% | 4.587 | 1.71% | 7.010 | 11.06% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 50.00% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.23494032367288487, "calmar": 12.492840203645637, "max_drawdown": 0.018805997662911356, "sharpe": 2.407873028538905, "sortino": 4.337596997610434, "total_return": 0.018084042774804576, "volatility": 0.08925975134071028}
- Report (oos): {"cagr": 1.0517698907502604, "calmar": 29.61426439536446, "max_drawdown": 0.03551565140057622, "sharpe": 3.4799981176665327, "sortino": 11.280737711131295, "total_return": 0.07134696771914362, "volatility": 0.21292851936329954}
- Train: {"cagr": -0.04459678662771138, "calmar": -0.3588394863605623, "max_drawdown": 0.12428059988610196, "sharpe": -0.3253705574784481, "sortino": -0.4979938724918525, "total_return": -0.04459678662771138, "volatility": 0.118664356335986}
- Val: {"cagr": 0.23494032367288487, "calmar": 12.492840203645637, "max_drawdown": 0.018805997662911356, "sharpe": 2.407873028538905, "sortino": 4.337596997610434, "total_return": 0.018084042774804576, "volatility": 0.08925975134071028}
- OOS: {"cagr": 1.0517698907502604, "calmar": 29.61426439536446, "max_drawdown": 0.03551565140057622, "sharpe": 3.4799981176665327, "sortino": 11.280737711131295, "total_return": 0.07134696771914362, "volatility": 0.21292851936329954}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 1.0381268442154372,
      "calmar": 29.128199483522987,
      "max_drawdown": 0.0356399249738274,
      "sharpe": 3.4486058420719945,
      "sortino": 11.178976728750364,
      "total_return": 0.07066179977233822,
      "volatility": 0.21292851936329954
    },
    "x3": {
      "cagr": 1.0245742688843982,
      "calmar": 28.64805318479949,
      "max_drawdown": 0.0357641848217467,
      "sharpe": 3.4172135664774554,
      "sortino": 11.077215746369435,
      "total_return": 0.06997705750361094,
      "volatility": 0.21292851936329954
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.9132441949001013,
      "calmar": 28.529693246188753,
      "max_drawdown": 0.03201030543930228,
      "sharpe": 3.4799981176665327,
      "sortino": 11.280737711131296,
      "total_return": 0.06418977025432548,
      "volatility": 0.1916356674269696
    },
    "plus_10pct_signal": {
      "cagr": 1.1993852469492263,
      "calmar": 30.744946001598738,
      "max_drawdown": 0.03901081000066997,
      "sharpe": 3.4799981176665327,
      "sortino": 11.280737711131296,
      "total_return": 0.07850810469110603,
      "volatility": 0.23422137129962955
    }
  }
}
```
