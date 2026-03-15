# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_triplet_probes/regime_breakout_1h_trend_ls_48_0.70.json`
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
| 1 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 33.33% | 3.539 | 8.61% | 2.009 | 8.08% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 33.33% | 1.641 | 1.84% | 1.464 | 3.24% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 33.33% | 3.880 | 1.61% | 4.557 | 7.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.6019155828720497, "calmar": 21.247754194554826, "max_drawdown": 0.028328433083356308, "sharpe": 2.673797341588021, "sortino": 5.130370270115141, "total_return": 0.040831318930683924, "volatility": 0.18231374075189086}
- Report (oos): {"cagr": 0.9136935217381941, "calmar": 15.072769803787512, "max_drawdown": 0.06061882013938802, "sharpe": 2.133380178799276, "sortino": 5.68859244401946, "total_return": 0.0642137331949928, "volatility": 0.3282951695760418}
- Train: {"cagr": -0.1264289750583817, "calmar": -0.6431356277393679, "max_drawdown": 0.1965821354086409, "sharpe": -0.6250881241397281, "sortino": -1.2007459581872466, "total_return": -0.1264289750583817, "volatility": 0.18814628902503222}
- Val: {"cagr": 0.6019155828720497, "calmar": 21.247754194554826, "max_drawdown": 0.028328433083356308, "sharpe": 2.673797341588021, "sortino": 5.130370270115141, "total_return": 0.040831318930683924, "volatility": 0.18231374075189086}
- OOS: {"cagr": 0.9136935217381941, "calmar": 15.072769803787512, "max_drawdown": 0.06061882013938802, "sharpe": 2.133380178799276, "sortino": 5.68859244401946, "total_return": 0.0642137331949928, "volatility": 0.3282951695760418}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.8944636879361585,
      "calmar": 14.710922685697708,
      "max_drawdown": 0.060802691105553586,
      "sharpe": 2.102566931074662,
      "sortino": 5.60642986937672,
      "total_return": 0.06318361249953885,
      "volatility": 0.3282951695760418
    },
    "x3": {
      "cagr": 0.8754265617374706,
      "calmar": 14.354424562076883,
      "max_drawdown": 0.0609865312225939,
      "sharpe": 2.0717536833500487,
      "sortino": 5.5242672947339795,
      "total_return": 0.062154460454119986,
      "volatility": 0.3282951695760418
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.8015074013708172,
      "calmar": 14.659895238102683,
      "max_drawdown": 0.05467347401553124,
      "sharpe": 2.133380178799277,
      "sortino": 5.688592444019463,
      "total_return": 0.05806669924397956,
      "volatility": 0.29546565261843766
    },
    "plus_10pct_signal": {
      "cagr": 1.0309040901323914,
      "calmar": 15.493346760267706,
      "max_drawdown": 0.06653850237032832,
      "sharpe": 2.133380178799276,
      "sortino": 5.68859244401946,
      "total_return": 0.07029738509814853,
      "volatility": 0.3611246865336461
    }
  }
}
```
