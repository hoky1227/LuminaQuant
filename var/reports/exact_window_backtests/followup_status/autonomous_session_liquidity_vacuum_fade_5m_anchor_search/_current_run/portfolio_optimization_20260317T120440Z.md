# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_session_liquidity_vacuum_fade_5m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 14.56%
- market_neutral: 30.00%
- mean_reversion: 30.00%
- trend: 25.44%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | session_liquidity_vacuum_fade_5m_utc_guarded_lo_64_0.008 | SessionLiquidityVacuumFadeStrategy | mean_reversion | 5m | 30.00% | 0.370 | 0.04% | -1.611 | -1.38% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.44% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 14.56% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.26846592534406577, "calmar": 43.82652538124353, "max_drawdown": 0.006125649318733384, "sharpe": 4.609965030881414, "sortino": 9.032899740903595, "total_return": 0.020402759848211938, "volatility": 0.05188451045849369}
- Report (oos): {"cagr": 0.36019690091849155, "calmar": 25.173173844348398, "max_drawdown": 0.014308759918223779, "sharpe": 2.674804893956267, "sortino": 8.716656909988613, "total_return": 0.029938113277242495, "volatility": 0.11753208073719748}
- Train: {"cagr": 0.004921743184571614, "calmar": 0.08336529365509358, "max_drawdown": 0.05903827562743669, "sharpe": 0.10967181710424874, "sortino": 0.2064090903727245, "total_return": 0.004921743184571614, "volatility": 0.062404878595758634}
- Val: {"cagr": 0.26846592534406577, "calmar": 43.82652538124353, "max_drawdown": 0.006125649318733384, "sharpe": 4.609965030881414, "sortino": 9.032899740903595, "total_return": 0.020402759848211938, "volatility": 0.05188451045849369}
- OOS: {"cagr": 0.36019690091849155, "calmar": 25.173173844348398, "max_drawdown": 0.014308759918223779, "sharpe": 2.674804893956267, "sortino": 8.716656909988613, "total_return": 0.029938113277242495, "volatility": 0.11753208073719748}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.35569875216610214,
      "calmar": 24.750189624364523,
      "max_drawdown": 0.014371556645204286,
      "sharpe": 2.6465982771062406,
      "sortino": 8.624737158298041,
      "total_return": 0.029611022375364726,
      "volatility": 0.11753208073719748
    },
    "x3": {
      "cagr": 0.3512154380962842,
      "calmar": 24.331919309355367,
      "max_drawdown": 0.014434349943008629,
      "sharpe": 2.618391660256215,
      "sortino": 8.532817406607473,
      "total_return": 0.02928403238486843,
      "volatility": 0.11753208073719748
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.3197809702790262,
      "calmar": 24.819836502073944,
      "max_drawdown": 0.012884088509298008,
      "sharpe": 2.6748048939562667,
      "sortino": 8.716656909988613,
      "total_return": 0.026963416653240424,
      "volatility": 0.10577887266347774
    },
    "plus_10pct_signal": {
      "cagr": 0.40166674614771214,
      "calmar": 25.531739264187788,
      "max_drawdown": 0.01573205577542114,
      "sharpe": 2.6748048939562667,
      "sortino": 8.716656909988613,
      "total_return": 0.03290844276111882,
      "volatility": 0.12928528881091725
    }
  }
}
```
