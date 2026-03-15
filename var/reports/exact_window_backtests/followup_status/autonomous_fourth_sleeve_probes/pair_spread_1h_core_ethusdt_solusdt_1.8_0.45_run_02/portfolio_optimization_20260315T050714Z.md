# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/pair_spread_1h_core_ethusdt_solusdt_1.8_0.45.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 8.45%
- market_neutral: 76.78%
- trend: 14.77%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_ethusdt_solusdt_1.8_0.45 | PairSpreadZScoreStrategy | market_neutral | 1h | 38.84% | 4.195 | 3.07% | 1.613 | 1.69% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 37.94% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 14.77% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 8.45% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.37107189328634926, "calmar": 53.97677920093407, "max_drawdown": 0.006874657932163686, "sharpe": 5.945351650814679, "sortino": 8.424767947883353, "total_return": 0.027166227564807244, "volatility": 0.05333657594765831}
- Report (oos): {"cagr": 0.5267920783725972, "calmar": 40.47857484519597, "max_drawdown": 0.013014096479118442, "sharpe": 3.743731574695107, "sortino": 12.488045212760385, "total_return": 0.04141236515658919, "volatility": 0.11478958703696437}
- Train: {"cagr": 0.06211661778854016, "calmar": 0.9229369409828354, "max_drawdown": 0.0673032089520571, "sharpe": 0.7012025628191155, "sortino": 1.2998884471619296, "total_return": 0.06211661778854016, "volatility": 0.09189732840541374}
- Val: {"cagr": 0.37107189328634926, "calmar": 53.97677920093407, "max_drawdown": 0.006874657932163686, "sharpe": 5.945351650814679, "sortino": 8.424767947883353, "total_return": 0.027166227564807244, "volatility": 0.05333657594765831}
- OOS: {"cagr": 0.5267920783725972, "calmar": 40.47857484519597, "max_drawdown": 0.013014096479118442, "sharpe": 3.743731574695107, "sortino": 12.488045212760385, "total_return": 0.04141236515658919, "volatility": 0.11478958703696437}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5203872475476465,
      "calmar": 39.81190768861671,
      "max_drawdown": 0.013071145739053325,
      "sharpe": 3.707068291444555,
      "sortino": 12.365746717329815,
      "total_return": 0.04099265385347706,
      "volatility": 0.11478958703696437
    },
    "x3": {
      "cagr": 0.5140092114227395,
      "calmar": 39.153083477990165,
      "max_drawdown": 0.013128192360933433,
      "sharpe": 3.670405008194004,
      "sortino": 12.243448221899243,
      "total_return": 0.04057310687173166,
      "volatility": 0.11478958703696437
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.46438751992585314,
      "calmar": 39.62857840775266,
      "max_drawdown": 0.011718500601954562,
      "sharpe": 3.7437315746951074,
      "sortino": 12.488045212760385,
      "total_return": 0.03725329650730824,
      "volatility": 0.10331062833326796
    },
    "plus_10pct_signal": {
      "cagr": 0.5916517463737125,
      "calmar": 41.34994773294063,
      "max_drawdown": 0.014308403729912933,
      "sharpe": 3.743731574695107,
      "sortino": 12.488045212760383,
      "total_return": 0.0455752452914675,
      "volatility": 0.12626854574066085
    }
  }
}
```
