# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_regime_conditioned_composite_trend_30m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 15.00%
- market_neutral: 30.00%
- trend: 55.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 28.81% | 5.033 | 6.29% | 1.625 | 2.21% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 26.19% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 15.00% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.5662524421307336, "calmar": 61.983712269153074, "max_drawdown": 0.009135503850945303, "sharpe": 5.397871148982835, "sortino": 12.584173991069363, "total_return": 0.038842964761541365, "volatility": 0.08380081254312782}
- Report (oos): {"cagr": 0.5204642328380453, "calmar": 33.83084953250547, "max_drawdown": 0.0153843087013813, "sharpe": 2.750412946924659, "sortino": 10.277143843038981, "total_return": 0.04099770821657733, "volatility": 0.15668097139649484}
- Train: {"cagr": 0.04402210959665953, "calmar": 0.5775469162202936, "max_drawdown": 0.07622256886896472, "sharpe": 0.5084092533100841, "sortino": 1.1497754225571943, "total_return": 0.04402210959665953, "volatility": 0.09318649479900046}
- Val: {"cagr": 0.5662524421307336, "calmar": 61.983712269153074, "max_drawdown": 0.009135503850945303, "sharpe": 5.397871148982835, "sortino": 12.584173991069363, "total_return": 0.038842964761541365, "volatility": 0.08380081254312782}
- OOS: {"cagr": 0.5204642328380453, "calmar": 33.83084953250547, "max_drawdown": 0.0153843087013813, "sharpe": 2.750412946924659, "sortino": 10.277143843038981, "total_return": 0.04099770821657733, "volatility": 0.15668097139649484}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5136338087225683,
      "calmar": 33.20276593691182,
      "max_drawdown": 0.015469609059031941,
      "sharpe": 2.721644606946821,
      "sortino": 10.16964857822492,
      "total_return": 0.040548363132147935,
      "volatility": 0.15668097139649484
    },
    "x3": {
      "cagr": 0.5068339854173778,
      "calmar": 32.44246718227171,
      "max_drawdown": 0.015622547526050634,
      "sharpe": 2.692876266968983,
      "sortino": 10.062153313410857,
      "total_return": 0.04009920646713683,
      "volatility": 0.15668097139649484
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.459606483680542,
      "calmar": 33.17965493617155,
      "max_drawdown": 0.013852057369635018,
      "sharpe": 2.7504129469246585,
      "sortino": 10.277143843038978,
      "total_return": 0.036928083540672896,
      "volatility": 0.14101287425684533
    },
    "plus_10pct_signal": {
      "cagr": 0.5834973483462009,
      "calmar": 34.49546693040042,
      "max_drawdown": 0.016915189161622046,
      "sharpe": 2.7504129469246594,
      "sortino": 10.277143843038983,
      "total_return": 0.04506039346426127,
      "volatility": 0.17234906853614432
    }
  }
}
```
