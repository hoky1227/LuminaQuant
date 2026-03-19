# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_single_asset_zscore_reversion_15m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 28.55%
- market_neutral: 30.00%
- mean_reversion: 11.45%
- trend: 30.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 30.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 28.55% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | mean_reversion_std_15m_balanced_ls_64_2.00 | MeanReversionStdStrategy | mean_reversion | 15m | 11.45% | -4.208 | -11.91% | -2.510 | -11.90% |

## Portfolio metrics

- Fit (val): {"cagr": 0.14391689831783205, "calmar": 17.34143608264216, "max_drawdown": 0.008299018468365782, "sharpe": 2.825831801746726, "sortino": 5.318179669353386, "total_return": 0.01148519586438601, "volatility": 0.04798432488807756}
- Report (oos): {"cagr": 0.3201956981323759, "calmar": 18.924785647105068, "max_drawdown": 0.016919383083282447, "sharpe": 2.4352730379440595, "sortino": 7.519913905364839, "total_return": 0.02699435733005795, "volatility": 0.11679548816973509}
- Train: {"cagr": -0.11203915823777533, "calmar": -0.7372503974597537, "max_drawdown": 0.1519689356883548, "sharpe": -1.66866052524752, "sortino": -2.6422459842400987, "total_return": -0.11203915823777533, "volatility": 0.0697481562970797}
- Val: {"cagr": 0.14391689831783205, "calmar": 17.34143608264216, "max_drawdown": 0.008299018468365782, "sharpe": 2.825831801746726, "sortino": 5.318179669353386, "total_return": 0.01148519586438601, "volatility": 0.04798432488807756}
- OOS: {"cagr": 0.3201956981323759, "calmar": 18.924785647105068, "max_drawdown": 0.016919383083282447, "sharpe": 2.4352730379440595, "sortino": 7.519913905364839, "total_return": 0.02699435733005795, "volatility": 0.11679548816973509}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.31258590684897936,
      "calmar": 18.35640125986332,
      "max_drawdown": 0.017028713985047572,
      "sharpe": 2.3857414288455923,
      "sortino": 7.366964552166596,
      "total_return": 0.026425226662259327,
      "volatility": 0.11679548816973509
    },
    "x3": {
      "cagr": 0.30501985997089887,
      "calmar": 17.797832102585105,
      "max_drawdown": 0.017138034464691643,
      "sharpe": 2.336209819747124,
      "sortino": 7.2140151989683545,
      "total_return": 0.02585640238349729,
      "volatility": 0.11679548816973509
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.2847876117937671,
      "calmar": 18.692341333572948,
      "max_drawdown": 0.015235523828267872,
      "sharpe": 2.4352730379440604,
      "sortino": 7.519913905364842,
      "total_return": 0.024320541274639895,
      "volatility": 0.10511593935276158
    },
    "plus_10pct_signal": {
      "cagr": 0.3564038360133932,
      "calmar": 19.16000295851923,
      "max_drawdown": 0.0186014499467978,
      "sharpe": 2.43527303794406,
      "sortino": 7.519913905364839,
      "total_return": 0.02966235859066213,
      "volatility": 0.1284750369867086
    }
  }
}
```
