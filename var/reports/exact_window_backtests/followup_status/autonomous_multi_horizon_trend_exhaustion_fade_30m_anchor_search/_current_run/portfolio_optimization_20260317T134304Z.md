# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_multi_horizon_trend_exhaustion_fade_30m_anchor_latest.json`
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
| 2 | multi_horizon_trend_exhaustion_fade_30m_guarded_lo_24_2.0 | MultiHorizonTrendExhaustionFadeStrategy | mean_reversion | 30m | 30.00% | -3.719 | -0.28% | 0.890 | 0.09% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.44% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 14.56% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.25389819145892933, "calmar": 36.81451624858474, "max_drawdown": 0.006896686886893155, "sharpe": 4.348400064486178, "sortino": 8.349125671219138, "total_return": 0.01940219213438321, "volatility": 0.05235284007173052}
- Report (oos): {"cagr": 0.42443195842118175, "calmar": 33.61344389847897, "max_drawdown": 0.012626851318867316, "sharpe": 3.1090923078449637, "sortino": 11.320271538731875, "total_return": 0.034505411033004973, "volatility": 0.1159106274725193}
- Train: {"cagr": 0.0126583713722912, "calmar": 0.22556781502627674, "max_drawdown": 0.05611780816698786, "sharpe": 0.23237505478022286, "sortino": 0.4559649200245854, "total_return": 0.0126583713722912, "volatility": 0.06247579069398485}
- Val: {"cagr": 0.25389819145892933, "calmar": 36.81451624858474, "max_drawdown": 0.006896686886893155, "sharpe": 4.348400064486178, "sortino": 8.349125671219138, "total_return": 0.01940219213438321, "volatility": 0.05235284007173052}
- OOS: {"cagr": 0.42443195842118175, "calmar": 33.61344389847897, "max_drawdown": 0.012626851318867316, "sharpe": 3.1090923078449637, "sortino": 11.320271538731875, "total_return": 0.034505411033004973, "volatility": 0.1159106274725193}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.41964440278288273,
      "calmar": 33.066879225221285,
      "max_drawdown": 0.012690777376499596,
      "sharpe": 3.0800192227657313,
      "sortino": 11.214415814623868,
      "total_return": 0.03417149226193095,
      "volatility": 0.1159106274725193
    },
    "x3": {
      "cagr": 0.4148728943140134,
      "calmar": 32.527060456456326,
      "max_drawdown": 0.01275469988655753,
      "sharpe": 3.0509461376864975,
      "sortino": 11.108560090515859,
      "total_return": 0.033837678194873044,
      "volatility": 0.1159106274725193
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.37572683748189917,
      "calmar": 33.04747085347576,
      "max_drawdown": 0.01136930687215909,
      "sharpe": 3.109092307844963,
      "sortino": 11.320271538731873,
      "total_return": 0.03105993301179799,
      "volatility": 0.10431956472526738
    },
    "plus_10pct_signal": {
      "cagr": 0.4746723192959368,
      "calmar": 34.19027351228677,
      "max_drawdown": 0.013883255982885201,
      "sharpe": 3.1090923078449646,
      "sortino": 11.320271538731877,
      "total_return": 0.037949642219640145,
      "volatility": 0.12750169021977123
    }
  }
}
```
