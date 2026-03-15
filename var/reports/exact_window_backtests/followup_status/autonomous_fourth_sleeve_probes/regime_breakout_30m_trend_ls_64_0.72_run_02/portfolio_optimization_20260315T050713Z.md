# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/regime_breakout_30m_trend_ls_64_0.72.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 15.38%
- market_neutral: 50.00%
- trend: 34.62%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 50.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 26.86% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 15.38% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | regime_breakout_30m_trend_ls_64_0.72 | RegimeBreakoutCandidateStrategy | trend | 30m | 7.76% | 1.477 | 2.00% | 1.070 | 1.27% |

## Portfolio metrics

- Fit (val): {"cagr": 0.23663203636532604, "calmar": 34.70031186089736, "max_drawdown": 0.0068193057547698555, "sharpe": 3.690308782268639, "sortino": 7.264688892047521, "total_return": 0.026531109970909794, "volatility": 0.05801547801050995}
- Report (oos): {"cagr": 0.6721279004221061, "calmar": 46.87683158137706, "max_drawdown": 0.014338168296534892, "sharpe": 3.3399147634411137, "sortino": 12.132653392050479, "total_return": 0.050532284452287746, "volatility": 0.15758053460783877}
- Train: {"cagr": 0.028662696026351187, "calmar": 0.40102065275235854, "max_drawdown": 0.07147436380053773, "sharpe": 0.40958310114573326, "sortino": 0.7178458814964644, "total_return": 0.031054755313634663, "volatility": 0.07600370109345929}
- Val: {"cagr": 0.23663203636532604, "calmar": 34.70031186089736, "max_drawdown": 0.0068193057547698555, "sharpe": 3.690308782268639, "sortino": 7.264688892047521, "total_return": 0.026531109970909794, "volatility": 0.05801547801050995}
- OOS: {"cagr": 0.6721279004221061, "calmar": 46.87683158137706, "max_drawdown": 0.014338168296534892, "sharpe": 3.3399147634411137, "sortino": 12.132653392050479, "total_return": 0.050532284452287746, "volatility": 0.15758053460783877}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.663514190718343,
      "calmar": 45.962060089312594,
      "max_drawdown": 0.014436128176783525,
      "sharpe": 3.30709511268746,
      "sortino": 12.01343195220991,
      "total_return": 0.05001214619964056,
      "volatility": 0.15758053460783877
    },
    "x3": {
      "cagr": 0.6549447322648463,
      "calmar": 45.06269025920586,
      "max_drawdown": 0.014534079712008485,
      "sharpe": 3.274275461933806,
      "sortino": 11.894210512369337,
      "total_return": 0.04949225812227631,
      "volatility": 0.15758053460783877
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.5900536383916661,
      "calmar": 45.702485757715365,
      "max_drawdown": 0.0129107559164231,
      "sharpe": 3.339914763441115,
      "sortino": 12.132653392050482,
      "total_return": 0.045474532242217425,
      "volatility": 0.14182248114705487
    },
    "plus_10pct_signal": {
      "cagr": 0.7580249529481484,
      "calmar": 48.085334525061505,
      "max_drawdown": 0.015764160953337547,
      "sharpe": 3.3399147634411146,
      "sortino": 12.132653392050482,
      "total_return": 0.055590690220074235,
      "volatility": 0.17333858806862268
    }
  }
}
```
