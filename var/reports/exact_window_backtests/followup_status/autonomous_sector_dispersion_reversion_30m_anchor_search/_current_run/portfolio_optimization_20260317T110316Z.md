# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_sector_dispersion_reversion_30m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 16.38%
- market_neutral: 55.00%
- trend: 28.62%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 28.62% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | pair_spread_30m_sector_btcusdt_trxusdt_2.0_0.50 | PairSpreadZScoreStrategy | market_neutral | 30m | 27.50% | 7.744 | 5.63% | -4.848 | -5.26% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 27.50% | 3.880 | 1.61% | 4.557 | 7.24% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 16.38% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.5394983873189243, "calmar": 80.55933079785193, "max_drawdown": 0.006696907508736527, "sharpe": 6.3233737369167295, "sortino": 17.24091626751489, "total_return": 0.037323940321677185, "volatility": 0.06863111436184181}
- Report (oos): {"cagr": 0.2162719951076466, "calmar": 8.177924819237724, "max_drawdown": 0.026445828237365188, "sharpe": 1.5648311364363126, "sortino": 3.4538703606343333, "total_return": 0.018951773287623208, "volatility": 0.13036546365103432}
- Train: {"cagr": -0.039510579122752465, "calmar": -0.41269229544699465, "max_drawdown": 0.09573859158179299, "sharpe": -0.575244412688082, "sortino": -1.1465890116801907, "total_return": -0.039510579122752465, "volatility": 0.06628269681146108}
- Val: {"cagr": 0.5394983873189243, "calmar": 80.55933079785193, "max_drawdown": 0.006696907508736527, "sharpe": 6.3233737369167295, "sortino": 17.24091626751489, "total_return": 0.037323940321677185, "volatility": 0.06863111436184181}
- OOS: {"cagr": 0.2162719951076466, "calmar": 8.177924819237724, "max_drawdown": 0.026445828237365188, "sharpe": 1.5648311364363126, "sortino": 3.4538703606343333, "total_return": 0.018951773287623208, "volatility": 0.13036546365103432}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.21054816551195255,
      "calmar": 7.864076949282338,
      "max_drawdown": 0.0267734111542699,
      "sharpe": 1.5286286906755526,
      "sortino": 3.3739648989623996,
      "total_return": 0.018490975385069497,
      "volatility": 0.13036546365103432
    },
    "x3": {
      "cagr": 0.20485119905491178,
      "calmar": 7.558837128216287,
      "max_drawdown": 0.027100888083727237,
      "sharpe": 1.4924262449147936,
      "sortino": 3.2940594372904677,
      "total_return": 0.018030379916435635,
      "volatility": 0.13036546365103432
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.1935620455008462,
      "calmar": 8.126943205549031,
      "max_drawdown": 0.023817324743783508,
      "sharpe": 1.564831136436312,
      "sortino": 3.4538703606343333,
      "total_return": 0.01711181367036918,
      "volatility": 0.1173289172859309
    },
    "plus_10pct_signal": {
      "cagr": 0.23921544531099848,
      "calmar": 8.228736228124799,
      "max_drawdown": 0.02907073925803938,
      "sharpe": 1.5648311364363126,
      "sortino": 3.453870360634334,
      "total_return": 0.020779375356146268,
      "volatility": 0.14340201001613775
    }
  }
}
```
