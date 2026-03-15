# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_cross_sectional_1h_tradecount_pair_topcap_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 50.00%
- market_neutral: 50.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 50.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 50.00% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.22790678265404307, "calmar": 13.39732143317693, "max_drawdown": 0.01701136930921565, "sharpe": 2.2390084552755107, "sortino": 3.6708511754862503, "total_return": 0.01759028394632467, "volatility": 0.0936118467216567}
- Report (oos): {"cagr": 0.7093249336383973, "calmar": 21.277386508508116, "max_drawdown": 0.03333703288017831, "sharpe": 2.6426419885484664, "sortino": 8.743407848945845, "total_return": 0.05275096752193753, "volatility": 0.21102872411596998}
- Train: {"cagr": -0.033069438661608386, "calmar": -0.26304760934648386, "max_drawdown": 0.12571655277067972, "sharpe": -0.2160743290962097, "sortino": -0.3590519869470726, "total_return": -0.033069438661608386, "volatility": 0.1216161294414054}
- Val: {"cagr": 0.22790678265404307, "calmar": 13.39732143317693, "max_drawdown": 0.01701136930921565, "sharpe": 2.2390084552755107, "sortino": 3.6708511754862503, "total_return": 0.01759028394632467, "volatility": 0.0936118467216567}
- OOS: {"cagr": 0.7093249336383973, "calmar": 21.277386508508116, "max_drawdown": 0.03333703288017831, "sharpe": 2.6426419885484664, "sortino": 8.743407848945845, "total_return": 0.05275096752193753, "volatility": 0.21102872411596998}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.696494618114867,
      "calmar": 20.804809210488877,
      "max_drawdown": 0.03347757775945981,
      "sharpe": 2.6068887911389043,
      "sortino": 8.625115326458761,
      "total_return": 0.05199065576252737,
      "volatility": 0.21102872411596996
    },
    "x3": {
      "cagr": 0.6837603459567898,
      "calmar": 20.339050742012777,
      "max_drawdown": 0.03361810512348051,
      "sharpe": 2.5711355937293425,
      "sortino": 8.421598843899153,
      "total_return": 0.05123087743157195,
      "volatility": 0.21102872411596998
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.6231888420491141,
      "calmar": 20.742384000882385,
      "max_drawdown": 0.03004422452224409,
      "sharpe": 2.642641988548467,
      "sortino": 8.743407848945845,
      "total_return": 0.04754424321302797,
      "volatility": 0.18992585170437296
    },
    "plus_10pct_signal": {
      "cagr": 0.7992882262008869,
      "calmar": 21.826058144177885,
      "max_drawdown": 0.03662082364671504,
      "sharpe": 2.642641988548467,
      "sortino": 8.743407848945845,
      "total_return": 0.05794164866034501,
      "volatility": 0.23213159652756693
    }
  }
}
```
