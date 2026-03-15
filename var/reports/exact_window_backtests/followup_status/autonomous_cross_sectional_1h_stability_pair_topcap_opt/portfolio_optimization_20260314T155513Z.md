# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_cross_sectional_1h_stability_pair_topcap_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 50.00%
- market_neutral: 50.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_ethusdt_solusdt_1.8_0.45 | PairSpreadZScoreStrategy | market_neutral | 1h | 50.00% | 4.195 | 3.07% | 1.613 | 1.69% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 50.00% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.33673564848475923, "calmar": 17.33458140815034, "max_drawdown": 0.019425657912133576, "sharpe": 3.0389700896246463, "sortino": 8.14654451898467, "total_return": 0.02495603468587304, "volatility": 0.0970332854281442}
- Report (oos): {"cagr": 0.2999222112824118, "calmar": 8.555188446761305, "max_drawdown": 0.035057347146567164, "sharpe": 1.8733583520585342, "sortino": 4.79754311812622, "total_return": 0.02547147160075869, "volatility": 0.14550730184982238}
- Train: {"cagr": -0.0008173555068567984, "calmar": -0.005828883927214936, "max_drawdown": 0.14022504429031135, "sharpe": 0.06784502752904552, "sortino": 0.11829287628677773, "total_return": -0.0008173555068567984, "volatility": 0.1487232913810957}
- Val: {"cagr": 0.33673564848475923, "calmar": 17.33458140815034, "max_drawdown": 0.019425657912133576, "sharpe": 3.0389700896246463, "sortino": 8.14654451898467, "total_return": 0.02495603468587304, "volatility": 0.0970332854281442}
- OOS: {"cagr": 0.2999222112824118, "calmar": 8.555188446761305, "max_drawdown": 0.035057347146567164, "sharpe": 1.8733583520585342, "sortino": 4.79754311812622, "total_return": 0.02547147160075869, "volatility": 0.14550730184982238}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.2905278171420618,
      "calmar": 8.255427083539303,
      "max_drawdown": 0.035192342467823656,
      "sharpe": 1.8234772034843425,
      "sortino": 4.669800894753215,
      "total_return": 0.024758498406163287,
      "volatility": 0.14550730184982238
    },
    "x3": {
      "cagr": 0.2812011305749105,
      "calmar": 7.959876883715117,
      "max_drawdown": 0.03532732160094232,
      "sharpe": 1.773596054910151,
      "sortino": 4.542058671380212,
      "total_return": 0.024046006761827776,
      "volatility": 0.14550730184982238
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.26743009149857677,
      "calmar": 8.464105148443984,
      "max_drawdown": 0.03159579031786253,
      "sharpe": 1.8733583520585342,
      "sortino": 4.79754311812622,
      "total_return": 0.022985377559837428,
      "volatility": 0.13095657166484018
    },
    "plus_10pct_signal": {
      "cagr": 0.332978428627984,
      "calmar": 8.646732621874964,
      "max_drawdown": 0.038509162152834175,
      "sharpe": 1.8733583520585342,
      "sortino": 4.797543118126222,
      "total_return": 0.027943726511222566,
      "volatility": 0.16005803203480468
    }
  }
}
```
