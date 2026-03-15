# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/perp_crowding_carry_30m_0.25_0.08.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- carry: 0.32%
- cross_sectional: 18.09%
- market_neutral: 50.00%
- trend: 31.60%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 50.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 31.60% | 4.428 | 4.99% | 2.295 | 2.95% |
| 3 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 18.09% | 1.641 | 1.84% | 1.464 | 3.24% |
| 4 | perp_crowding_carry_30m_0.25_0.08 | PerpCrowdingCarryStrategy | carry | 30m | 0.32% | 0.000 | 0.00% | 3.438 | 0.06% |

## Portfolio metrics

- Fit (val): {"cagr": 0.24410102999339345, "calmar": 33.023346858157424, "max_drawdown": 0.007391771374411604, "sharpe": 3.8441763550229218, "sortino": 5.797991570558576, "total_return": 0.027293480930211, "volatility": 0.05724952055791281}
- Report (oos): {"cagr": 0.6921575436787641, "calmar": 42.29291696563759, "max_drawdown": 0.01636580291307721, "sharpe": 3.2529551834853865, "sortino": 11.749226220980828, "total_return": 0.05173246905462037, "volatility": 0.16584122354706904}
- Train: {"cagr": 0.027536550320812703, "calmar": 0.36950973194457093, "max_drawdown": 0.07452185406836154, "sharpe": 0.3706806857246583, "sortino": 0.662680745358558, "total_return": 0.029833270676661305, "volatility": 0.08236828174331089}
- Val: {"cagr": 0.24410102999339345, "calmar": 33.023346858157424, "max_drawdown": 0.007391771374411604, "sharpe": 3.8441763550229218, "sortino": 5.797991570558576, "total_return": 0.027293480930211, "volatility": 0.05724952055791281}
- OOS: {"cagr": 0.6921575436787641, "calmar": 42.29291696563759, "max_drawdown": 0.01636580291307721, "sharpe": 3.2529551834853865, "sortino": 11.749226220980828, "total_return": 0.05173246905462037, "volatility": 0.16584122354706904}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.6839216783826829,
      "calmar": 41.5551127309405,
      "max_drawdown": 0.016458183685143957,
      "sharpe": 3.2234945103099966,
      "sortino": 11.642818325932867,
      "total_return": 0.051240535624247574,
      "volatility": 0.16584122354706904
    },
    "x3": {
      "cagr": 0.6757257884026291,
      "calmar": 40.82797863363149,
      "max_drawdown": 0.016550557020376444,
      "sharpe": 3.1940338371346066,
      "sortino": 11.536410430884906,
      "total_return": 0.05074882571737338,
      "volatility": 0.16584122354706904
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.6073607179610345,
      "calmar": 41.21137592443307,
      "max_drawdown": 0.014737695705057674,
      "sharpe": 3.252955183485387,
      "sortino": 11.749226220980832,
      "total_return": 0.04656039060933814,
      "volatility": 0.1492571011923621
    },
    "plus_10pct_signal": {
      "cagr": 0.7809675951695147,
      "calmar": 43.40630170672652,
      "max_drawdown": 0.017992032595776086,
      "sharpe": 3.252955183485387,
      "sortino": 11.749226220980832,
      "total_return": 0.05690391963527608,
      "volatility": 0.18242534590177592
    }
  }
}
```
