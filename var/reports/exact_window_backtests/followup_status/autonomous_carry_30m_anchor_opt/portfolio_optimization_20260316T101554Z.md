# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/.omx/team/continue-the-autonomous-portfo/worktrees/worker-1/var/reports/exact_window_backtests/followup_status/autonomous_carry_30m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- carry: 25.00%
- cross_sectional: 25.00%
- market_neutral: 25.00%
- trend: 25.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 25.00% | 1.641 | 1.84% | 1.464 | 3.24% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 25.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 4 | perp_crowding_carry_30m_0.25_0.08 | PerpCrowdingCarryStrategy | carry | 30m | 25.00% | 0.000 | 0.00% | 0.852 | 0.03% |

## Portfolio metrics

- Fit (val): {"cagr": 0.28059634306691206, "calmar": 32.56933947828844, "max_drawdown": 0.008615352585027547, "sharpe": 3.72976013306876, "sortino": 7.138464135293636, "total_return": 0.021227932036253616, "volatility": 0.0669132004330647}
- Report (oos): {"cagr": 0.41825005965558826, "calmar": 22.350911308249387, "max_drawdown": 0.0187128861945427, "sharpe": 2.688499325618447, "sortino": 9.76683134890767, "total_return": 0.03407404924288149, "volatility": 0.13318672274881993}
- Train: {"cagr": 0.007794352907496416, "calmar": 0.11395473074924835, "max_drawdown": 0.06839867775781505, "sharpe": 0.13931960180393568, "sortino": 0.2776573403667975, "total_return": 0.007794352907496416, "volatility": 0.07666485713686731}
- Val: {"cagr": 0.28059634306691206, "calmar": 32.56933947828844, "max_drawdown": 0.008615352585027547, "sharpe": 3.72976013306876, "sortino": 7.138464135293636, "total_return": 0.021227932036253616, "volatility": 0.0669132004330647}
- OOS: {"cagr": 0.41825005965558826, "calmar": 22.350911308249387, "max_drawdown": 0.0187128861945427, "sharpe": 2.688499325618447, "sortino": 9.76683134890767, "total_return": 0.03407404924288149, "volatility": 0.13318672274881993}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.4123272487262024,
      "calmar": 21.941723705449718,
      "max_drawdown": 0.018791926024653738,
      "sharpe": 2.65704894676122,
      "sortino": 9.652577816005213,
      "total_return": 0.03365916861095797,
      "volatility": 0.13318672274881993
    },
    "x3": {
      "cagr": 0.4064291047985549,
      "calmar": 21.537277183129042,
      "max_drawdown": 0.01887096039776681,
      "sharpe": 2.6255985679039933,
      "sortino": 9.538324283102755,
      "total_return": 0.033244449679012744,
      "volatility": 0.13318672274881993
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.37059978225916557,
      "calmar": 21.988970278995655,
      "max_drawdown": 0.016853894364174504,
      "sharpe": 2.688499325618447,
      "sortino": 9.766831348907676,
      "total_return": 0.03069084694042612,
      "volatility": 0.11986805047393793
    },
    "plus_10pct_signal": {
      "cagr": 0.46731120295185247,
      "calmar": 22.71902645362493,
      "max_drawdown": 0.020569156161059476,
      "sharpe": 2.6884993256184453,
      "sortino": 9.766831348907669,
      "total_return": 0.03745169694380657,
      "volatility": 0.14650539502370197
    }
  }
}
```
