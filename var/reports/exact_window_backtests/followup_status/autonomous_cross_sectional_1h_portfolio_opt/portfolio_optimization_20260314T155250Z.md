# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_cross_sectional_1h_best_per_strategy_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 4

## Sleeve budgets

- carry: 15.00%
- cross_sectional: 15.00%
- formulaic_alpha: 15.00%
- market_neutral: 15.00%
- trend: 40.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 15.00% | 1.641 | 1.84% | 1.464 | 3.24% |
| 2 | alpha101_formula_1h_a005_a005_vwap_tuned_dir | Alpha101FormulaStrategy | formulaic_alpha | 1h | 15.00% | -2.531 | -3.87% | -6.943 | -15.98% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | market_neutral | 1h | 15.00% | 4.587 | 1.71% | 7.010 | 11.06% |
| 4 | perp_crowding_carry_1h_0.45_0.15 | PerpCrowdingCarryStrategy | carry | 1h | 15.00% | 0.000 | 0.00% | 0.000 | 0.00% |
| 5 | rolling_breakout_1h_guarded_ls_48_0.002 | RollingBreakoutStrategy | trend | 1h | 13.33% | 5.111 | 15.37% | -0.458 | -3.29% |
| 6 | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | trend | 1h | 13.33% | 3.735 | 8.90% | 1.506 | 5.54% |
| 7 | composite_trend_stable_1h_stable_lo_guarded_lo_0.75_0.60_0.25_0.95 | CompositeTrendStrategy | trend | 1h | 13.33% | -1.895 | -0.25% | -2.855 | -0.19% |

## Portfolio metrics

- Fit (val): {"cagr": 0.44238912429231014, "calmar": 30.388499236855502, "max_drawdown": 0.014557781246260948, "sharpe": 3.3831083996892968, "sortino": 9.046589640755585, "total_return": 0.03159947220203185, "volatility": 0.11004509901572851}
- Report (oos): {"cagr": 0.004404885408293646, "calmar": 0.06261148985801177, "max_drawdown": 0.07035266878783586, "sharpe": 0.12235638923936147, "sortino": 0.2508777223343695, "total_return": 0.0004215475395292767, "volatility": 0.2127124440853589}
- Train: {"cagr": -0.21633771956576087, "calmar": -0.8716334788583109, "max_drawdown": 0.2481980382960116, "sharpe": -2.037073569107019, "sortino": -3.3239795916310473, "total_return": -0.21633771956576087, "volatility": 0.11632483429208519}
- Val: {"cagr": 0.44238912429231014, "calmar": 30.388499236855502, "max_drawdown": 0.014557781246260948, "sharpe": 3.3831083996892968, "sortino": 9.046589640755585, "total_return": 0.03159947220203185, "volatility": 0.11004509901572851}
- OOS: {"cagr": 0.004404885408293646, "calmar": 0.06261148985801177, "max_drawdown": 0.07035266878783586, "sharpe": 0.12235638923936147, "sortino": 0.2508777223343695, "total_return": 0.0004215475395292767, "volatility": 0.2127124440853589}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": -0.004639994800545444,
      "calmar": -0.06539516154818671,
      "max_drawdown": 0.07095318201984169,
      "sharpe": 0.07983196497251807,
      "sortino": 0.16368627471183636,
      "total_return": -0.0004458670230055306,
      "volatility": 0.2127124440853589
    },
    "x3": {
      "cagr": -0.013603645107500895,
      "calmar": -0.1901189864920192,
      "max_drawdown": 0.07155332225628053,
      "sharpe": 0.03730754070567513,
      "sortino": 0.07693766422050372,
      "total_return": -0.0013125509672401448,
      "volatility": 0.21271244408535886
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.00589292466295066,
      "calmar": 0.09281301680280045,
      "max_drawdown": 0.06349243743979727,
      "sharpe": 0.12235638923936122,
      "sortino": 0.2508777223343689,
      "total_return": 0.0005635752567296759,
      "volatility": 0.191441199676823
    },
    "plus_10pct_signal": {
      "cagr": 0.0024981934854000354,
      "calmar": 0.03237069521304799,
      "max_drawdown": 0.07717453916136663,
      "sharpe": 0.1223563892393617,
      "sortino": 0.25087772233436995,
      "total_return": 0.00023928269630157928,
      "volatility": 0.2339836884938948
    }
  }
}
```
