# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT
- Excluded symbols: 
- Candidate count: 82
- Evaluated count: 82

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| Alpha101FormulaStrategy | alpha101_formula_1h_a005_a005_vwap_tuned_dir | 1h | -16.186 | 0 | 0 | -15.98% | -6.943 |
| CompositeTrendStrategy | composite_trend_stable_1h_stable_lo_guarded_lo_0.75_0.60_0.25_0.95 | 1h | -6.379 | 0 | 0 | -0.19% | -2.855 |
| PairSpreadZScoreStrategy | pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70 | 1h | 11.994 | 0 | 0 | 11.06% | 7.010 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_1h_0.45_0.15 | 1h | 0.000 | 0 | 0 | 0.00% | 0.000 |
| RegimeBreakoutCandidateStrategy | regime_breakout_1h_trend_ls_48_0.70 | 1h | 9.799 | 0 | 1 | 5.54% | 1.506 |
| RollingBreakoutStrategy | rolling_breakout_1h_guarded_ls_48_0.002 | 1h | 8.129 | 0 | 1 | -3.29% | -0.458 |
| TopCapTimeSeriesMomentumStrategy | topcap_tsmom_1h_exec_takeprofit_16_4_0.015 | 1h | 3.850 | 0 | 0 | 3.61% | 1.667 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| composite_trend_stable_1h_stable_lo_guarded_lo_0.75_0.60_0.25_0.95 | CompositeTrendStrategy | 35.00% | -0.19% | -2.855 |
| perp_crowding_carry_1h_0.45_0.15 | PerpCrowdingCarryStrategy | 35.00% | 0.00% | 0.000 |
| pair_spread_1h_core_bnbusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 16.82% | 11.06% | 7.010 |
| topcap_tsmom_1h_exec_takeprofit_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | 4.59% | 3.61% | 1.667 |
| alpha101_formula_1h_a005_a005_vwap_tuned_dir | Alpha101FormulaStrategy | 4.18% | -15.98% | -6.943 |
| regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | 2.34% | 5.54% | 1.506 |
| rolling_breakout_1h_guarded_ls_48_0.002 | RollingBreakoutStrategy | 2.06% | -3.29% | -0.458 |

## Portfolio Monthly Hurdle

- 2026-02: return=1.25%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=0.06%, btc=3.40%, threshold=3.40%, pass=False
