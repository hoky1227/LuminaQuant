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
- Candidate count: 27
- Evaluated count: 27

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| CompositeTrendStrategy | composite_trend_stable_1h_stable_lo_guarded_lo_0.75_0.60_0.25_0.95 | 1h | -6.378 | 0 | 0 | -0.19% | -2.855 |
| PairSpreadZScoreStrategy | pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70 | 1h | -5.689 | 0 | 0 | -2.11% | -3.833 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_1h_0.45_0.15 | 1h | 0.000 | 0 | 0 | 0.00% | 0.000 |
| RegimeBreakoutCandidateStrategy | regime_breakout_1h_trend_ls_48_0.70 | 1h | 10.679 | 0 | 1 | 8.08% | 2.009 |
| RollingBreakoutStrategy | rolling_breakout_1h_guarded_ls_48_0.002 | 1h | 8.227 | 0 | 1 | -3.29% | -0.458 |
| TopCapTimeSeriesMomentumStrategy | topcap_tsmom_1h_balanced_16_4_0.015 | 1h | 3.735 | 0 | 0 | 3.24% | 1.464 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| composite_trend_stable_1h_stable_lo_guarded_lo_0.75_0.60_0.25_0.95 | CompositeTrendStrategy | 35.00% | -0.19% | -2.855 |
| perp_crowding_carry_1h_0.45_0.15 | PerpCrowdingCarryStrategy | 35.00% | 0.00% | 0.000 |
| pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 15.25% | -2.11% | -3.833 |
| topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | 7.43% | 3.24% | 1.464 |
| rolling_breakout_1h_guarded_ls_48_0.002 | RollingBreakoutStrategy | 3.67% | -3.29% | -0.458 |
| regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | 3.65% | 8.08% | 2.009 |

## Portfolio Monthly Hurdle

- 2026-02: return=0.47%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-0.33%, btc=3.40%, threshold=3.40%, pass=False
