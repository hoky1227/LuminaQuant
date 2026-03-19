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
- Candidate count: 23
- Evaluated count: 23

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| CompositeTrendStrategy | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | 30m | 11.692 | 0 | 1 | 2.21% | 1.625 |
| MeanReversionStdStrategy | mean_reversion_std_30m_balanced_ls_48_1.80 | 30m | -21.972 | 0 | 0 | -13.13% | -2.855 |
| PairSpreadZScoreStrategy | pair_spread_30m_sector_btcusdt_trxusdt_2.0_0.50 | 30m | 10.165 | 0 | 1 | -5.26% | -4.848 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_30m_0.25_0.08 | 30m | -0.001 | 0 | 0 | 0.06% | 2.417 |
| RegimeBreakoutCandidateStrategy | regime_breakout_30m_trend_ls_64_0.72 | 30m | 4.204 | 0 | 1 | 2.45% | 0.778 |
| RollingBreakoutStrategy | rolling_breakout_30m_guarded_ls_64_0.002 | 30m | 12.350 | 0 | 1 | 4.08% | 1.042 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| perp_crowding_carry_30m_0.25_0.08 | PerpCrowdingCarryStrategy | 35.00% | 0.06% | 2.417 |
| pair_spread_30m_sector_btcusdt_trxusdt_2.0_0.50 | PairSpreadZScoreStrategy | 32.70% | -5.26% | -4.848 |
| composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | 17.38% | 2.21% | 1.625 |
| rolling_breakout_30m_guarded_ls_64_0.002 | RollingBreakoutStrategy | 5.63% | 4.08% | 1.042 |
| regime_breakout_30m_trend_ls_64_0.72 | RegimeBreakoutCandidateStrategy | 4.80% | 2.45% | 0.778 |
| mean_reversion_std_30m_balanced_ls_48_1.80 | MeanReversionStdStrategy | 4.49% | -13.13% | -2.855 |

## Portfolio Monthly Hurdle

- 2026-02: return=-1.85%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=0.44%, btc=3.40%, threshold=3.40%, pass=False
