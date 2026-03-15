# Exact-Window Validation Suite

## Windows
- Train: `2026-01-07T10:00:00+00:00` → `2026-02-05T10:00:00+00:00`
- Validation: `2026-02-05T10:00:00+00:00` → `2026-02-18T10:00:00+00:00`
- OOS requested end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT, XAU/USDT, XAG/USDT
- Excluded symbols: 
- Candidate count: 12
- Evaluated count: 12

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| CompositeTrendStrategy | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | 30m | 5.156 | 0 | 1 | 1.99% | 4.354 |
| MeanReversionStdStrategy | mean_reversion_std_30m_guarded_lo_72_2.20 | 30m | -1.893 | 0 | 1 | -2.75% | -3.159 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_30m_0.25_0.08 | 30m | -0.001 | 0 | 1 | 0.06% | 3.438 |
| RegimeBreakoutCandidateStrategy | regime_breakout_30m_trend_ls_64_0.72 | 30m | 2.747 | 0 | 1 | 1.27% | 1.070 |
| RollingBreakoutStrategy | rolling_breakout_30m_guarded_ls_64_0.002 | 30m | -4.837 | 0 | 1 | 1.78% | 1.383 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | 35.00% | 1.99% | 4.354 |
| perp_crowding_carry_30m_0.25_0.08 | PerpCrowdingCarryStrategy | 35.00% | 0.06% | 3.438 |
| mean_reversion_std_30m_guarded_lo_72_2.20 | MeanReversionStdStrategy | 12.79% | -2.75% | -3.159 |
| regime_breakout_30m_trend_ls_64_0.72 | RegimeBreakoutCandidateStrategy | 9.74% | 1.27% | 1.070 |
| rolling_breakout_30m_guarded_ls_64_0.002 | RollingBreakoutStrategy | 7.47% | 1.78% | 1.383 |

## Portfolio Monthly Hurdle

- 2026-02: return=0.81%, btc=-5.11%, threshold=2.00%, pass=False
- 2026-03: return=-0.17%, btc=3.40%, threshold=3.40%, pass=False
