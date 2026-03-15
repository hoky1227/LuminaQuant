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
- Candidate count: 51
- Evaluated count: 51

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| CompositeTrendStrategy | composite_trend_30m_retune_core_relaxed_ls_0.60_0.40_0.15_0.75 | 30m | 11.820 | 1 | 1 | 4.32% | 2.586 |
| RegimeBreakoutCandidateStrategy | regime_breakout_1h_tight_vol_stop_ls_48_0.78 | 1h | 11.673 | 0 | 1 | 4.76% | 1.391 |
| RollingBreakoutStrategy | rolling_breakout_30m_guarded_ls_64_0.0015 | 30m | 13.008 | 0 | 1 | 3.56% | 0.943 |
| TopCapTimeSeriesMomentumStrategy | topcap_tsmom_1h_slow_rebalance_16_6_0.015 | 1h | 7.165 | 0 | 1 | 2.20% | 1.292 |

## Portfolio

- Construction basis: `promoted_only`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| composite_trend_30m_retune_core_relaxed_ls_0.60_0.40_0.15_0.75 | CompositeTrendStrategy | 100.00% | 4.32% | 2.586 |

## Portfolio Monthly Hurdle

- 2026-02: return=3.38%, btc=-12.99%, threshold=2.00%, pass=True
- 2026-03: return=0.91%, btc=3.40%, threshold=3.40%, pass=False
