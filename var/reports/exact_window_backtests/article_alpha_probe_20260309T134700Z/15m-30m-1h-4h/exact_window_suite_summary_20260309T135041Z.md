# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT
- Excluded symbols: XAU/USDT, XAG/USDT
- Candidate count: 105
- Evaluated count: 105

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| CompositeTrendStrategy | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | 30m | 11.569 | 0 | 1 | 2.21% | 1.625 |
| LeadLagSpilloverStrategy | leadlag_spillover_15m_0.50_lag4 | 15m | -185.803 | 0 | 0 | -25.37% | -81.211 |
| PairSpreadZScoreStrategy | pair_spread_4h_fast_cycle_btcusdt_ethusdt_1.6_0.35 | 4h | 9.813 | 0 | 0 | 2.49% | 2.899 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_1h_0.45_0.15 | 1h | 0.000 | 0 | 0 | 0.00% | 0.000 |
| VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | 15m | -4.577 | 0 | 0 | -0.33% | -1.817 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| perp_crowding_carry_1h_0.45_0.15 | PerpCrowdingCarryStrategy | 35.00% | 0.00% | 0.000 |
| volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | VolCompressionVWAPReversionStrategy | 35.00% | -0.33% | -1.817 |
| leadlag_spillover_15m_0.50_lag4 | LeadLagSpilloverStrategy | 13.21% | -25.37% | -81.211 |
| pair_spread_4h_fast_cycle_btcusdt_ethusdt_1.6_0.35 | PairSpreadZScoreStrategy | 10.44% | 2.49% | 2.899 |
| composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | 6.35% | 2.21% | 1.625 |

## Portfolio Monthly Hurdle

- 2026-02: return=-2.88%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-0.64%, btc=3.40%, threshold=3.40%, pass=False
