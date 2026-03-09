# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT
- Excluded symbols: XAG/USDT, XAU/USDT
- Candidate count: 49
- Evaluated count: 49

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| CompositeTrendStrategy | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | 30m | 6.961 | 0 | 1 | -0.43% | -0.175 |
| PairSpreadZScoreStrategy | pair_spread_4h_fast_cycle_btcusdt_ethusdt_1.6_0.35 | 4h | 9.831 | 0 | 0 | 2.49% | 2.899 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_4h_0.25_0.08 | 4h | 0.000 | 0 | 0 | 0.00% | 0.000 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| pair_spread_4h_fast_cycle_btcusdt_ethusdt_1.6_0.35 | PairSpreadZScoreStrategy | 35.00% | 2.49% | 2.899 |
| perp_crowding_carry_4h_0.25_0.08 | PerpCrowdingCarryStrategy | 35.00% | 0.00% | 0.000 |
| composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | 30.00% | -0.43% | -0.175 |

## Portfolio Monthly Hurdle

- 2026-02: return=0.24%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=0.52%, btc=3.40%, threshold=3.40%, pass=False
