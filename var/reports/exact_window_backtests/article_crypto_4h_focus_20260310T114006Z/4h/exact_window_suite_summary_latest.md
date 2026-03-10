# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT
- Excluded symbols: 
- Candidate count: 63
- Evaluated count: 63

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| PairSpreadZScoreStrategy | pair_spread_4h_fast_cycle_btcusdt_ethusdt_1.6_0.35 | 4h | 9.823 | 0 | 0 | 2.49% | 2.899 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_4h_0.25_0.08 | 4h | 0.000 | 0 | 0 | 0.00% | 0.000 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| pair_spread_4h_fast_cycle_btcusdt_ethusdt_1.6_0.35 | PairSpreadZScoreStrategy | 50.00% | 2.49% | 2.899 |
| perp_crowding_carry_4h_0.25_0.08 | PerpCrowdingCarryStrategy | 50.00% | 0.00% | 0.000 |

## Portfolio Monthly Hurdle

- 2026-02: return=0.95%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=0.29%, btc=3.40%, threshold=3.40%, pass=False
