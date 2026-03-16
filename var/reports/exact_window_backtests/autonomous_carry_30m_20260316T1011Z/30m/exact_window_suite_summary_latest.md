# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, XRP/USDT, BNB/USDT, SOL/USDT, TRX/USDT, DOGE/USDT, ADA/USDT, TON/USDT, AVAX/USDT
- Excluded symbols: 
- Candidate count: 3
- Evaluated count: 3

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| PerpCrowdingCarryStrategy | perp_crowding_carry_30m_0.35_0.10 | 30m | -0.001 | 0 | 0 | 0.01% | 0.359 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| perp_crowding_carry_30m_0.35_0.10 | PerpCrowdingCarryStrategy | 100.00% | 0.01% | 0.359 |

## Portfolio Monthly Hurdle

- 2026-02: return=0.00%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=0.01%, btc=3.40%, threshold=3.40%, pass=False
