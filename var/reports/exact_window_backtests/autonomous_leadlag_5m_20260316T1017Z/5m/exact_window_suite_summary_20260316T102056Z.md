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
- Candidate count: 9
- Evaluated count: 9

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LeadLagSpilloverStrategy | leadlag_spillover_5m_0.50_lag4 | 5m | -169.675 | 0 | 0 | -75.37% | -53.957 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| leadlag_spillover_5m_0.50_lag4 | LeadLagSpilloverStrategy | 100.00% | -75.37% | -53.957 |

## Portfolio Monthly Hurdle

- 2026-02: return=-67.47%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-24.27%, btc=3.40%, threshold=3.40%, pass=False
