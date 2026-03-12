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
- Candidate count: 8
- Evaluated count: 8

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| TopCapTimeSeriesMomentumStrategy | topcap_tsmom_1h_slow_rebalance_16_6_0.015 | 1h | 7.302 | 0 | 1 | 2.20% | 1.292 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| topcap_tsmom_1h_slow_rebalance_16_6_0.015 | TopCapTimeSeriesMomentumStrategy | 100.00% | 2.20% | 1.292 |

## Portfolio Monthly Hurdle

- 2026-02: return=1.92%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=0.28%, btc=3.40%, threshold=3.40%, pass=False
