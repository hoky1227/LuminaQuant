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
| TopCapTimeSeriesMomentumStrategy | topcap_tsmom_1h_longonly_bridge18_slow_2_018_18_6_0.018 | 1h | -1.115 | 0 | 0 | 0.19% | 0.216 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| topcap_tsmom_1h_longonly_bridge18_slow_2_018_18_6_0.018 | TopCapTimeSeriesMomentumStrategy | 100.00% | 0.19% | 0.216 |

## Portfolio Monthly Hurdle

- 2026-02: return=1.03%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-0.83%, btc=3.40%, threshold=3.40%, pass=False
