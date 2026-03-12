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
| TopCapTimeSeriesMomentumStrategy | topcap_tsmom_1h_fast36_slow_tight_16_6_0.020 | 1h | 4.493 | 0 | 0 | 2.25% | 1.400 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| topcap_tsmom_1h_fast36_slow_tight_16_6_0.020 | TopCapTimeSeriesMomentumStrategy | 100.00% | 2.25% | 1.400 |

## Portfolio Monthly Hurdle

- 2026-02: return=2.68%, btc=-12.99%, threshold=2.00%, pass=True
- 2026-03: return=-0.42%, btc=3.40%, threshold=3.40%, pass=False
