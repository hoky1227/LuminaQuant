# Exact-Window Validation Suite

## Windows
- Train: `2026-01-07T10:00:00+00:00` → `2026-02-05T10:00:00+00:00`
- Validation: `2026-02-05T10:00:00+00:00` → `2026-02-18T10:00:00+00:00`
- OOS requested end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, XAU/USDT, XAG/USDT
- Excluded symbols: 
- Candidate count: 6
- Evaluated count: 6

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| PairSpreadZScoreStrategy | pair_spread_4h_balanced_btcusdt_xauusdt_1.8_0.45 | 4h | 0.000 | 0 | 1 | 0.00% | 0.000 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| pair_spread_4h_balanced_btcusdt_xauusdt_1.8_0.45 | PairSpreadZScoreStrategy | 100.00% | 0.00% | 0.000 |

## Portfolio Monthly Hurdle

- 2026-02: return=0.00%, btc=-5.11%, threshold=2.00%, pass=False
- 2026-03: return=0.00%, btc=3.40%, threshold=3.40%, pass=False
