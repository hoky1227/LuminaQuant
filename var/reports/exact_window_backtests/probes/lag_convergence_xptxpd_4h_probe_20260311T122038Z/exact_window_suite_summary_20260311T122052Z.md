# Exact-Window Validation Suite

## Windows
- Train: `2026-01-30T10:15:00+00:00` → `2026-02-17T10:15:00+00:00`
- Validation: `2026-02-17T10:15:00+00:00` → `2026-02-25T10:15:00+00:00`
- OOS requested end-exclusive: `2026-03-07T11:02:00.001000+00:00`
- OOS actual end-exclusive: `2026-03-07T11:02:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T11:02:00+00:00`

## Universe
- Eligible symbols: XPT/USDT, XPD/USDT
- Excluded symbols: 
- Candidate count: 6
- Evaluated count: 6

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LagConvergenceStrategy | lag_convergence_4h_patience_3_0018_xptusdt_xpdusdt_3_0.018 | 4h | 14.956 | 0 | 1 | 0.56% | 2.010 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| lag_convergence_4h_patience_3_0018_xptusdt_xpdusdt_3_0.018 | LagConvergenceStrategy | 100.00% | 0.56% | 2.010 |

## Portfolio Monthly Hurdle

- 2026-02: return=-1.65%, btc=0.77%, threshold=2.00%, pass=False
- 2026-03: return=2.25%, btc=3.41%, threshold=3.41%, pass=False
