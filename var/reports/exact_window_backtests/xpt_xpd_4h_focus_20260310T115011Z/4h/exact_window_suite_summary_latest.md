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
- Candidate count: 22
- Evaluated count: 22

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LagConvergenceStrategy | lag_convergence_4h_metals_fast_xptusdt_xpdusdt_1_0.014 | 4h | 12.969 | 0 | 1 | 1.69% | 7.948 |
| PairSpreadZScoreStrategy | pair_spread_4h_balanced_xptusdt_xpdusdt_1.6_0.35 | 4h | 3.398 | 0 | 1 | -4.22% | -18.519 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| lag_convergence_4h_metals_fast_xptusdt_xpdusdt_1_0.014 | LagConvergenceStrategy | 50.00% | 1.69% | 7.948 |
| pair_spread_4h_balanced_xptusdt_xpdusdt_1.6_0.35 | PairSpreadZScoreStrategy | 50.00% | -4.22% | -18.519 |

## Portfolio Monthly Hurdle

- 2026-02: return=-0.32%, btc=0.77%, threshold=2.00%, pass=False
- 2026-03: return=-0.97%, btc=3.41%, threshold=3.41%, pass=False
