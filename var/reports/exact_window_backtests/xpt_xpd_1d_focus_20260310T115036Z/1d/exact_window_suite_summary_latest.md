# Exact-Window Validation Suite

## Windows
- Train: `2026-01-30T10:15:00+00:00` → `2026-02-17T10:15:00+00:00`
- Validation: `2026-02-17T10:15:00+00:00` → `2026-02-24T10:15:00+00:00`
- OOS requested end-exclusive: `2026-03-07T11:02:00.001000+00:00`
- OOS actual end-exclusive: `2026-03-07T11:02:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T11:02:00+00:00`

## Universe
- Eligible symbols: XPT/USDT, XPD/USDT
- Excluded symbols: 
- Candidate count: 17
- Evaluated count: 17

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LagConvergenceStrategy | lag_convergence_1d_metals_patience_xptusdt_xpdusdt_2_0.015 | 1d | 21.794 | 0 | 1 | -3.02% | -7.576 |
| PairSpreadZScoreStrategy | pair_spread_1d_participation_xptusdt_xpdusdt_1.4_0.30 | 1d | -2.000 | 0 | 1 | 0.00% | 0.000 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| pair_spread_1d_participation_xptusdt_xpdusdt_1.4_0.30 | PairSpreadZScoreStrategy | 50.00% | 0.00% | 0.000 |
| lag_convergence_1d_metals_patience_xptusdt_xpdusdt_2_0.015 | LagConvergenceStrategy | 50.00% | -3.02% | -7.576 |

## Portfolio Monthly Hurdle

- 2026-02: return=-0.95%, btc=0.77%, threshold=2.00%, pass=False
- 2026-03: return=-0.57%, btc=3.41%, threshold=3.41%, pass=False
