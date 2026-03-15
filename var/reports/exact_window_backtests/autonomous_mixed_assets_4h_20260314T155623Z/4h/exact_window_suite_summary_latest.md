# Exact-Window Validation Suite

## Windows
- Train: `2026-01-07T10:00:00+00:00` → `2026-02-05T10:00:00+00:00`
- Validation: `2026-02-05T10:00:00+00:00` → `2026-02-18T10:00:00+00:00`
- OOS requested end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT, XAU/USDT, XAG/USDT
- Excluded symbols: 
- Candidate count: 168
- Evaluated count: 168

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| Alpha101FormulaStrategy | alpha101_formula_4h_a101_a101_bodyrange_swing_dir | 4h | 11.376 | 0 | 1 | 9.69% | 6.358 |
| LagConvergenceStrategy | lag_convergence_4h_metals_core_xauusdt_xagusdt_2_0.018 | 4h | -3.592 | 0 | 1 | -0.49% | -0.691 |
| PairSpreadZScoreStrategy | pair_spread_4h_participation_ethusdt_xauusdt_2.2_0.55 | 4h | 1.586 | 0 | 1 | -2.54% | -5.976 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_4h_0.25_0.08 | 4h | 0.000 | 0 | 1 | 0.00% | 0.000 |
| TopCapTimeSeriesMomentumStrategy | topcap_tsmom_4h_balanced_10_2_0.020 | 4h | -13.326 | 0 | 1 | -1.64% | -1.251 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| perp_crowding_carry_4h_0.25_0.08 | PerpCrowdingCarryStrategy | 35.00% | 0.00% | 0.000 |
| pair_spread_4h_participation_ethusdt_xauusdt_2.2_0.55 | PairSpreadZScoreStrategy | 34.72% | -2.54% | -5.976 |
| lag_convergence_4h_metals_core_xauusdt_xagusdt_2_0.018 | LagConvergenceStrategy | 13.80% | -0.49% | -0.691 |
| alpha101_formula_4h_a101_a101_bodyrange_swing_dir | Alpha101FormulaStrategy | 8.51% | 9.69% | 6.358 |
| topcap_tsmom_4h_balanced_10_2_0.020 | TopCapTimeSeriesMomentumStrategy | 7.96% | -1.64% | -1.251 |

## Portfolio Monthly Hurdle

- 2026-02: return=-0.42%, btc=-5.11%, threshold=2.00%, pass=False
- 2026-03: return=0.15%, btc=3.40%, threshold=3.40%, pass=False
