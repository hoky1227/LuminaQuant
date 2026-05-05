# Profit Moonshot Research Summary

Generated: `2026-05-05T06:35:00Z`
Decision: `promoted_candidate_found`
Candidates scanned: `44`
Promotion-eligible candidates: `29`

## Promoted Candidate

- Candidate: `profit_moonshot_momentum_hybrid_safe_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_momentum_hybrid_safe_mode/live_equivalent_revalidation_latest.json`
- val return: `0.28%`
- max_drawdown: `1.04%`
- Sharpe / Sortino: `0.010` / `0.010`
- trades / liquidations: `183` / `0`
- final_equity: `10028.407`

## Best Validation Return Candidate

- Candidate: `profit_moonshot_leadlag_slow_diffusion_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_leadlag_slow_diffusion_mode/live_equivalent_revalidation_latest.json`
- val return: `0.68%`
- max_drawdown: `1.26%`
- Sharpe / Sortino: `0.028` / `0.029`
- trades / liquidations: `40` / `0`

## Current-tail OOS Gate Override

The summary ranking above is the legacy validation-split ranking. For the active 2026-05-05 task, the operator gate is stricter: complete positive raw-first OOS evidence is required before success or deployment claims. Under that gate, `profit_moonshot_leadlag_slow_diffusion_mode` is the current deployment-review candidate, while `profit_moonshot_momentum_hybrid_safe_mode` remains a conservative fallback only because current-tail OOS is negative. See `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.md`.

## Top Ranked Candidates

| rank | candidate | source | split | return | MDD | Sharpe | Sortino | trades | liq | final equity | blockers |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 2 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 3 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 4 | `profit_moonshot_derivatives_taker_flow_sparse_mode` | `live_equivalent` | `val` | 0.08% | 0.05% | 0.055 | 0.027 | 108 | 0 | 10007.985 | - |
| 5 | `profit_moonshot_momentum_hybrid_core_mode` | `live_equivalent` | `val` | 0.25% | 1.01% | 0.009 | 0.009 | 138 | 0 | 10025.524 | - |
| 6 | `profit_moonshot_momentum_hybrid_return_mode` | `live_equivalent` | `val` | 0.27% | 1.01% | 0.010 | 0.009 | 134 | 0 | 10026.897 | - |
| 7 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 10000.001 | - |
| 8 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 0.000 | - |
| 9 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 10 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 11 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 0.000 | - |
| 12 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 10026.530 | - |
| 13 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 14 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 0.000 | - |
| 15 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 16 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 0.000 | - |
| 17 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 10032.962 | - |
| 18 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 10039.730 | - |
| 19 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 0.000 | - |
| 20 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 10041.849 | - |
| 21 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 0.000 | - |
| 22 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 10041.849 | - |
| 23 | `profit_moonshot_adaptive_momentum_130_mode` | `live_equivalent` | `val` | 0.33% | 1.06% | 0.011 | 0.011 | 53 | 0 | 0.000 | - |
| 24 | `profit_moonshot_adaptive_momentum_130_mode` | `live_equivalent` | `val` | 0.33% | 1.06% | 0.011 | 0.011 | 53 | 0 | 10033.283 | - |
| 25 | `profit_moonshot_leadlag_slow_diffusion_ensemble_mode` | `live_equivalent` | `val` | 0.19% | 2.03% | 0.004 | 0.005 | 96 | 0 | 10019.409 | - |

## Blocker Summary

- `val_sharpe_not_positive`: 10
- `val_sortino_not_positive`: 10
- `val_total_return_not_positive`: 10
- `train_trade_count_below_min`: 5
- `val_trade_count_below_min`: 5
- `legacy_train_val_mdd_gate_failed`: 4
- `status=ready_for_live_equivalent_backtest`: 4
- `status_not_validated:ready_for_live_equivalent_backtest`: 4
