# Profit Moonshot Research Summary

Generated: `2026-05-05T10:10:00Z`
Decision: `operator_oos_override_candidate_found`
Candidates scanned: `547`
Promotion-eligible candidates: `32`

## Operator OOS Override Candidate

The generated summary is val-ranked, but this session uses the user-required raw-first OOS gate. The OOS override supersedes the val-only promoted row.

- Candidate: `profit_moonshot_hourly_shock_reversion_eth_12h_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_mode/live_equivalent_revalidation_latest.json`
- OOS return: `0.83%`
- OOS max_drawdown: `0.28%`
- OOS Sharpe / Sortino: `0.101` / `0.128`
- OOS trades / liquidations: `30` / `0`
- Train / val return: `2.45%` / `0.63%`
- Reason: beats prior weak OOS bar `+0.2910%`; no gross exposure increase; complete ETH raw-first train/val/OOS coverage.

## Generated Val-only Promoted Candidate (not operator-promoted)

- Candidate: `profit_moonshot_momentum_hybrid_safe_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_momentum_hybrid_safe_mode/live_equivalent_revalidation_latest.json`
- val return: `0.28%`
- Operator decision: not promoted because current-tail OOS gate is worse/negative or weaker than the new OOS candidate.

## Generated Best Validation Return Candidate (not operator-promoted)

- Candidate: `profit_moonshot_hourly_shock_reversion_eth_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_mode/live_equivalent_revalidation_latest.json`
- val return: `1.04%`
- Operator decision: not promoted because OOS `+0.2716%` is weaker than the prior `+0.2910%` bar.

## Top Ranked Candidates

| rank | candidate | source | split | return | MDD | Sharpe | Sortino | trades | liq | final equity | blockers |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 2 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 3 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 4 | `profit_moonshot_hourly_shock_reversion_eth_mode` | `live_equivalent` | `val` | 1.04% | 0.88% | 0.066 | 0.068 | 66 | 0 | 10104.251 | - |
| 5 | `profit_moonshot_derivatives_taker_flow_sparse_mode` | `live_equivalent` | `val` | 0.08% | 0.05% | 0.055 | 0.027 | 108 | 0 | 10007.985 | - |
| 6 | `profit_moonshot_momentum_hybrid_core_mode` | `live_equivalent` | `val` | 0.25% | 1.01% | 0.009 | 0.009 | 138 | 0 | 10025.524 | - |
| 7 | `profit_moonshot_momentum_hybrid_return_mode` | `live_equivalent` | `val` | 0.27% | 1.01% | 0.010 | 0.009 | 134 | 0 | 10026.897 | - |
| 8 | `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | `live_equivalent` | `val` | 0.63% | 0.44% | 0.071 | 0.081 | 47 | 0 | 10063.269 | - |
| 9 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 10000.001 | - |
| 10 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 0.000 | - |
| 11 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 12 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 13 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 0.000 | - |
| 14 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 10026.530 | - |
| 15 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 16 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 0.000 | - |
| 17 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 18 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 0.000 | - |
| 19 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 10032.962 | - |
| 20 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 10039.730 | - |
| 21 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 0.000 | - |
| 22 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 10041.849 | - |
| 23 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 0.000 | - |
| 24 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 10041.849 | - |
| 25 | `profit_moonshot_adaptive_momentum_130_mode` | `live_equivalent` | `val` | 0.33% | 1.06% | 0.011 | 0.011 | 53 | 0 | 0.000 | - |

## Blocker Summary

- `val_sharpe_not_positive`: 10
- `val_sortino_not_positive`: 10
- `val_total_return_not_positive`: 10
- `train_trade_count_below_min`: 5
- `val_trade_count_below_min`: 5
- `legacy_train_val_mdd_gate_failed`: 4
- `status=ready_for_live_equivalent_backtest`: 4
- `status_not_validated:ready_for_live_equivalent_backtest`: 4
