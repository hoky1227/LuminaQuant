# Profit Moonshot Research Summary

Generated: `2026-05-02T08:52:51.046063Z`
Decision: `promoted_candidate_found`
Candidates scanned: `14`
Promotion-eligible candidates: `9`

## Promoted Candidate

- Candidate: `profit_moonshot_adaptive_momentum_asym_dynamic_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/latest/profit_moonshot_adaptive_momentum_asym_dynamic_mode/live_equivalent_revalidation_latest.json`
- val return: `0.00%`
- max_drawdown: `0.07%`
- Sharpe / Sortino: `0.000` / `0.000`
- trades / liquidations: `102` / `0`
- final_equity: `10000.001`

## Best Validation Return Candidate

- Candidate: `profit_moonshot_adaptive_momentum_boost_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/continuation/adaptive_boost/live_equivalent_revalidation_latest.json`
- val return: `0.51%`
- max_drawdown: `1.36%`
- Sharpe / Sortino: `0.015` / `0.015`
- trades / liquidations: `56` / `0`

## Top Ranked Candidates

| rank | candidate | source | split | return | MDD | Sharpe | Sortino | trades | liq | final equity | blockers |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 10000.001 | - |
| 2 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 10026.530 | - |
| 3 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 4 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 10032.962 | - |
| 5 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 10039.730 | - |
| 6 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 10041.849 | - |
| 7 | `profit_moonshot_adaptive_momentum_130_mode` | `live_equivalent` | `val` | 0.33% | 1.06% | 0.011 | 0.011 | 53 | 0 | 10033.283 | - |
| 8 | `profit_moonshot_adaptive_momentum_governed_mode` | `live_equivalent` | `val` | 0.17% | 1.16% | 0.006 | 0.006 | 51 | 0 | 10016.980 | - |
| 9 | `profit_moonshot_trend_mode` | `live_equivalent` | `val` | 0.01% | 0.03% | 0.017 | 0.004 | 6 | 0 | 10000.904 | - |
| 10 | `profit_moonshot_balanced_mode` | `live_equivalent` | `val` | -0.01% | 0.04% | -0.010 | -0.006 | 112 | 0 | 9999.449 | val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |
| 11 | `profit_moonshot_adaptive_momentum_volume_guard_mode` | `live_equivalent` | `val` | -0.05% | 0.09% | -0.055 | -0.014 | 122 | 0 | 9995.152 | val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |
| 12 | `profit_moonshot_reversion_mode` | `live_equivalent` | `val` | -0.03% | 0.13% | -0.017 | -0.010 | 106 | 0 | 9997.233 | val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |
| 13 | `profit_reboot_compression_breakout_mode` | `live_equivalent` | `val` | -1.03% | 1.04% | -0.114 | -0.050 | 38 | 0 | 9897.479 | val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |
| 14 | `profit_moonshot_breakout_mode` | `live_equivalent` | `val` | 0.00% | 0.00% | 0.000 | 0.000 | 0 | 0 | 10000.000 | train_trade_count_below_min, val_trade_count_below_min, val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |

## Blocker Summary

- `val_sharpe_not_positive`: 5
- `val_sortino_not_positive`: 5
- `val_total_return_not_positive`: 5
- `train_trade_count_below_min`: 1
- `val_trade_count_below_min`: 1
