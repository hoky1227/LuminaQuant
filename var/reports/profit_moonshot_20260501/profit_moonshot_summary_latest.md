# Profit Moonshot Research Summary

Generated: `2026-05-01T13:15:36.816961Z`
Decision: `promoted_candidate_found`
Candidates scanned: `7`
Promotion-eligible candidates: `3`

## Promoted Candidate

- Candidate: `profit_moonshot_adaptive_momentum_mode`
- Source: `live_equivalent` from `var/reports/profit_moonshot_20260501/live_equivalent/live_equivalent_revalidation_latest.json`
- val return: `0.26%`
- max_drawdown: `0.75%`
- Sharpe / Sortino: `0.012` / `0.012`
- trades / liquidations: `52` / `0`
- final_equity: `10026.530`

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
| 1 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 10026.530 | - |
| 2 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 3 | `profit_moonshot_trend_mode` | `live_equivalent` | `val` | 0.01% | 0.03% | 0.017 | 0.004 | 6 | 0 | 10000.904 | - |
| 4 | `profit_moonshot_balanced_mode` | `live_equivalent` | `val` | -0.01% | 0.04% | -0.010 | -0.006 | 112 | 0 | 9999.449 | val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |
| 5 | `profit_moonshot_reversion_mode` | `live_equivalent` | `val` | -0.03% | 0.13% | -0.017 | -0.010 | 106 | 0 | 9997.233 | val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |
| 6 | `profit_reboot_compression_breakout_mode` | `live_equivalent` | `val` | -1.03% | 1.04% | -0.114 | -0.050 | 38 | 0 | 9897.479 | val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |
| 7 | `profit_moonshot_breakout_mode` | `live_equivalent` | `val` | 0.00% | 0.00% | 0.000 | 0.000 | 0 | 0 | 10000.000 | train_trade_count_below_min, val_trade_count_below_min, val_total_return_not_positive, val_sharpe_not_positive, val_sortino_not_positive |

## Blocker Summary

- `val_sharpe_not_positive`: 4
- `val_sortino_not_positive`: 4
- `val_total_return_not_positive`: 4
- `train_trade_count_below_min`: 1
- `val_trade_count_below_min`: 1
