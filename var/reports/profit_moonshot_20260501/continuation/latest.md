# Profit Moonshot continuation — new alpha follow-up

Generated: `2026-05-05T12:10:46.009037Z`

## Current state

- Latest `private/main` base before this follow-up: `bbb736eb7fb74c76d81fb6723edff69828cd68c7`.
- Built and tested new `TakerFlowExhaustionReversalStrategy`; no variant passed live-equivalent train/val/OOS gates.
- OOS-return best remains `profit_moonshot_hourly_shock_reversion_eth_12h_mode`: OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`.
- Risk-adjusted shadow remains `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`: OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`.
- No candidate is deployment-ready under Sharpe `1.0`; do not promote sub-1 Sharpe as success.

## New alpha family results
| mode | train ret / MDD / trades | val ret / MDD / trades | OOS ret / MDD / Sharpe / trades | decision |
|---|---:|---:|---:|---|
| `profit_moonshot_taker_flow_exhaustion_eth_mode` | +0.0089% / 0.0048% / 4 | +0.0000% / 0.0000% / 0 | +0.0000% / 0.0000% / 0.000000 / 0 | Rejected: 7-day live-equivalent chunk phase left val/OOS with zero trades; raw cadence screen was not live-equivalent. |
| `profit_moonshot_taker_flow_exhaustion_eth_reactive_mode` | -0.0019% / 0.0753% / 147 | +0.0210% / 0.0134% / 18 | -0.0291% / 0.0364% / -0.085988 / 32 | Rejected: validation was barely positive, but train and OOS were negative after fees/one-position realism. |
| `profit_moonshot_taker_flow_exhaustion_eth_hold_mode` | -1.3901% / 1.6426% / 119 | -0.0041% / 0.0408% / 16 | -0.0875% / 0.1860% / -0.031058 / 23 | Rejected: wider exits increased train loss and kept val/OOS negative. |
| `profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode` | -0.2422% / 0.7383% / 103 | -0.3210% / 0.3237% / 18 | -0.0507% / 0.1051% / -0.036304 / 16 | Rejected: cooldown reduced churn but train/val/OOS all remained negative. |

## Key artifacts

- `var/reports/profit_moonshot_20260501/current_tail_20260505/taker_flow_exhaustion_new_alpha_report_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/taker_flow_exhaustion_screen/taker_flow_exhaustion_screen_20260505.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.json`
- `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_taker_flow_exhaustion_eth_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_taker_flow_exhaustion_eth_reactive_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_taker_flow_exhaustion_eth_hold_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode/live_equivalent_revalidation_latest.json`

## Next priority

1. Do not spend another full backtest on overlapping-event raw screens without a stateful/non-overlap replay first.
2. Benchmark against OOS Sharpe `0.111225` and OOS return `+0.8284%`; lower numbers are not better.
3. Next viable direction: regime-aware ETH shock model or true OI/funding replay once train/val/OOS feature history exists.

## Verification

- Full repo ruff passed.
- Compileall passed.
- Targeted pytest passed: `51 passed in 0.22s`.
- Continuation validator passed: `var/reports/profit_moonshot_20260501/current_tail_20260505/continuation_validation_20260505_new_alpha.json`.
