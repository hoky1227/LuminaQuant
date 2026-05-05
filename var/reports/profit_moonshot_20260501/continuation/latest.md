# Profit Moonshot continuation — Sharpe follow-up

Generated: `2026-05-05T11:25:43Z`

## Current state

- Latest `private/main` base before this follow-up: `54baf68600f8767908df4c067f25d4ee36f26a7f`.
- Latest data tail remains refreshed through `2026-05-05T04:14:33Z`; OOS materialized coverage is complete through `2026-05-04`.
- OOS-return best is still `profit_moonshot_hourly_shock_reversion_eth_12h_mode`: return `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`, Sortino `0.128321`.
- New Sharpe/MDD shadow is `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`: return `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`, Sortino `0.135831`.
- No candidate reached Sharpe `1.0`; do not call this deployment-ready.
- Conservative `profit_moonshot_momentum_hybrid_safe_mode` remains fallback only and OOS-negative.

## Fresh Sharpe follow-up table

| mode | train ret | train MDD | val ret | OOS ret | OOS MDD | OOS Sharpe | user decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | +2.4523% | 2.1092% | +0.6323% | +0.8284% | 0.2819% | 0.100651 | current OOS-return best / deployment-review only |
| `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` | +1.3945% | 2.2014% | +0.5023% | +0.7206% | 0.1778% | 0.111225 | new Sharpe/MDD-improved shadow |
| `profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode` | +0.5103% | 2.5341% | +0.6329% | +0.4702% | 0.1668% | 0.078533 | rejected dense trigger |
| `profit_moonshot_filtered_shock_reversion_diversified_mode` | +0.6525% | 2.7466% | +0.0858% | +0.4849% | 1.4491% | 0.014267 | rejected diversified filtered sleeve |
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | 12.3695% | +0.2837% | -0.3342% | 3.4942% | -0.001411 | conservative fallback only / OOS failed |

## Key artifacts

- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/filtered_hourly_shock_screen/stage_filtered_reversion_screen_20260505.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_filtered_shock_reversion_diversified_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.json`

## Next priority

1. Beat OOS Sharpe `0.111225` and preferably return `+0.8284%` without changing gross exposure.
2. Do not reuse BNB/TRX taker-flow rows until support inventory shows nonzero flow coverage.
3. OI/funding/liquidation family needs enough train/val/OOS replay coverage before another full backtest.
