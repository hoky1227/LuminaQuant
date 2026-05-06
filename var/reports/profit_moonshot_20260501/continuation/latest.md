# Profit Moonshot — useful-alpha execution status

Generated: `2026-05-06T12:55:59.857827Z`; updated: `2026-05-06T13:26:37Z`

## Current decision

No new successful alpha was promoted. The strict gate remains unmet: OOS return must exceed `+0.8284%`, OOS MDD must improve on `0.1778%`, and OOS Sharpe must be `>1.0` with separated train/val/OOS raw-first evidence.

## Evidence reports to read first next session

- `var/reports/profit_moonshot_20260501/current_tail_20260506/useful_alpha_execution_report_20260506.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/cadence_sweep/profit_moonshot_cadence_sweep_latest.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/eth_shock_filter_replay/eth_shock_filter_replay_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode_context_fix/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260506/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode/live_equivalent_revalidation_latest.json`

## Tested and rejected

| mode | train return | OOS return | OOS MDD | OOS Sharpe | decision |
|---|---:|---:|---:|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode` | `+0.0255%` | `+0.5871%` | `0.3203%` | `0.070688` | reject: below return/MDD/Sharpe gates |
| `profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode` | `+1.3156%` | `+0.3221%` | `0.9275%` | `0.014160` | reject: below return/MDD/Sharpe gates |
| `profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode` | `-0.2259%` | `+0.1759%` | `0.2204%` | `0.043476` | reject: train negative + train liquidation + below OOS return/MDD/Sharpe gates |
| `profit_moonshot_adaptive_momentum_boost_mode__cadence_1b` | `-100.0000%*` | `-0.9619%` | `33.0023%` | `0.012010` | discard/reject: train equity breach and `5` liquidations; raw train arithmetic was `-111.5894%` return / `175.7880%` MDD |

## Leverage/cadence conclusion

- Broad cadence screen covered `174` variants from frequent to low-frequency rebalancing, with spread/fee/slippage/partial-fill engine realism and no exposure increase.
- Best screen survivor collapsed under full train/val/OOS replay: train equity breach, train liquidations `5`, OOS return `-0.9619%`. Report-facing train loss/MDD is capped at `-100%`/`100%`; raw negative-equity arithmetic is preserved only in `raw_*` diagnostics.
- Operator-discarded candidate: `profit_moonshot_adaptive_momentum_boost_mode__cadence_1b`; do not revisit as useful alpha without a new liquidation-free train pre-gate.
- Safe bottleneck work completed: raw-first chunk cache, epoch-ms prefrozen-row reuse, memory trimming, cumulative liquidation carry, and Rust raw-first backend availability (`rust:/home/hoky/Quants-agent/LuminaQuant/native/rust_rawfirst/target/release/liblumina_rawfirst.so`).
- Remaining exactness bottleneck is structural: open leveraged positions require portfolio/funding/liquidation/order checks in the Python event/strategy state loop. Do not replace this with vector-only or non-stateful Rust unless parity tests cover fills, fees, liquidation, funding, cooldown, and chunk state.

## Ground truth remains

- OOS-return best: `profit_moonshot_hourly_shock_reversion_eth_12h_mode` — OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`.
- Sharpe/MDD shadow: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` — OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`.
- Metals direct alpha remains rejected due raw-first coverage and OOS failure.

## Next-session warning

Do not spend another full backtest on remaining cadence-screen winners unless a shorter exact train pre-gate survives without liquidation and without equity breach. Current `1b` adaptive-momentum variants are screen artifacts, not useful alpha.
