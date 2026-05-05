# Profit Moonshot Research Summary

Generated: `2026-05-05T11:25:00Z`; operator override patched: `2026-05-05T11:25:43Z`
Decision: `operator_oos_override_candidate_found`
Candidates scanned by generated ranker: `1050`
Promotion-eligible candidates by generated val-ranker: `35`

## Operator OOS override candidate

- Candidate: `profit_moonshot_hourly_shock_reversion_eth_12h_mode`
- Source: `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_mode/live_equivalent_revalidation_latest.json`
- OOS return / MDD: `+0.8284%` / `0.2819%`
- OOS Sharpe / Sortino: `0.100651` / `0.128321`
- OOS trades / liquidations: `30` / `0`
- Rationale: generated ranker is validation-split biased; user gate requires complete raw-first train/val/OOS evidence. This is current OOS-return best, not deployment-ready.

## Sharpe-focused shadow

- Candidate: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`
- OOS return / MDD: `+0.7206%` / `0.1778%`
- OOS Sharpe / Sortino: `0.111225` / `0.135831`
- Versus OOS-return best: Sharpe `+10.51%`, MDD reduction `36.93%`, return `-13.01%`.
- Decision: keep as risk-adjusted shadow; do not call it deployment-ready because Sharpe is still below `1.0`.

## External material used

- [Binance USDⓈ-M Futures: Open Interest Statistics](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- [Binance USDⓈ-M Futures: Taker Buy/Sell Volume](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume)
- [Binance USDⓈ-M Futures: Funding Rate History](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
- [arXiv 2212.06888: Crypto perpetual futures / funding context](https://arxiv.org/abs/2212.06888)

## Generated val-ranker promoted candidate (not operator-promoted)

- Generated ranker candidate: `profit_moonshot_momentum_hybrid_safe_mode`
- Generated primary split: `val` return `+0.2837%` Sharpe `0.010168`
- Operator status: not promoted because it fails the stricter current-tail OOS gate / Sharpe demand.

## Top Ranked Candidates

| rank | candidate | source | split | return | MDD | Sharpe | Sortino | trades | liq | final equity | blockers |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 2 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 3 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 4 | `profit_moonshot_hourly_shock_reversion_eth_mode` | `live_equivalent` | `val` | 1.04% | 0.88% | 0.066 | 0.068 | 66 | 0 | 10104.251 | - |
| 5 | `profit_moonshot_derivatives_taker_flow_sparse_mode` | `live_equivalent` | `val` | 0.08% | 0.05% | 0.055 | 0.027 | 108 | 0 | 10007.985 | - |
| 6 | `profit_moonshot_filtered_shock_reversion_diversified_mode` | `live_equivalent` | `val` | 0.09% | 1.34% | 0.003 | 0.003 | 182 | 0 | 10008.580 | - |
| 7 | `profit_moonshot_momentum_hybrid_core_mode` | `live_equivalent` | `val` | 0.25% | 1.01% | 0.009 | 0.009 | 138 | 0 | 10025.524 | - |
| 8 | `profit_moonshot_momentum_hybrid_return_mode` | `live_equivalent` | `val` | 0.27% | 1.01% | 0.010 | 0.009 | 134 | 0 | 10026.897 | - |
| 9 | `profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode` | `live_equivalent` | `val` | 0.63% | 0.44% | 0.071 | 0.081 | 48 | 0 | 10063.326 | - |
| 10 | `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | `live_equivalent` | `val` | 0.63% | 0.44% | 0.071 | 0.081 | 47 | 0 | 10063.269 | - |
| 11 | `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` | `live_equivalent` | `val` | 0.50% | 0.27% | 0.063 | 0.072 | 47 | 0 | 10050.235 | - |
| 12 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 10000.001 | - |
| 13 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 0.000 | - |
| 14 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 15 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 16 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 0.000 | - |
| 17 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 10026.530 | - |
| 18 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 19 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 0.000 | - |
| 20 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 21 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 0.000 | - |
| 22 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 10032.962 | - |
| 23 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 10039.730 | - |
| 24 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 0.000 | - |
| 25 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 10041.849 | - |

## Blocker Summary

- `val_sharpe_not_positive`: 10
- `val_sortino_not_positive`: 10
- `val_total_return_not_positive`: 10
- `train_trade_count_below_min`: 5
- `val_trade_count_below_min`: 5
- `legacy_train_val_mdd_gate_failed`: 4
- `status=ready_for_live_equivalent_backtest`: 4
- `status_not_validated:ready_for_live_equivalent_backtest`: 4

## New alpha follow-up — taker-flow exhaustion

- Implemented `TakerFlowExhaustionReversalStrategy` with true taker-flow feature requirement, funding cap, realized-vol cap, UTC session filter, same `target_allocation=0.008` / `max_order_value=175.0`, and a cooldown risk-control variant.
- Screen artifact: `var/reports/profit_moonshot_20260501/current_tail_20260505/taker_flow_exhaustion_screen/taker_flow_exhaustion_screen_20260505.json` (`1,296,000` combinations, `506` survivors, peak RSS `2577.43 MiB`).
- Full live-equivalent verdict: **no variant promoted**. Raw overlapping-event edge did not survive one-position/fee/partial-fill path realism.

| mode | train ret | val ret | OOS ret | OOS Sharpe | decision |
|---|---:|---:|---:|---:|---|
| `profit_moonshot_taker_flow_exhaustion_eth_mode` | +0.0089% | +0.0000% | +0.0000% | 0.000000 | rejected: zero val/OOS coverage after cadence/chunk phase |
| `profit_moonshot_taker_flow_exhaustion_eth_reactive_mode` | -0.0019% | +0.0210% | -0.0291% | -0.085988 | rejected: train/OOS negative |
| `profit_moonshot_taker_flow_exhaustion_eth_hold_mode` | -1.3901% | -0.0041% | -0.0875% | -0.031058 | rejected: wider hold worsened train and kept val/OOS negative |
| `profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode` | -0.2422% | -0.3210% | -0.0507% | -0.036304 | rejected: cooldown risk control still failed all splits |

Detailed report: `var/reports/profit_moonshot_20260501/current_tail_20260505/taker_flow_exhaustion_new_alpha_report_20260505.md`.
