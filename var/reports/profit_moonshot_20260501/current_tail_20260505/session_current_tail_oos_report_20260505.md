# Profit Moonshot current-tail OOS report — 2026-05-05

Generated: `2026-05-05T06:30:00Z`
Base commit: `20ce7529b404fcbbf0d3158cb6b747d59da2b0b3` (`private/main`).

## Decision

- **Weak positive shadow baseline, not deployment-ready:** `profit_moonshot_leadlag_slow_diffusion_mode`.
  - This is the only refreshed current-tail candidate with complete raw-first train/val/OOS coverage, positive train/val/OOS return, positive OOS Sharpe/Sortino, and zero liquidations.
  - It is better than `profit_moonshot_momentum_hybrid_safe_mode` on OOS sign, but the edge is too small for a deployment-ready claim: OOS return is only `+0.2910%`, OOS MDD is `7.0817%`, and OOS Sharpe is `0.004059`.
- **Conservative fallback retained, not deployment-ready:** `profit_moonshot_momentum_hybrid_safe_mode`.
  - The framework marks it `live_equivalent_validated` from train/val selection, but the user-level OOS gate fails because OOS return, Sharpe, Sortino, and Calmar are negative.
- **No gross exposure increase was used.** The improvement comes from the cross-crypto slow-diffusion/lead-lag alpha family, not from size-only risk expansion.
- **One mode at a time / RSS guard respected:** every heavy run below the 8GB cap; no duplicate backtest process was present before launching the runs.

## Fresh data / coverage repair

- Latest tail refresh cutoff: `2026-05-05T04:14:33Z`.
- Canonical source: `binance_raw_aggtrades`; aggregation backend: `rust:/home/hoky/Quants-agent/LuminaQuant/native/rust_rawfirst/target/release/liblumina_rawfirst.so`.
- Refreshed symbols: `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`.
- OOS committed-day gap repaired by materializing `2026-05-03` and `2026-05-04` 1s windows for BTC/ETH/BNB/SOL/TRX.
- Materialized row counts:

| symbol | 2026-05-03 rows | 2026-05-04 rows |
|---|---:|---:|
| BTC/USDT | 86400 | 86398 |
| ETH/USDT | 86399 | 86398 |
| BNB/USDT | 86400 | 86398 |
| SOL/USDT | 86400 | 86398 |
| TRX/USDT | 86400 | 86397 |


## Train / validation / OOS live-equivalent evidence

| mode | train return / MDD / liq / trades | val return / MDD / liq / trades | OOS return / MDD / Sharpe / Sortino / liq / trades | user OOS gate |
|---|---:|---:|---:|---|
| `profit_moonshot_leadlag_slow_diffusion_mode` | +3.1274% / 9.1302% / 0 / 176 | +0.6833% / 1.2601% / 0 / 40 | +0.2910% / 7.0817% / 0.004059 / 0.004142 / 0 / 40 | PASS |
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% / 12.3695% / 0 / 1185 | +0.2837% / 1.0438% / 0 / 183 | -0.3342% / 3.4942% / -0.001411 / -0.001416 / 0 / 136 | FAIL |

Raw-first OOS coverage:

- `profit_moonshot_leadlag_slow_diffusion_mode`: BTC/USDT 65/65 missing=0 raw_first=true; ETH/USDT 65/65 missing=0 raw_first=true.
- `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode`: ETH/USDT 65/65 missing=0 raw_first=true; SOL/USDT 65/65 missing=0 raw_first=true.
- `profit_moonshot_momentum_hybrid_safe_mode`: BNB/USDT 65/65 missing=0 raw_first=true; BTC/USDT 65/65 missing=0 raw_first=true; ETH/USDT 65/65 missing=0 raw_first=true; SOL/USDT 65/65 missing=0 raw_first=true; TRX/USDT 65/65 missing=0 raw_first=true.

Important interpretation: `revalidate_live_equivalent_candidates.py` treats OOS as report-only for its built-in recommendation object. This report applies the user's stricter promotion rule: **no complete positive OOS, no success**.

## External alpha / feature replay screen

Cheap raw-first screen through `2026-05-04`:

- `funding_taker_flow`: **0 survivors**. Funding/taker candidates with good-looking validation either had negative train edge, negative OOS edge, or too few OOS events after the `0.1800%` roundtrip cost assumption.
- `cross_crypto_slow_diffusion`: **2 screen survivors**. The top BTC→ETH survivor is implemented but weak; the second SOL→ETH survivor was fully backtested after the weak-result challenge and rejected on OOS.
- OI replay remains unsuitable for old OOS claims because the current support inventory only has OI from early March 2026 onward; liquidation rows remain `0` for all tracked symbols.

| survivor | train n/edge | val n/edge | OOS n/edge | OOS hit |
|---|---:|---:|---:|---:|
| BTC/USDT→ETH/USDT lag=2h horizon=8h params={'leader_abs_ret_min': 0.015, 'target_underreaction_cap': 999.0} | 359 / +0.0591% | 93 / +0.1047% | 63 / +0.0456% | 52.3810% |
| SOL/USDT→ETH/USDT lag=1h horizon=8h params={'leader_abs_ret_min': 0.015, 'target_underreaction_cap': 999.0} | 700 / +0.0372% | 95 / +0.0540% | 49 / +0.0376% | 57.1429% |

## Reality check after weak positive result

The `+0.2910%` OOS result is positive but economically weak. After that challenge, the second raw-first lead-lag survivor was wired as a standalone equal-risk replacement probe and fully revalidated:

| mode | train return / MDD / liq / trades | val return / MDD / liq / trades | OOS return / MDD / Sharpe / Sortino / liq / trades | decision |
|---|---:|---:|---:|---|
| `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode` | +8.8076% / 19.6453% / 0 / 354 | +0.4457% / 2.5293% / 0 / 54 | -0.3629% / 0.9596% / -0.024122 / -0.025618 / 0 / 34 | REJECT: OOS negative and train MDD too high |

The 5-symbol external alpha screen was also rerun after fixing a taker-flow `0/0` NaN issue in the funding/taker screen. The corrected all-symbol screen has `funding_taker_flow` survivors `0` and only the same two lead-lag survivors, so there is no stronger current-tail raw-first candidate ready for another full backtest.

## Failed / non-promoted candidates

- `profit_moonshot_momentum_hybrid_safe_mode`: retained as conservative train/val fallback, but rejected for deployment because current-tail OOS is `-0.3342%` with negative OOS Sharpe/Sortino.
- `profit_moonshot_leadlag_slow_diffusion_mode` first current-tail run: train/val completed, but OOS was skipped because committed materialized days `2026-05-03` and `2026-05-04` were missing. This is not success evidence; success evidence is only the rerun after materialization.
- `funding_taker_flow` family: rejected before full backtest; corrected cheap raw-first screens produced zero survivors after costs and OOS checks. The temporary all-symbol TRX survivors were invalidated as missing-flow/NaN artifacts before any full backtest.
- Prior same-risk lead-lag ensemble: remains rejected from the previous follow-up because OOS was negative (`-0.3059%` in `external_alpha/leadlag_ensemble_followup_report_20260503.md`).
- Prior adaptive boost raw-val leader (`profit_moonshot_adaptive_momentum_boost_mode`): still not promotable because train return/MDD were unacceptable in the handoff state.

## Resource guard evidence

| operation | elapsed | max RSS MiB | exit |
|---|---:|---:|---:|
| data_tail_refresh | 41:19.30 | 4589.08 | 0 |
| materialize_oos_2026_05_03_04 | 0:12.20 | 1750.93 | 0 |
| leadlag_first_run_train_val_oos_skipped | 19:58.35 | 1904.75 | 0 |
| leadlag_oos_rerun_after_materialization | 3:21.15 | 1758.70 | 0 |
| hybrid_safe_live_equivalent | 50:45.56 | 3480.89 | 0 |
| external_alpha_screen | 0:54.90 | 2286.33 | 0 |
| sol_eth_standalone_live_equivalent | 16:27.06 | 1881.70 | 0 |
| external_alpha_screen_all5_corrected | 2:07.11 | 2619.11 | 0 |


## Verification

- `uv run python scripts/research/profit_moonshot_research.py --input-dir var/reports/profit_moonshot_20260501 --output-dir var/reports/profit_moonshot_20260501 --max-files 400 --max-bytes 12000000 --generated-at 2026-05-05T09:10:00Z --print-json` — passed and refreshed the bounded research summary.
- `uv run python scripts/research/validate_profit_moonshot_continuation.py --summary-path var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.json --result-path var/reports/profit_moonshot_20260501/current_tail_20260505/continuation_validation_20260505.json` — passed; validator still finds the best validation-return candidate but the operator OOS override says no deployment-ready candidate.
- `uv run ruff check` — passed.
- Targeted pytest suite (`test_artifact_portfolio_mode`, `test_profit_moonshot_research`, `test_live_selection_infer`, `test_profit_moonshot_strategies`, `test_strategy_factory_library`, `test_adaptive_regime_momentum`, `test_live_equivalent_revalidation`, `test_derivatives_flow_squeeze_strategy`, `test_historic_data_feature_support`, `test_screen_profit_moonshot_external_alpha`, `test_materialize_from_raw`) — `75 passed in 3.16s`.
- Corrected all-symbol screen rerun after the zero-denominator guard: `funding_survivors=0`, `leadlag_survivors=2`, peak RSS `2619.11 MiB`.

## Artifacts

- Data refresh: `var/reports/profit_moonshot_20260501/current_tail_20260505/data_refresh_20260505.md` / `var/reports/profit_moonshot_20260501/current_tail_20260505/data_refresh_20260505.json`.
- OOS materialization log: `var/reports/profit_moonshot_20260501/current_tail_20260505/materialize_oos_20260503_04/run.log`.
- Lead-lag live-equivalent evidence: `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_leadlag_slow_diffusion_mode/live_equivalent_revalidation_latest.json`.
- Hybrid-safe live-equivalent evidence: `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_momentum_hybrid_safe_mode/live_equivalent_revalidation_latest.json`.
- External alpha screen: `var/reports/profit_moonshot_20260501/current_tail_20260505/external_alpha_screen/external_alpha_screen_20260505.md` / `var/reports/profit_moonshot_20260501/current_tail_20260505/external_alpha_screen/external_alpha_screen_20260505.json`.
- Corrected 5-symbol external alpha screen: `var/reports/profit_moonshot_20260501/current_tail_20260505/external_alpha_screen_all5/external_alpha_screen_all5_20260505.md` / `var/reports/profit_moonshot_20260501/current_tail_20260505/external_alpha_screen_all5/external_alpha_screen_all5_20260505.json`.
- SOL→ETH standalone live-equivalent rejection: `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_leadlag_slow_diffusion_sol_eth_mode/live_equivalent_revalidation_latest.json`.
- Structured summary: `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.json`.

## Next stop condition

1. Do not promote `hybrid_safe` unless a future raw-first OOS rerun flips positive without gross-exposure-only changes.
2. If continuing alpha research, run at most one full live-equivalent backtest for one newly implemented lead-lag survivor/funding-taker variant at a time.
3. Next candidate must materially beat the weak refreshed lead-lag bar: complete raw-first train/val/OOS, OOS return well above `+0.2910%`, positive OOS Sharpe/Sortino, acceptable train drawdown, zero liquidations, and RSS under 8GB.
