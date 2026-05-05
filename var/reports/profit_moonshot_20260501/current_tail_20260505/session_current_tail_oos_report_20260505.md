# Profit Moonshot current-tail OOS report — 2026-05-05

Generated: `2026-05-05T10:05:00Z`
Continuation base commit: `933442026ecc006f833087dd6164f51e62923e2a` (`private/main`).

## Decision

- **New best current-tail deployment-review candidate:** `profit_moonshot_hourly_shock_reversion_eth_12h_mode`.
  - OOS return is `+0.8284%` versus the prior weak shadow `+0.2910%`.
  - Train/val/OOS are all positive, OOS Sharpe/Sortino are `0.100651` / `0.128321`, OOS MDD is `0.2819%`, and liquidations are `0` on every split.
  - This is **not** a gross-exposure bump: single ETH sleeve, `target_allocation=0.008`, `max_order_value=175.0`, same conservative cap as the weak lead-lag sleeve.
- **Weak positive shadow only:** `profit_moonshot_leadlag_slow_diffusion_mode` remains a bar to beat, not the best candidate anymore.
- **Conservative fallback retained, not deployment-ready:** `profit_moonshot_momentum_hybrid_safe_mode` still fails the user OOS gate because current-tail OOS is negative.
- **Failed probes are retained with reasons:** `profit_moonshot_hourly_shock_reversion_eth_mode` was positive but weaker than the old bar; `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode` stayed OOS-negative.
- **One mode at a time / RSS guard respected:** no duplicate full-backtest process was present before launches; max RSS stayed well below 8GB.

## Fresh data / coverage repair

- Latest tail refresh cutoff: `2026-05-05T04:14:33Z`.
- Canonical source: `binance_raw_aggtrades`; aggregation backend: `rust:/home/hoky/Quants-agent/LuminaQuant/native/rust_rawfirst/target/release/liblumina_rawfirst.so`.
- Refreshed symbols: `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`.
- OOS committed-day gap repaired by materializing `2026-05-03` and `2026-05-04` 1s windows for BTC/ETH/BNB/SOL/TRX.

## Train / validation / OOS live-equivalent evidence

| mode | train return / MDD / liq / trades | val return / MDD / liq / trades | OOS return / MDD / Sharpe / Sortino / liq / trades | user OOS gate |
|---|---:|---:|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | +2.4523% / 2.1092% / 0 / 264 | +0.6323% / 0.4377% / 0 / 47 | +0.8284% / 0.2819% / 0.100651 / 0.128321 / 0 / 30 | PASS — new best |
| `profit_moonshot_leadlag_slow_diffusion_mode` | +3.1274% / 9.1302% / 0 / 176 | +0.6833% / 1.2601% / 0 / 40 | +0.2910% / 7.0817% / 0.004059 / 0.004142 / 0 / 40 | PASS but weak shadow |
| `profit_moonshot_hourly_shock_reversion_eth_mode` | +0.7538% / 1.2379% / 0 / 451 | +1.0422% / 0.8750% / 0 / 66 | +0.2716% / 0.5648% / 0.020248 / 0.024811 / 0 / 69 | FAIL — weaker than +0.2910% bar |
| `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode` | +8.8076% / 19.6453% / 0 / 354 | +0.4457% / 2.5293% / 0 / 54 | -0.3629% / 0.9596% / -0.024122 / -0.025618 / 0 / 34 | FAIL — OOS negative |
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% / 12.3695% / 0 / 1185 | +0.2837% / 1.0438% / 0 / 183 | -0.3342% / 3.4942% / -0.001411 / -0.001416 / 0 / 136 | FAIL — OOS negative |

Raw-first OOS coverage:

- `profit_moonshot_hourly_shock_reversion_eth_12h_mode`: ETH/USDT 65/65 missing=0 raw_first=true.
- `profit_moonshot_hourly_shock_reversion_eth_mode`: ETH/USDT 65/65 missing=0 raw_first=true.
- `profit_moonshot_leadlag_slow_diffusion_mode`: BTC/USDT 65/65 missing=0 raw_first=true; ETH/USDT 65/65 missing=0 raw_first=true.
- `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode`: ETH/USDT 65/65 missing=0 raw_first=true; SOL/USDT 65/65 missing=0 raw_first=true.
- `profit_moonshot_momentum_hybrid_safe_mode`: BNB/USDT 65/65 missing=0 raw_first=true; BTC/USDT 65/65 missing=0 raw_first=true; ETH/USDT 65/65 missing=0 raw_first=true; SOL/USDT 65/65 missing=0 raw_first=true; TRX/USDT 65/65 missing=0 raw_first=true.

Important interpretation: `revalidate_live_equivalent_candidates.py` treats OOS as report-only for its built-in recommendation object. This report applies the user's stricter promotion rule: **no complete positive OOS, no success**.

## New hourly shock reversion alpha family

Stateful screen source: `var/reports/profit_moonshot_20260501/current_tail_20260505/simple_hourly_alpha_screen/stateful_reversion_screen_20260505.json`. The selected family fades completed ETH hourly shocks; it uses completed 1h bars only and does not add exposure. The two implemented probes were:

| mode | screened thesis | live-equivalent result | decision |
|---|---|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | ETH reversion, 12h lookback, 72h hold, 1.0% trigger, 5% stop, 10% take-profit | OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651` | **new best** |
| `profit_moonshot_hourly_shock_reversion_eth_mode` | ETH reversion, 4h lookback, 48h hold, 0.6% trigger, 2% stop | OOS `+0.2716%`, MDD `0.5648%`, Sharpe `0.020248` | reject: weaker than old bar |

Cheap stateful screen top rows showed positive train/val/OOS edge for the two probes before full backtest, but final promotion is based only on live-equivalent evidence above.

## External alpha / feature replay screen

- Corrected funding/taker-flow screen: `funding_taker_flow` survivors `0`; previous TRX/BNB-looking rows were invalidated by the missing-flow `0/0` guard.
- Cross-crypto slow diffusion still has only the same two survivors; BTC→ETH is weak, SOL→ETH failed OOS in full live-equivalent.
- OI replay remains unsuitable for old train/val claims because the current support inventory has usable OI from early March 2026 onward; liquidation rows remain `0` for all tracked symbols.

## Failed / non-promoted candidates

- `profit_moonshot_hourly_shock_reversion_eth_mode`: train `+0.7538%`, val `+1.0422%`, OOS `+0.2716%`; positive but below the prior weak `+0.2910%` OOS bar, so rejected.
- `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode`: train `+8.8076%` but train MDD `19.6453%`, OOS `-0.3629%` with negative OOS Sharpe/Sortino; rejected.
- `profit_moonshot_momentum_hybrid_safe_mode`: retained as conservative train/val fallback, but current-tail OOS `-0.3342%` with negative OOS Sharpe/Sortino; not deployment-ready.
- `funding_taker_flow` family: rejected before full backtest; corrected cheap raw-first screens produced zero valid survivors after costs and OOS checks.
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
| simple_hourly_alpha_overlap_screen | 5:59.38 | 2795.70 | 0 |
| stateful_hourly_shock_reversion_screen | 2:12.29 | 2890.64 | 0 |
| hourly_shock_reversion_eth_4h_live_equivalent | 8:07.00 | 1307.52 | 0 |
| hourly_shock_reversion_eth_12h_live_equivalent | 8:02.04 | 1300.17 | 0 |

## Verification

- `uv run python scripts/research/profit_moonshot_research.py --input-dir var/reports/profit_moonshot_20260501 --output-dir var/reports/profit_moonshot_20260501 --max-files 650 --max-bytes 18000000 --generated-at 2026-05-05T10:10:00Z --print-json` — passed; summary patched with `operator_oos_override_candidate_found` because the generated ranker is validation-split biased.
- `uv run python scripts/research/validate_profit_moonshot_continuation.py --summary-path var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.json --result-path var/reports/profit_moonshot_20260501/current_tail_20260505/continuation_validation_20260505.json` — passed; candidate `profit_moonshot_hourly_shock_reversion_eth_12h_mode`, primary split `oos`, override active.
- `uv run ruff check` — passed full repo.
- Targeted pytest suite — `81 passed in 1.44s` (artifact portfolio mode, hourly shock strategy, research summary/validator, live selection, strategy factory, live-equivalent revalidation, derivatives-flow screen, historic data support, materialization).
- Live-equivalent backtests completed for `profit_moonshot_hourly_shock_reversion_eth_mode` and `profit_moonshot_hourly_shock_reversion_eth_12h_mode` with `Exit status: 0` and max RSS below 8GB.

## Artifacts

- New strategy: `src/lumina_quant/strategies/hourly_shock_reversion.py`.
- Stateful screen: `var/reports/profit_moonshot_20260501/current_tail_20260505/simple_hourly_alpha_screen/stateful_reversion_screen_20260505.json` / `var/reports/profit_moonshot_20260501/current_tail_20260505/simple_hourly_alpha_screen/stateful_reversion_screen_20260505.log`.
- Overlap screen: `var/reports/profit_moonshot_20260501/current_tail_20260505/simple_hourly_alpha_screen/simple_hourly_alpha_screen_20260505.json` / `var/reports/profit_moonshot_20260501/current_tail_20260505/simple_hourly_alpha_screen/simple_hourly_alpha_screen_20260505.log`.
- New best live-equivalent evidence: `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_mode/live_equivalent_revalidation_latest.json`.
- Rejected 4h shock live-equivalent evidence: `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_mode/live_equivalent_revalidation_latest.json`.
- Structured summary: `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.json`.

## Next stop condition

1. Treat `profit_moonshot_hourly_shock_reversion_eth_12h_mode` as the current best deployment-review candidate, but do not increase size without a new raw-first OOS gate.
2. Keep `profit_moonshot_leadlag_slow_diffusion_mode` only as weak shadow baseline and `profit_moonshot_momentum_hybrid_safe_mode` only as conservative fallback.
3. Any next candidate must beat OOS `+0.8284%`, keep positive train/val/OOS, zero liquidations, acceptable train drawdown, and RSS under 8GB.
