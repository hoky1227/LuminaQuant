# Profit Moonshot current-tail OOS / Sharpe follow-up — 2026-05-05

Generated: `2026-05-05T11:25:43Z`
Continuation base commit before this follow-up: `54baf68600f8767908df4c067f25d4ee36f26a7f` (`private/main`).

## Decision

- **No Sharpe≥1 deployment-ready candidate was found.** I am not counting any sub-1 Sharpe result as satisfying the new user demand.
- **OOS-return best remains:** `profit_moonshot_hourly_shock_reversion_eth_12h_mode` — OOS `+0.8284%`, MDD `0.2819%`, Sharpe `0.100651`, Sortino `0.128321`, liq `0`.
- **New risk-adjusted shadow:** `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` — OOS `+0.7206%`, MDD `0.1778%`, Sharpe `0.111225`, Sortino `0.135831`, liq `0`.
  - Versus incumbent: Sharpe `+10.51%`, Sortino `+5.85%`, MDD reduction `36.93%`, OOS return `-13.01%`.
- **Conservative fallback retained but not deployment-ready:** `profit_moonshot_momentum_hybrid_safe_mode` remains OOS-negative.
- **No gross exposure bump:** all new candidates keep `target_allocation=0.008` and `max_order_value=175`; diversified mode only splits that cap across sleeves.

## External material used

- [Binance USDⓈ-M Futures: Open Interest Statistics](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics) — Confirmed OI replay is an external derivative-state feature family, but current train/val history is too short for old claims.
- [Binance USDⓈ-M Futures: Taker Buy/Sell Volume](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume) — Used to constrain taker-flow ideas to symbols with nonzero raw feature support; BNB/TRX missing flow were rejected.
- [Binance USDⓈ-M Futures: Funding Rate History](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History) — Informed funding-settlement-hour entry exclusions around 00/08/16 UTC and adjacent windows.
- [arXiv 2212.06888: Crypto perpetual futures / funding context](https://arxiv.org/abs/2212.06888) — Background support for funding/crowding/perpetual-market alpha families; not used as promotion evidence.

## Fresh data / raw-first coverage

- Data tail refreshed through `2026-05-05T04:14:33Z`; feature rows through `2026-05-05T04:14:00Z` for BTC/ETH/BNB/SOL/TRX.
- BTC/ETH/SOL have nonzero taker-flow support; BNB/TRX taker-flow rows are zero and were excluded from flow-based promotion claims.
- Open-interest replay is available only from early March 2026 on this inventory, so it is not used for train/val promotion claims yet.
- Liquidation rows remain `0` for all tracked symbols; liquidation-feature alpha could not be honestly replayed.

## Train / validation / OOS live-equivalent evidence

| mode | role | train return / MDD / liq / trades | val return / MDD / liq / trades | OOS return / MDD / Sharpe / Sortino / liq / trades | decision |
|---|---|---:|---:|---:|---|
| `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | current OOS-return best / deployment-review only | +2.4523% / 2.1092% / 0 / 264 | +0.6323% / 0.4377% / 0 / 47 | +0.8284% / 0.2819% / 0.100651 / 0.128321 / 0 / 30 | Highest current-tail OOS return after full live-equivalent validation; still Sharpe<1, so not deployment-ready under the new Sharpe complaint. |
| `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` | new Sharpe/MDD-improved shadow | +1.3945% / 2.2014% / 0 / 254 | +0.5023% / 0.2682% / 0 / 47 | +0.7206% / 0.1778% / 0.111225 / 0.135831 / 0 / 34 | Funding-window entry exclusion improved OOS Sharpe and drawdown versus incumbent at identical cap, but lowered OOS return and still far below Sharpe 1. |
| `profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode` | rejected dense trigger | +0.5103% / 2.5341% / 0 / 261 | +0.6329% / 0.4377% / 0 / 48 | +0.4702% / 0.1668% / 0.078533 / 0.098619 / 0 / 38 | Lower 0.8% trigger matched validation but diluted train/OOS edge; OOS return and Sharpe below incumbent. |
| `profit_moonshot_filtered_shock_reversion_diversified_mode` | rejected diversified filtered sleeve | +0.6525% / 2.7466% / 0 / 1157 | +0.0858% / 1.3384% / 0 / 182 | +0.4849% / 1.4491% / 0.014267 / 0.014228 / 0 / 144 | SOL post-funding sleeve did not survive engine realism; partial fills/turnover caused weak OOS Sharpe and higher MDD. |
| `profit_moonshot_leadlag_slow_diffusion_mode` | weak positive shadow baseline | +3.1274% / 9.1302% / 0 / 176 | +0.6833% / 1.2601% / 0 / 40 | +0.2910% / 7.0817% / 0.004059 / 0.004142 / 0 / 40 | Positive but OOS MDD is too high and Sharpe is near zero; only a weak bar to beat. |
| `profit_moonshot_hourly_shock_reversion_eth_mode` | rejected 4h shock probe | +0.7538% / 1.2379% / 0 / 451 | +1.0422% / 0.8750% / 0 / 66 | +0.2716% / 0.5648% / 0.020248 / 0.024811 / 0 / 69 | Positive but OOS return below the prior weak +0.2910% bar. |
| `profit_moonshot_momentum_hybrid_safe_mode` | conservative fallback only / OOS failed | -1.3551% / 12.3695% / 0 / 1185 | +0.2837% / 1.0438% / 0 / 183 | -0.3342% / 3.4942% / -0.001411 / -0.001416 / 0 / 136 | Validation positive but train and OOS are negative; not promotable. |

## Filtered alpha family built/tested

- Stage screen: `var/reports/profit_moonshot_20260501/current_tail_20260505/filtered_hourly_shock_screen/stage_filtered_reversion_screen_20260505.json` checked `105300` raw-first variants; log shows max RSS `2097.85 MiB`, exit `0`.
- Implemented filters in `HourlyShockReversionStrategy`: UTC entry-hour allow/block lists, optional BTC/regime counterguard, and realized-volatility cap hooks.
- Funding-hour guard blocks entries at Binance 8h settlement/adjacent UTC hours `0,1,8,9,16,17`; exits remain unchanged.

| screened ETH 12h 0.8% profile | train sum/count | val sum/count | OOS sum/count | OOS trade-sharpe | full-engine decision |
|---|---:|---:|---:|---:|---|
| `no_funding_hours` | `+0.7208` / `125` | `+0.3506` / `18` | `+0.4609` / `16` | `9.086` | implemented as `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`; kept as Sharpe/MDD shadow |
| `plain` | `+0.5669` / `149` | `+0.2243` / `26` | `+0.5665` / `22` | `8.545` | implemented as `profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode`; rejected after OOS `+0.4702%` / Sharpe `0.078533` |
| `post_funding` | `+0.7656` / `58` | `+0.0958` / `9` | `+0.1600` / `7` | `8.520` | screened positive but not selected over no-funding-hours for engine run priority |
| `btc_counterguard_no_funding_hours` | `+0.1823` / `64` | `+0.1204` / `7` | `+0.2902` / `10` | `7.892` | too few validation/OOS trades for full promotion after screen |
| `btc_counterguard` | `+0.1340` / `74` | `+0.0560` / `8` | `+0.3637` / `13` | `8.684` | too few validation/OOS trades for full promotion after screen |

## Failed / non-promoted explanations

- `profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode`: `+0.4702%` OOS and Sharpe `0.078533`; lower trigger added lower-quality trades and did not beat incumbent.
- `profit_moonshot_filtered_shock_reversion_diversified_mode`: `+0.4849%` OOS, MDD `1.4491%`, Sharpe `0.014267`; SOL sleeve generated many realism partial fills, so rejected.
- Funding/taker-flow family: corrected support guards eliminated BNB/TRX missing-flow false positives; no valid full-flow survivor was promoted.
- OI/liquidation alpha: OI train/val history is too short and liquidation rows are zero, so no honest replay promotion was possible in this session.

## Resource guard evidence

| operation | elapsed | max RSS MiB | exit |
|---|---:|---:|---:|
| `data_tail_refresh` | 41:19.30 | 4589.08 | 0 |
| `materialize_oos_2026_05_03_04` | 0:12.20 | 1750.93 | 0 |
| `leadlag_first_run_train_val_oos_skipped` | 19:58.35 | 1904.75 | 0 |
| `leadlag_oos_rerun_after_materialization` | 3:21.15 | 1758.70 | 0 |
| `hybrid_safe_live_equivalent` | 50:45.56 | 3480.89 | 0 |
| `external_alpha_screen` | 0:54.90 | 2286.33 | 0 |
| `sol_eth_standalone_live_equivalent` | 16:27.06 | 1881.70 | 0 |
| `external_alpha_screen_all5_corrected` | 2:07.11 | 2619.11 | 0 |
| `simple_hourly_alpha_overlap_screen` | 5:59.38 | 2795.70 | 0 |
| `stateful_hourly_shock_reversion_screen` | 2:12.29 | 2890.64 | 0 |
| `hourly_shock_reversion_eth_4h_live_equivalent` | 8:07.00 | 1307.52 | 0 |
| `hourly_shock_reversion_eth_12h_live_equivalent` | 8:02.04 | 1300.17 | 0 |
| `stage_filtered_reversion_screen` | 0:18.49 | 2097.85 | 0 |
| `hourly_shock_reversion_eth_12h_funding_guard_live_equivalent` | 6:50.06 | 1289.82 | 0 |
| `filtered_shock_reversion_diversified_live_equivalent` | 13:32.31 | 1876.93 | 0 |
| `hourly_shock_reversion_eth_12h_dense_live_equivalent` | 6:13.85 | 1303.14 | 0 |

## Verification

- `uv run ruff check` — passed full repo.
- `uv run python -m compileall -q src tests` — passed.
- Targeted pytest — `43 passed in 0.15s` for hourly shock strategy, artifact portfolio mode, live selection, live-equivalent revalidation, derivatives-flow strategy, profit-moonshot research, and external-alpha screen tests.
- Continuation validator — passed: `profit_moonshot_hourly_shock_reversion_eth_12h_mode`, primary split `oos`, return `0.008284`, Sharpe `0.100651`, operator override active.
- Full live-equivalent backtests completed sequentially for `funding_guard`, `diversified`, and `dense` modes; all exited `0` and max RSS was below 8GB.
- Raw-first train/val/OOS coverage is complete for the ETH shock modes; existing OOS materialization covers `2026-03-01` through `2026-05-04`.

## Next stop condition

1. Do not promote anything as deployment-ready until a live-equivalent candidate clears the user’s stricter Sharpe demand or the metric definition is explicitly corrected and revalidated.
2. If optimizing return, benchmark against `profit_moonshot_hourly_shock_reversion_eth_12h_mode` OOS `+0.8284%`.
3. If optimizing risk-adjusted behavior, benchmark against `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` OOS Sharpe `0.111225` and MDD `0.1778%`.
4. Next new family should prioritize real nonzero taker-flow/OI/funding replay with enough train/val/OOS coverage, not extra gross exposure.
