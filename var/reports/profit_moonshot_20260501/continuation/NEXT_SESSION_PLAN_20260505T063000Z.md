# Profit Moonshot continuation — weak OOS positive downgraded

Generated: `2026-05-05T09:05:00Z`

## Current state

- Started this continuation from `private/main` commit `20ce7529b404fcbbf0d3158cb6b747d59da2b0b3`; follow-up evidence now sits on top of commit `714e99f4a6580093aa0b78e28377203eb26cf454`.
- Latest data tail remains refreshed through `2026-05-05T04:14:33Z`; OOS materialized coverage is complete through `2026-05-04`.
- The positive BTC→ETH lead-lag result is now classified as **weak shadow baseline, not deployment-ready** because OOS is only `+0.2910%` with Sharpe `0.004059` and MDD `7.0817%`.
- The second raw-first lead-lag survivor was tested as `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode` at the same `0.8%` target allocation. It is rejected: train `+8.8076%` but MDD `19.6453%`, val `+0.4457%`, OOS `-0.3629%`, OOS Sharpe `-0.024122`, liquidations `0`.
- A 5-symbol external alpha screen initially exposed false TRX funding survivors from missing taker-flow `0/0` NaN flow. The screen now guards zero denominators; corrected all-symbol result has `funding_taker_flow` survivors `0` and the same two lead-lag survivors only.
- All heavy operations stayed below 8GB RSS; SOL→ETH full run peak RSS was `1881.70 MiB`, corrected all5 screen peak RSS was `2619.11 MiB`.

## Current candidate decision

- **No deployment-ready profit moonshot candidate.**
- Weak positive shadow baseline: `profit_moonshot_leadlag_slow_diffusion_mode`.
- Conservative fallback only: `profit_moonshot_momentum_hybrid_safe_mode`; still OOS negative.
- Rejected follow-up: `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode`.
- Funding/taker-flow remains blocked until real replay produces nonzero flow survivors after corrected screening.

## Fresh raw-first results

| mode | train ret | train MDD | val ret | OOS ret | OOS MDD | OOS Sharpe | user decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `profit_moonshot_leadlag_slow_diffusion_mode` | +3.1274% | 9.1302% | +0.6833% | +0.2910% | 7.0817% | 0.004059 | weak shadow only |
| `profit_moonshot_leadlag_slow_diffusion_sol_eth_mode` | +8.8076% | 19.6453% | +0.4457% | -0.3629% | 0.9596% | -0.024122 | reject |
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | 12.3695% | +0.2837% | -0.3342% | 3.4942% | -0.001411 | reject deployment |

## Key artifacts

- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_leadlag_slow_diffusion_sol_eth_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/external_alpha_screen_all5/external_alpha_screen_all5_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/external_alpha_screen_all5/external_alpha_screen_all5_20260505.json`

## Next priority

1. Do not deploy the weak positive lead-lag sleeve; use it only as a bar to beat.
2. Next full backtest should require a cheap screen edge materially above the current BTC→ETH lead-lag (`OOS +0.2910%`) and must not come from missing-flow artifacts.
3. Funding/taker-flow needs real replay coverage for the symbols screened; no full backtest from NaN/zero-flow derived candidates.
4. Continue one mode at a time, RSS <8GB, and record failures explicitly.
