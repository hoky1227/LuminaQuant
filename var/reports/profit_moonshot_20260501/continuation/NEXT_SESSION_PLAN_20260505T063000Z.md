# Profit Moonshot continuation — current-tail OOS gate restored

Generated: `2026-05-05T06:30:00Z`

## Current state

- Started from latest `private/main` on `private-main` before this run (`20ce7529b404fcbbf0d3158cb6b747d59da2b0b3`).
- Duplicate process checks found no competing heavy backtests before launching the one-mode-at-a-time runs.
- Latest data tail refresh completed at `2026-05-05T04:14:33Z` from `binance_raw_aggtrades`.
- OOS materialized coverage was repaired for `2026-05-03` and `2026-05-04` across BTC/ETH/BNB/SOL/TRX.
- All heavy operations stayed below the 8GB RSS guard.

## Candidate decision

- Deployment-review candidate: `profit_moonshot_leadlag_slow_diffusion_mode`.
  - train `+3.1274%`, MDD `9.1302%`, liquidations `0`
  - val `+0.6833%`, MDD `1.2601%`, liquidations `0`
  - OOS `+0.2910%`, MDD `7.0817%`, Sharpe `0.004059`, Sortino `0.004142`, liquidations `0`
- Conservative fallback retained but not deployment-ready: `profit_moonshot_momentum_hybrid_safe_mode`.
  - OOS `-0.3342%`, Sharpe `-0.001411`, Sortino `-0.001416`.

## Fresh raw-first results

| mode | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | user gate |
|---|---:|---:|---:|---:|---:|---|
| `profit_moonshot_leadlag_slow_diffusion_mode` | +3.1274% | +0.6833% | +0.2910% | 7.0817% | 0.004059 | PASS |
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | +0.2837% | -0.3342% | 3.4942% | -0.001411 | FAIL |

## Key artifacts

- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/session_current_tail_oos_report_20260505.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/data_refresh_20260505.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_leadlag_slow_diffusion_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_momentum_hybrid_safe_mode/live_equivalent_revalidation_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260505/external_alpha_screen/external_alpha_screen_20260505.md`

## Next priority

1. Treat `profit_moonshot_leadlag_slow_diffusion_mode` as the bar for any future alpha: complete raw-first train/val/OOS and positive OOS gate required.
2. Keep `profit_moonshot_momentum_hybrid_safe_mode` as conservative fallback only; do not deploy it while OOS remains negative.
3. Do not use gross exposure increases as an improvement path.
4. Funding/taker flow needs a new feature-replay contract or better raw inputs before another full live-equivalent run; current cheap screen had zero survivors.
