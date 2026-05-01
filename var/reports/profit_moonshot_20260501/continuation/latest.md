# Profit Moonshot Continuation — adaptive boost

Generated: `2026-05-01T13:15:36.885385Z`
Status: `passed`

## Conclusion

- New useful candidate found: `profit_moonshot_adaptive_momentum_boost_mode`.
- Validation return improved from `0.264933%` to `0.509082%`.
- This is not a moonshot-level return, but it is a live-equivalent, gate-passing PnL improvement over the previous best.

## Live-equivalent gate evidence

| metric | value |
|---|---:|
| train return | -2.994796% |
| train MDD | 18.021085% |
| train trades | 361 |
| val return | 0.509082% |
| val MDD | 1.358270% |
| val Sharpe | 0.014751 |
| val Sortino | 0.014527 |
| val trades | 56 |
| train/val liquidations | 0/0 |
| alpha blockers | `-` |

## Runtime evidence

- Backtest command: `uv run python scripts/research/revalidate_live_equivalent_candidates.py --portfolio-modes profit_moonshot_adaptive_momentum_boost_mode --execute-backtests --chunk-days 7 --no-live-decision-update`
- Max RSS: `5,821,420 KB` (< 8GB).
- Output: `var/reports/profit_moonshot_20260501/continuation/adaptive_boost/live_equivalent_revalidation_latest.json`
- Autoresearch result: `.omx/specs/autoresearch-profit-moonshot-continuation/result.json`

## Caveat

- Train return is close to the gate floor (`-2.9948%` vs allowed `>-3%`), so this should be treated as a candidate for further robustness checks rather than immediate deployment.
- The conservative summary rank still favors the lower-drawdown baseline; by raw validation PnL, the boost candidate is better.
