# autonomous lag convergence 4h probe

- generated_at: `2026-03-16T11:43:09.504758+00:00`
- lane: reused lag_convergence 4h exact-window probe plus incumbent-anchor portfolio comparison
- status: `discard`
- reused_supporting_runs: `1`
- lag_variant_count: `6`
- representative_variant: `lag_convergence_4h_fast_1_0014_xptusdt_xpdusdt_1_0.014`
- representative_variant_oos_return: 1.6941%
- anchor_oos_return: 3.1332%
- incumbent_oos_return: 5.7628%

## takeaway

- The focused XPT/XPD lag-convergence 4h probe already existed and showed a few positive sleeves, so the only new compute step was an anchored four-sleeve comparison against the current incumbent.
- The lag-convergence anchor improved locked-OOS Sharpe and drawdown, but it still lowered locked-OOS total return versus the incumbent, so the incumbent stays in place and the next lane should move to rolling_breakout 30m.
