# Exact-Window Decision Artifact

- Generated at: `2026-03-10T12:06:42.641974+00:00`
- Total evaluated: 95
- Promoted total: 0
- BTC-beating candidate total: 0
- Recent-3M 2% candidate total: 0
- Provisional candidate total: 0
- Candidate pool total: 0
- Valid strategy found: `False`
- Next action: `ralplan_team_ralph_required`
- Clamp timestamp: ``

## Timeframe Best Rows

| TF | Strategy | Name | Val Score | Promoted | BTC-beat | 3M>=2% | OOS Return | OOS Sharpe | Rejects | Peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 15m | VwapReversionStrategy | vwap_reversion_15m_guarded_lo_48_0.014 | -4.549 | 0 | 0 | 0 | -1.21% | -0.153 | oos_sharpe | 2175.19 |
| 30m | RollingBreakoutStrategy | rolling_breakout_30m_guarded_ls_64_0.002 | 19.343 | 0 | 0 | 0 | 4.08% | 1.042 | pbo | 2175.19 |
| 1h | RegimeBreakoutCandidateStrategy | regime_breakout_1h_trend_ls_48_0.70 | 11.976 | 0 | 0 | 0 | 8.08% | 2.009 | train_hurdle | 2175.19 |
| 4h | PairSpreadZScoreStrategy | pair_spread_4h_balanced_xptusdt_xpdusdt_1.6_0.35 | 26.888 | 0 | 0 | 0 | -4.22% | -18.519 | oos_sharpe, pbo | 768.94 |
| 1d | LagConvergenceStrategy | lag_convergence_1d_metals_patience_xptusdt_xpdusdt_2_0.015 | 40.715 | 0 | 0 | 0 | -3.02% | -7.576 | oos_sharpe, pbo | 755.57 |

## Reject Counts (All Selected Rows)

| Reason | Count |
|---|---:|
| oos_sharpe | 84 |
| pbo | 45 |
| trade_count | 26 |
| train_hurdle | 4 |
