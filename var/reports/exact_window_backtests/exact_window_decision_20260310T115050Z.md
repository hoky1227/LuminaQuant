# Exact-Window Decision Artifact

- Generated at: `2026-03-10T11:50:50.191241+00:00`
- Total evaluated: 81
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
| 15m | VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | -4.165 | 0 | 0 | 0 | -0.33% | -1.817 | oos_sharpe | 2379.78 |
| 30m | CompositeTrendStrategy | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | 12.241 | 0 | 0 | 0 | -0.43% | -0.175 | oos_sharpe | 2959.91 |
| 1h | PairSpreadZScoreStrategy | pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70 | -2.391 | 0 | 0 | 0 | -2.11% | -3.833 | oos_sharpe | 2379.78 |
| 4h | PairSpreadZScoreStrategy | pair_spread_4h_balanced_xptusdt_xpdusdt_1.6_0.35 | 26.888 | 0 | 0 | 0 | -4.22% | -18.519 | oos_sharpe, pbo | 768.94 |
| 1d | LagConvergenceStrategy | lag_convergence_1d_metals_patience_xptusdt_xpdusdt_2_0.015 | 40.715 | 0 | 0 | 0 | -3.02% | -7.576 | oos_sharpe, pbo | 755.57 |

## Reject Counts (All Selected Rows)

| Reason | Count |
|---|---:|
| oos_sharpe | 76 |
| pbo | 42 |
| trade_count | 26 |
| not_promoted | 1 |
| train_hurdle | 1 |
