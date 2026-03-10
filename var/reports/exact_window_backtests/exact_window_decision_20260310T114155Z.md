# Exact-Window Decision Artifact

- Generated at: `2026-03-10T11:41:55.239309+00:00`
- Total evaluated: 105
- Promoted total: 0
- BTC-beating candidate total: 0
- Recent-3M 2% candidate total: 0
- Provisional candidate total: 0
- Candidate pool total: 0
- Valid strategy found: `False`
- Next action: `ralplan_team_ralph_required`
- Clamp timestamp: `2026-03-07T10:00:00+00:00`

## Timeframe Best Rows

| TF | Strategy | Name | Val Score | Promoted | BTC-beat | 3M>=2% | OOS Return | OOS Sharpe | Rejects | Peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| 15m | VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | -4.165 | 0 | 0 | 0 | -0.33% | -1.817 | oos_sharpe | 2379.78 |
| 30m | CompositeTrendStrategy | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | 12.241 | 0 | 0 | 0 | -0.43% | -0.175 | oos_sharpe | 2959.91 |
| 1h | PairSpreadZScoreStrategy | pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70 | -2.391 | 0 | 0 | 0 | -2.11% | -3.833 | oos_sharpe | 2379.78 |
| 4h | PairSpreadZScoreStrategy | pair_spread_4h_fast_cycle_btcusdt_ethusdt_1.6_0.35 | 10.345 | 0 | 0 | 0 | 2.49% | 2.899 | pbo | 2059.32 |

## Reject Counts (All Selected Rows)

| Reason | Count |
|---|---:|
| oos_sharpe | 53 |
| pbo | 49 |
| trade_count | 12 |
| train_hurdle | 5 |
| not_promoted | 1 |
