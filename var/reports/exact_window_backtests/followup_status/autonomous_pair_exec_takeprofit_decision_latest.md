# autonomous pair execution-risk decision

- generated_at: `2026-03-16T10:03:04.569038+00:00`
- status: `discard`
- decision_reason: Pair execution-risk retune materially improved the BNB/TRX sleeve itself, but the anchored portfolio still failed the incumbent on validation robustness plus locked-OOS Sharpe and max drawdown; keep the current one-shot baseline.
- best_exec_variant: `pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70`
- current_pair_reference: `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55`
- sleeve_oos_total_return_delta: 3.8659%
- sleeve_oos_sharpe_delta: 2.924
- anchor_oos_total_return_delta_vs_incumbent: 0.0019%
- anchor_oos_sharpe_delta_vs_incumbent: -0.109
- anchor_oos_max_drawdown_delta_vs_incumbent: 0.8797%
- peak_rss_mib: 2678.1

## takeaway

- Execution-risk/take-profit variants clearly improved the incumbent BNB/TRX pair sleeve at the sleeve level.
- The anchored portfolio still failed to beat the incumbent on the locked-OOS decision surface because Sharpe fell and drawdown widened materially.
- Recommended next action: continue to the next unused backlog lane under ideas_backlog_latest.md, with leadlag_spillover 5m currently the top untested priority.
