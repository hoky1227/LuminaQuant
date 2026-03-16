# autonomous pair exec takeprofit probe

- generated_at: `2026-03-16T10:03:04.569038+00:00`
- lane: targeted 1h BNB/TRX pair execution-risk retune
- status: `discard`
- run_id: `autonomous_pair_exec_takeprofit_bnbtrx_1h_20260316T0958Z`
- evaluated_count: 18
- best_exec_variant: `pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70`
- sleeve_oos_return: 11.1034% vs incumbent pair 7.2375%
- anchor_portfolio_oos_return: 5.7647% vs incumbent 5.7628%
- peak_rss_mib: 2678.1

## takeaway

- Pair take-profit plus tighter-stop support is now empirically verified on the incumbent BNB/TRX sleeve under the 8 GiB memory contract.
- The sleeve-level gain did not survive at the anchored portfolio level, so the incumbent stays live.
- Next backlog suggestion: continue to the next unused backlog lane under ideas_backlog_latest.md, with leadlag_spillover 5m currently the top untested priority.
