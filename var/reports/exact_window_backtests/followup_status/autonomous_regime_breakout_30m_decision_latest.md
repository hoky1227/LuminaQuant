# autonomous regime breakout 30m decision

- generated_at: `2026-03-16T11:56:12.684820+00:00`
- status: `discard`
- decision_reason: Reused the existing regime_breakout 30m fourth-sleeve probe and anchored optimization artifacts instead of launching a duplicate heavy run. The regime-breakout sleeve remained positive on validation and locked OOS, but it still failed train/validation hurdle quality, and the anchored challenger trailed the incumbent on locked-OOS total return, Sharpe, and max drawdown.
- reused_supporting_runs: `1`
- representative_variant: `regime_breakout_30m_trend_ls_64_0.72`
- candidate_val_total_return: 2.0039%
- candidate_oos_total_return: 1.2731%
- anchor_oos_total_return: 5.0532%
- incumbent_oos_total_return: 5.7628%
- anchor_oos_total_return_delta_vs_incumbent: -0.7096%
- anchor_oos_sharpe_delta_vs_incumbent: -0.166
- anchor_oos_max_drawdown_delta_vs_incumbent: 0.0061%
- source_peak_rss_mib: 3219.5
