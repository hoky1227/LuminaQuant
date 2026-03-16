# autonomous rolling breakout 30m decision

- generated_at: `2026-03-16T11:50:16.032678+00:00`
- status: `discard`
- decision_reason: Reused the existing rolling_breakout 30m regime-gate and fourth-sleeve probe/anchored optimization artifacts instead of launching a duplicate heavy run. The gated breakout sleeve stayed positive on validation and sleeve-level locked OOS, but it still failed robustness gates, and the anchored portfolio challenger trailed the incumbent on locked-OOS total return, Sharpe, and max drawdown.
- reused_supporting_runs: `1`
- representative_variant: `rolling_breakout_30m_guarded_ls_64_0.002`
- candidate_val_total_return: 10.0008%
- candidate_oos_total_return: 7.2152%
- anchor_oos_total_return: 5.1829%
- incumbent_oos_total_return: 5.7628%
- anchor_oos_total_return_delta_vs_incumbent: -0.5799%
- anchor_oos_sharpe_delta_vs_incumbent: -0.257
- anchor_oos_max_drawdown_delta_vs_incumbent: 0.2175%
- breakout_selected_in_anchor: `False`
