# autonomous volcomp vwap reversion 5m decision

- generated_at: `2026-03-16T11:19:22.964108+00:00`
- status: `discard`
- decision_reason: Reused the prior autonomous_intraday_5m exact-window batch instead of launching a duplicate heavy run; both 5m volcomp_vwap_rev_guarded variants remained negative on train, validation, and locked OOS, so the family is discarded before any incumbent-anchor portfolio stage.
- reused_supporting_runs: `1`
- family_variant_count: `2`
- best_volcomp_variant: `volcomp_vwap_rev_guarded_5m_guarded_lo_strict_2.60_0.10`
- best_volcomp_val_total_return: -0.9415%
- best_volcomp_val_sharpe: -10.331
- best_volcomp_oos_total_return: -0.5574%
- best_volcomp_oos_sharpe: -1.725
- best_volcomp_oos_total_return_delta_vs_incumbent: -6.3202%
- best_volcomp_oos_sharpe_delta_vs_incumbent: -5.231
- best_volcomp_oos_max_drawdown_delta_vs_incumbent: -0.1336%
- representative_peak_rss_mib: 2690.7
