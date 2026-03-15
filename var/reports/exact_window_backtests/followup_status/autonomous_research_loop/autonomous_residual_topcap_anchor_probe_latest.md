# autonomous residual topcap anchor probe

- generated_at: `2026-03-15T07:10:49.810447+00:00`
- lane: residual/factor-neutral topcap momentum replacement
- status: `discard`
- exact_window_evaluated_count: `33`
- residual_candidate: `topcap_tsmom_1h_resid_beta_neutral_24_4_0.008`
- residual_candidate_oos_return: 4.3041%
- anchored_portfolio_oos_return: 4.8804% vs incumbent 5.5960%
- anchored_portfolio_oos_sharpe: 2.936 vs incumbent 3.447

## takeaway

- Residual beta-neutralization improved the isolated topcap sleeve OOS return, but the anchored portfolio still failed the locked-OOS gate.
- Continue to deterministic crash-aware momentum gates next.
