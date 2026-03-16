# autonomous volcomp vwap reversion 5m probe

- generated_at: `2026-03-16T11:19:22.964108+00:00`
- lane: reused volcomp_vwap_rev_guarded 5m exact-window evidence
- status: `discard`
- reused_supporting_runs: `1`
- family_variant_count: `2`
- best_variant: `volcomp_vwap_rev_guarded_5m_guarded_lo_strict_2.60_0.10`
- best_variant_val_return: -0.9415%
- best_variant_oos_return: -0.5574%
- incumbent_oos_return: 5.7628%

## takeaway

- The reused autonomous_intraday_5m batch already screened both guarded 5m vol-compression VWAP reversion variants, and each one stayed negative on train, validation, and locked OOS.
- Discard the family without spending another heavy-run slot, then move the next manual follow-up lane to lag_convergence 4h.
