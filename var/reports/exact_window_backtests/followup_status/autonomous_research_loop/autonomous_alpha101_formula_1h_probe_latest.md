# autonomous alpha101 formula 1h probe

- generated_at: `2026-03-16T11:08:47.644190+00:00`
- lane: reused alpha101_formula 1h exact-window evidence
- status: `discard`
- reused_supporting_runs: `8`
- best_variant: `alpha101_formula_1h_a005_a005_vwap_tuned_dir`
- best_variant_val_return: -3.8665%
- best_variant_oos_return: -15.9797%
- incumbent_oos_return: 5.7628%

## takeaway

- The 1h Alpha101 family was already screened repeatedly inside prior exact-window batches, and every reused run showed the same negative train/validation/OOS profile.
- Discard the family without spending another heavy-run slot, then move the next manual follow-up lane to volcomp_vwap_rev_guarded 5m.
