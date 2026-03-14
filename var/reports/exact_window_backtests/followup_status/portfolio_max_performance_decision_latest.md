# portfolio max-performance decision

- generated_at: `2026-03-14T10:57:12.860075+00:00`
- selection_basis: `locked_oos_promotion_score`
- winner: `Current one-shot incumbent` (retained_incumbent)
- winner_reason: No challenger cleared the locked-OOS promotion rule; keep the current one-shot incumbent.
- oos_start: `2026-02-01T00:00:00Z`

## candidate summary
- `Current one-shot incumbent` | score=4.3065 | delta=0.0000 | oos_return=4.7899% | oos_sharpe=1.707 | oos_max_dd=5.8704% | promotable=False
  - Current one-shot incumbent baseline.
- `Exact-window frozen tuned challenger` | score=0.2757 | delta=-4.0308 | oos_return=2.6260% | oos_sharpe=0.210 | oos_max_dd=6.7676% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve; OOS max drawdown did not improve
- `Backbone-preserving triplet search challenger` | score=0.6768 | delta=-3.6297 | oos_return=0.9079% | oos_sharpe=0.503 | oos_max_dd=5.8576% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve
- `Causal dynamic challenger` | score=2.8167 | delta=-1.4898 | oos_return=1.2457% | oos_sharpe=1.176 | oos_max_dd=2.3024% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve
- `Causal overlay challenger` | score=1.9747 | delta=-2.3318 | oos_return=2.2606% | oos_sharpe=0.954 | oos_max_dd=5.7192% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve

## notes
- Promotion uses locked-OOS metrics only after each challenger artifact is frozen.
- Locked OOS starts at 2026-02-01T00:00:00Z and remains excluded from tuning decisions.
- If thresholds are not clearly exceeded, incumbent tie policy keeps the current one-shot leader.
