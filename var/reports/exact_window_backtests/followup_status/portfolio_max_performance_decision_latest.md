# portfolio max-performance decision

- generated_at: `2026-03-15T13:32:34.466835+00:00`
- selection_basis: `locked_oos_promotion_score`
- winner: `Current one-shot incumbent` (retained_incumbent)
- winner_reason: No challenger cleared the locked-OOS promotion rule; keep the current one-shot incumbent.
- oos_start: `2026-02-01T00:00:00Z`

## candidate summary
- `Current one-shot incumbent` | score=13.7266 | delta=0.0000 | oos_return=5.7628% | oos_sharpe=3.506 | oos_max_dd=1.4277% | promotable=False
  - Current one-shot incumbent baseline.
- `Exact-window frozen tuned challenger` | score=0.2757 | delta=-13.4509 | oos_return=2.6260% | oos_sharpe=0.210 | oos_max_dd=6.7676% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve; OOS max drawdown did not improve
- `Backbone-preserving triplet search challenger` | score=0.6768 | delta=-13.0497 | oos_return=0.9079% | oos_sharpe=0.503 | oos_max_dd=5.8576% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve; OOS max drawdown did not improve
- `Causal dynamic challenger` | score=9.3040 | delta=-4.4226 | oos_return=3.1796% | oos_sharpe=2.719 | oos_max_dd=1.1849% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve
- `Causal overlay challenger` | score=13.6678 | delta=-0.0587 | oos_return=5.3384% | oos_sharpe=3.450 | oos_max_dd=1.2861% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve
- `Autonomous vol-managed refined challenger` | score=13.7266 | delta=0.0000 | oos_return=5.7628% | oos_sharpe=3.506 | oos_max_dd=1.4277% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve; OOS max drawdown did not improve

## notes
- Promotion uses locked-OOS metrics only after each challenger artifact is frozen.
- Locked OOS starts at 2026-02-01T00:00:00Z and remains excluded from tuning decisions.
- If thresholds are not clearly exceeded, incumbent tie policy keeps the current one-shot leader.
