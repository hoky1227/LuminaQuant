# portfolio max-performance decision

- generated_at: `2026-03-15T04:15:50.303738+00:00`
- selection_basis: `locked_oos_promotion_score`
- winner: `Autonomous triplet pair+topcap+composite challenger` (promoted_challenger)
- winner_reason: Promotable because promotion score and OOS total return both improved over the incumbent.
- oos_start: `2026-02-01T00:00:00Z`

## candidate summary
- `Current one-shot incumbent` | score=8.0665 | delta=0.0000 | oos_return=5.2751% | oos_sharpe=2.643 | oos_max_dd=3.3337% | promotable=False
  - Current one-shot incumbent baseline.
- `Exact-window frozen tuned challenger` | score=0.2757 | delta=-7.7908 | oos_return=2.6260% | oos_sharpe=0.210 | oos_max_dd=6.7676% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve; OOS max drawdown did not improve
- `Backbone-preserving triplet search challenger` | score=0.6768 | delta=-7.3896 | oos_return=0.9079% | oos_sharpe=0.503 | oos_max_dd=5.8576% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve; OOS max drawdown did not improve
- `Causal dynamic challenger` | score=10.2436 | delta=2.1771 | oos_return=3.6451% | oos_sharpe=2.945 | oos_max_dd=1.4214% | promotable=False
  - OOS total return did not improve
- `Causal overlay challenger` | score=7.4696 | delta=-0.5968 | oos_return=4.5271% | oos_sharpe=2.467 | oos_max_dd=3.0351% | promotable=False
  - promotion score improvement did not clear the base threshold; OOS total return did not improve
- `Autonomous triplet pair+topcap+composite challenger` | score=13.4725 | delta=5.4060 | oos_return=5.5960% | oos_sharpe=3.447 | oos_max_dd=1.4277% | promotable=True
  - Promotable because promotion score and OOS total return both improved over the incumbent.

## notes
- Promotion uses locked-OOS metrics only after each challenger artifact is frozen.
- Locked OOS starts at 2026-02-01T00:00:00Z and remains excluded from tuning decisions.
- If thresholds are not clearly exceeded, incumbent tie policy keeps the current one-shot leader.
