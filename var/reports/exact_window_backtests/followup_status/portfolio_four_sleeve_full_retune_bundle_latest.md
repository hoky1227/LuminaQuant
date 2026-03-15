# portfolio exact-window freeze

- generated_at: `2026-03-14T13:26:43.737171+00:00`
- selection_basis: `incumbent_anchor_rolling_gate`
- frozen_count: `4`
- rolling_admission_blocked: `False`
- pair_survives: `False`
- equal_weight_baseline: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/committee_portfolio_followup_latest.json`
- one_shot_baseline: `/home/hoky/Quants-agent/LuminaQuant/reports/portfolio_optimization_latest.json`

## frozen sleeves
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | val_sharpe=4.428 | val_return=4.9879% | val_pbo=0.250
- `regime_breakout_1h_tightest_ls_72_0.80` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | val_sharpe=5.293 | val_return=12.5536% | val_pbo=0.125
- `rolling_breakout_30m_guarded_ls_64_0.0015` | strategy=RollingBreakoutStrategy | tf=30m | val_sharpe=2.904 | val_return=4.5729% | val_pbo=0.500
- `topcap_tsmom_1h_persistence_24_6_020_24_6_0.020` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | val_sharpe=3.004 | val_return=2.7703% | val_pbo=0.375

## notes
- Incumbent-aware mode seeds the existing one-shot incumbent sleeves and only promotes local challengers when the explicit anchor thresholds are met.
- RollingBreakout admission is conditional on the rebuilt train+val-only gate; locked OOS is excluded from sleeve admission.
- PairSpread 4h remains excluded unless the follow-up guard explicitly survives.
- Equal-weight baseline default handling is rebuild/normalize from component streams before comparison.
- RollingBreakout cleared the train+val-only gate and is admitted as the conditional fourth sleeve.
- One-shot optimized baseline defaults to reports/portfolio_optimization_latest.json.
