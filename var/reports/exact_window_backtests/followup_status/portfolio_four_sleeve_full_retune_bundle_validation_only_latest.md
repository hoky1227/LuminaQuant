# portfolio exact-window freeze

- generated_at: `2026-03-14T13:29:42.728494+00:00`
- selection_basis: `validation_only`
- frozen_count: `4`
- rolling_admission_blocked: `False`
- pair_survives: `False`
- equal_weight_baseline: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/committee_portfolio_followup_latest.json`
- one_shot_baseline: `/home/hoky/Quants-agent/LuminaQuant/reports/portfolio_optimization_latest.json`

## frozen sleeves
- `composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | val_sharpe=5.033 | val_return=6.2949% | val_pbo=0.375
- `regime_breakout_1h_tightest_ls_72_0.80` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | val_sharpe=5.293 | val_return=12.5536% | val_pbo=0.125
- `rolling_breakout_30m_guarded_ls_64_0.0015` | strategy=RollingBreakoutStrategy | tf=30m | val_sharpe=2.904 | val_return=4.5729% | val_pbo=0.500
- `topcap_tsmom_1h_slow_rebalance_16_6_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | val_sharpe=3.148 | val_return=2.9241% | val_pbo=0.500

## notes
- Exact-window sleeve freeze uses the deterministic train/val-only formula; OOS-derived helper fields are excluded from selection.
- RollingBreakout is frozen from exact-window candidate-detail rows first, then optionally supplemented by the gate artifact for stream/metadata only.
- PairSpread 4h is excluded unless the follow-up guard explicitly survives.
- Equal-weight baseline default handling is rebuild/normalize from component streams before 3-way comparison.
- One-shot optimized baseline defaults to reports/portfolio_optimization_latest.json.
