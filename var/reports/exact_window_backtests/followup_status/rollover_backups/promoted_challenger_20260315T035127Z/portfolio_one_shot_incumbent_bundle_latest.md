# portfolio one-shot incumbent bundle

- generated_at: `2026-03-14T10:53:25.614057+00:00`
- selection_basis: `incumbent_saved_one_shot_weights`
- current_bundle: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- current_portfolio: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- oos_start: `2026-02-01T00:00:00Z`

## incumbent sleeves
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=35.00% | val_return=4.9879% | val_sharpe=4.428
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=35.00% | val_return=1.8372% | val_sharpe=1.641
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | weight=30.00% | val_return=8.6067% | val_sharpe=3.539

## incumbent portfolio oos
- total_return: `4.7899%`
- sharpe: `1.707`
- sortino: `4.341`
- calmar: `10.714`
- max_drawdown: `5.8704%`
- volatility: `31.3226%`

## notes
- Bundle is restricted to the currently saved one-shot incumbent sleeves only.
- The incumbent bundle preserves saved weight ordering so challenger lanes can remain incumbent-first.
- Locked OOS starts at 2026-02-01T00:00:00Z and is excluded from tuning decisions.
