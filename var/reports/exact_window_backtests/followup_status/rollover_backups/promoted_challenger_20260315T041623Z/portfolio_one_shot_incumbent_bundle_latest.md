# portfolio one-shot incumbent bundle

- generated_at: `2026-03-15T03:51:28.822024+00:00`
- selection_basis: `incumbent_saved_one_shot_weights`
- current_bundle: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- current_portfolio: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- oos_start: `2026-02-01T00:00:00Z`

## incumbent sleeves
- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | weight=50.00% | val_return=1.6067% | val_sharpe=3.880
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=50.00% | val_return=1.8372% | val_sharpe=1.641

## incumbent portfolio oos
- total_return: `5.2751%`
- sharpe: `2.643`
- sortino: `8.743`
- calmar: `21.277`
- max_drawdown: `3.3337%`
- volatility: `21.1029%`

## notes
- Bundle is restricted to the currently saved one-shot incumbent sleeves only.
- The incumbent bundle preserves saved weight ordering so challenger lanes can remain incumbent-first.
- Locked OOS starts at 2026-02-01T00:00:00Z and is excluded from tuning decisions.
