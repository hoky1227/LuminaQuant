# portfolio one-shot incumbent bundle

- generated_at: `2026-03-15T13:32:34.462562+00:00`
- selection_basis: `incumbent_saved_one_shot_weights`
- current_bundle: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_bundle_latest.json`
- current_portfolio: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- oos_start: `2026-02-01T00:00:00Z`

## incumbent sleeves
- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | weight=60.00% | val_return=1.6067% | val_sharpe=3.880
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=25.44% | val_return=4.9879% | val_sharpe=4.428
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=14.56% | val_return=1.8372% | val_sharpe=1.641

## incumbent portfolio oos
- total_return: `5.7628%`
- sharpe: `3.506`
- sortino: `12.200`
- calmar: `55.594`
- max_drawdown: `1.4277%`
- volatility: `17.0745%`

## notes
- Bundle is restricted to the currently saved one-shot incumbent sleeves only.
- The incumbent bundle preserves saved weight ordering so challenger lanes can remain incumbent-first.
- Locked OOS starts at 2026-02-01T00:00:00Z and is excluded from tuning decisions.
