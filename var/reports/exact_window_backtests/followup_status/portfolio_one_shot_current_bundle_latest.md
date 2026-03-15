# portfolio one-shot current bundle

- generated_at: `2026-03-15T04:16:23.342963+00:00`
- pair_survives: `True`
- selection_basis: `rolled_over_from_promoted_challenger`
- source_report: `autonomous_triplet_pair_topcap_composite_latest.json`
- source_portfolio: `autonomous_triplet_pair_topcap_composite_opt/portfolio_optimization_latest.json`

## sleeves
- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | val_sharpe=3.880
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | val_sharpe=1.641
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | val_sharpe=4.428
