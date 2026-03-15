# portfolio exact-window freeze manifest

- generated_at: `2026-03-14T06:37:57.569769+00:00`
- selection_basis: `validation_only`
- oos_start: `2026-02-01T00:00:00Z`

## sources
- composite_trend_30m: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/expansion_crypto_15m_30m_1h_20260310T115853Z/15m-30m-1h/exact_window_candidate_details_20260310T120626Z.json`
- regime_breakout_1h: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/topcap_crypto_1h_focus_20260310T123813Z/1h/exact_window_candidate_details_20260310T124047Z.json`
- rolling_breakout_30m: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/expansion_crypto_15m_30m_1h_20260310T115853Z/15m-30m-1h/exact_window_candidate_details_20260310T120626Z.json`
- topcap_tsmom_1h: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/topcap_crypto_1h_focus_20260310T123813Z/1h/exact_window_candidate_details_20260310T124047Z.json`

## baselines
- equal_weight: `summary_only_missing_component_streams`
- one_shot_optimized: `summary_only_missing_streams`

## exclusions
- PairSpread 4h excluded: `True` | review_not_before=`2026-03-31`
