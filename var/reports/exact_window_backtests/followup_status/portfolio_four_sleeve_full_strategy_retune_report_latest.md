# portfolio four-sleeve full strategy retune report

- generated_at: `2026-03-14T13:32:30Z`
- peak_rss_gib: `2.9854`
- memory_budget_gib: `8.0`

## best per strategy
- `composite_trend_30m_retune_core_relaxed_ls_0.60_0.40_0.15_0.75` | strategy=CompositeTrendStrategy | tf=30m | val_sharpe=4.445 | val_return=6.0753% | oos_sharpe=2.586 | oos_return=4.3192%
- `regime_breakout_1h_tight_vol_stop_ls_48_0.78` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | val_sharpe=4.658 | val_return=10.5418% | oos_sharpe=1.391 | oos_return=4.7614%
- `rolling_breakout_30m_guarded_ls_64_0.0015` | strategy=RollingBreakoutStrategy | tf=30m | val_sharpe=5.889 | val_return=16.5308% | oos_sharpe=0.943 | oos_return=3.5608%
- `topcap_tsmom_1h_slow_rebalance_16_6_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | val_sharpe=3.148 | val_return=2.9241% | oos_sharpe=1.292 | oos_return=2.2000%

## portfolio results
- anchored_full_retune_oos: return=-3.9066% | sharpe=-0.269 | max_dd=8.6974%
- validation_only_full_retune_oos: return=-2.0589% | sharpe=-0.129 | max_dd=7.5370%

## conclusion
- neither full strategy-retune branch beat the current incumbent on locked OOS.
- retain the incumbent production portfolio.
