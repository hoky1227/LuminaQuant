# Portfolio Comparison

- generated_at: `2026-03-14T06:38:42.313085+00:00`
- comparison_scope: `equal-weight diagnostic` vs `current one-shot optimized` vs `exact-window frozen tuned`
- freeze_artifact: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_exact_window_freeze_latest.json`
- tuned_portfolio: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_opt_tuned/portfolio_optimization_latest.json`
- current_one_shot_portfolio: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- equal_weight_normalization: `component rows lacked reusable return_streams, so comparison falls back to trusted summary metrics only`

## Validation metrics

- equal-weight diagnostic: return=6.4830% | sharpe=0.634 | sortino=0.816 | calmar=0.471 | max_dd=3.2914% | turnover=0.0371 | deflated_sharpe=0.0812 | pbo=0.625
- current one-shot optimized: return=5.0180% | sharpe=3.003 | sortino=6.198 | calmar=26.466 | max_dd=2.9463% | turnover=0.0000 | deflated_sharpe=0.0000 | pbo=0.000
- exact-window frozen tuned: return=6.6133% | sharpe=0.826 | sortino=1.164 | calmar=0.657 | max_dd=2.4107% | turnover=0.0000 | deflated_sharpe=0.0000 | pbo=0.000

## OOS metrics

- equal-weight diagnostic: return=5.8773% | sharpe=0.346 | sortino=0.487 | calmar=0.183 | max_dd=6.9219% | turnover=0.0381 | deflated_sharpe=0.0325 | pbo=0.500
- current one-shot optimized: return=4.7899% | sharpe=1.707 | sortino=4.341 | calmar=10.714 | max_dd=5.8704% | turnover=0.0000 | deflated_sharpe=0.0000 | pbo=0.000
- exact-window frozen tuned: return=2.6260% | sharpe=0.210 | sortino=0.250 | calmar=0.085 | max_dd=6.7676% | turnover=0.0000 | deflated_sharpe=0.0000 | pbo=0.000

## Frozen candidates

- `composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h
- `rolling_breakout_30m_guarded_ls_64_0.002` | strategy=RollingBreakoutStrategy | tf=30m
- `topcap_tsmom_1h_defensive_24_6_0.020` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h

## Deltas

- tuned vs equal-weight OOS return: -3.2513%
- tuned vs equal-weight OOS sharpe: -0.136
- tuned vs current one-shot OOS return: -2.1640%
- tuned vs current one-shot OOS sharpe: -1.497

## Notes

- legacy one-shot reference remains at `/home/hoky/Quants-agent/LuminaQuant/reports/portfolio_optimization_latest.json` but the current 3-way comparison uses a refreshed one-shot portfolio built from the current sleeve universe for apples-to-apples comparison.
