# Overlay Portfolio Comparison

- overlay_vs_current_one_shot_oos_return: -2.5293%
- overlay_vs_current_one_shot_oos_sharpe: -0.753

## Overlay OOS metrics

{"cagr": 0.26254137673823785, "calmar": 4.590543652738419, "max_drawdown": 0.05719178306508921, "sharpe": 0.9540071089926616, "sortino": 2.221828079263361, "total_return": 0.022606347417474693, "volatility": 0.28440295396074344}

## Overlay final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=35.00%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=35.00%
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | weight=23.53%
