# Dynamic Portfolio Comparison

- dynamic_vs_current_one_shot_oos_return: -3.5442%
- dynamic_vs_current_one_shot_oos_sharpe: -0.530

## Dynamic OOS metrics

{"cagr": 0.13781667344586013, "calmar": 5.985883861822879, "max_drawdown": 0.023023612991363795, "sharpe": 1.1764008956920013, "sortino": 3.130311250245071, "total_return": 0.012457484650929773, "volatility": 0.11517336627567139}

## Dynamic final allocation

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=50.00%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=11.54%
- `regime_breakout_1h_trend_ls_48_0.70` | strategy=RegimeBreakoutCandidateStrategy | tf=1h | weight=2.58%
