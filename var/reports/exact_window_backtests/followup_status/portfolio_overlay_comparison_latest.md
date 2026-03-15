# Overlay Portfolio Comparison

- overlay_vs_current_one_shot_oos_return: -0.2576%
- overlay_vs_current_one_shot_oos_sharpe: 0.003

## Overlay OOS metrics

{"cagr": 0.7200821058511753, "calmar": 55.98898268422918, "max_drawdown": 0.01286113930507271, "sharpe": 3.4496548450543796, "sortino": 12.164513377393309, "total_return": 0.05338446074194625, "volatility": 0.16091888831080745}

## Overlay final allocation

- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | tf=1h | weight=56.46%
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | strategy=CompositeTrendStrategy | tf=30m | weight=25.44%
- `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | tf=1h | weight=14.56%
