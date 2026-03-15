# Autonomous 1h tradecount pair + topcap probe

- selection_basis: `autonomous_tradecount_screen_pair_topcap`
- candidates:
  - `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | strategy=PairSpreadZScoreStrategy | train_return=3.3331% | val_return=1.6067% | oos_return=7.2375% | oos_trade_count=11.0
  - `topcap_tsmom_1h_balanced_16_4_0.015` | strategy=TopCapTimeSeriesMomentumStrategy | train_return=-10.7790% | val_return=1.8372% | oos_return=3.2379% | oos_trade_count=119.0
