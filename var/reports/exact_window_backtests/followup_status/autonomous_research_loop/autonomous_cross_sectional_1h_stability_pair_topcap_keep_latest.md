# Autonomous keep candidate — 1h stability pair + topcap

- basis: validation-only train-stability filtered subset from autonomous_cross_sectional_1h_20260314T150019Z
- sleeves: `pair_spread_1h_core_ethusdt_solusdt_1.8_0.45` + `topcap_tsmom_1h_balanced_16_4_0.015`
- train_total_return: `-0.0817%`
- train_sharpe: `0.068`
- val_total_return: `2.4956%`
- val_sharpe: `3.039`
- oos_total_return: `2.5471%`
- oos_sharpe: `1.873`
- note: promotion-score delta vs incumbent is positive, but this remains a research keep pending stricter robustness checks (pair trade count / pbo / stability follow-up).