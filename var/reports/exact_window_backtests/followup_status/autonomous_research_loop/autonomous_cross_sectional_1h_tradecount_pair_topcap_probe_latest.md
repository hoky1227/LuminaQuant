# Autonomous 1h tradecount pair + topcap probe

- probe artifact: `var/reports/exact_window_backtests/followup_status/autonomous_cross_sectional_1h_tradecount_pair_topcap_opt/portfolio_optimization_latest.json`
- sleeves: `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` + `topcap_tsmom_1h_balanced_16_4_0.015`
- train_total_return: `-3.3069%`
- train_sharpe: `-0.216`
- val_total_return: `1.7590%`
- val_sharpe: `2.239`
- oos_total_return: `5.2751%` vs incumbent `4.7899%`
- oos_sharpe: `2.643` vs incumbent `1.707`
- oos_max_drawdown: `3.3337%` vs incumbent `5.8704%`
- oos_promotion_score: `8.0665` vs incumbent `4.3065` (delta `3.7599`)
- note: this probe materially improves locked-OOS metrics, but it still needs a formal decision artifact/wiring if we want it to enter the canonical portfolio decision ledger.
