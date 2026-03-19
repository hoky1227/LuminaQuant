# anchored four-sleeve portfolio search

- objective: `val_sharpe + (12 * val_return) - (4 * val_max_drawdown) - (0.75 * hhi) - (0.15 * turnover)`
- runs: `384`
- best_params: `{"correlation_threshold": 0.35, "cost_penalty": 0.35, "max_family_cap": 0.45, "max_strategy_cap": 0.3, "target_vol": 0.06}`
- best_metric: `[4.19522905140387, 4.09846624658193, 0.02786612859065074, 0.010311986796541328, 0.26184372143960477]`
- bundle_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_topcap_rotation_relative_momentum_1h_anchor_latest.json`
