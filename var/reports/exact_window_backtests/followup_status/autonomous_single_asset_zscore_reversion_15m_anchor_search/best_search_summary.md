# anchored four-sleeve portfolio search

- objective: `val_sharpe + (12 * val_return) - (4 * val_max_drawdown) - (0.75 * hhi) - (0.15 * turnover)`
- runs: `384`
- best_params: `{"correlation_threshold": 0.35, "cost_penalty": 0.0, "max_family_cap": 0.45, "max_strategy_cap": 0.3, "target_vol": 0.06}`
- best_metric: `[2.7259315615446407, 2.8271856934058976, 0.011494829604180135, 0.008301231359634231, 0.27464954889717547]`
- bundle_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_single_asset_zscore_reversion_15m_anchor_latest.json`
