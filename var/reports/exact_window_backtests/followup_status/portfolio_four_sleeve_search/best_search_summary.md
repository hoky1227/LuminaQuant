# anchored four-sleeve portfolio search

- objective: `val_sharpe + (12 * val_return) - (4 * val_max_drawdown) - (0.75 * hhi) - (0.15 * turnover)`
- runs: `384`
- best_params: `{"correlation_threshold": 0.35, "cost_penalty": 0.0, "max_family_cap": 0.75, "max_strategy_cap": 0.25, "target_vol": 0.06}`
- best_metric: `[0.7809740640390175, 0.5130599889377316, 0.048555188197246135, 0.031812045816416945, 0.25]`
- bundle_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_four_sleeve_anchored_bundle_latest.json`
