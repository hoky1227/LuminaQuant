# anchored four-sleeve portfolio search

- objective: `val_sharpe + (12 * val_return) - (4 * val_max_drawdown) - (0.75 * hhi) - (0.15 * turnover)`
- runs: `384`
- best_params: `{"correlation_threshold": 0.35, "cost_penalty": 0.0, "max_family_cap": 0.45, "max_strategy_cap": 0.3, "target_vol": 0.06}`
- best_metric: `[3.7841872681662254, 3.72976013306876, 0.0254805616159397, 0.010334901073452807, 0.27999999999999997]`
- bundle_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_cross_asset_liquidation_contagion_fade_5m_anchor_latest.json`
