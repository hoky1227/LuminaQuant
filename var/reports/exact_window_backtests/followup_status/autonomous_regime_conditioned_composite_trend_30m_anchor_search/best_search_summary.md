# anchored four-sleeve portfolio search

- objective: `val_sharpe + (12 * val_return) - (4 * val_max_drawdown) - (0.75 * hhi) - (0.15 * turnover)`
- runs: `384`
- best_params: `{"correlation_threshold": 0.35, "cost_penalty": 0.35, "max_family_cap": 0.55, "max_strategy_cap": 0.3, "target_vol": 0.06}`
- best_metric: `[5.62937367803804, 5.397871148982835, 0.038842964761541365, 0.009135503850945303, 0.2640947102393469]`
- bundle_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_regime_conditioned_composite_trend_30m_anchor_latest.json`
