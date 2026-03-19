# anchored four-sleeve portfolio search

- objective: `val_sharpe + (12 * val_return) - (4 * val_max_drawdown) - (0.75 * hhi) - (0.15 * turnover)`
- runs: `384`
- best_params: `{"correlation_threshold": 0.35, "cost_penalty": 0.35, "max_family_cap": 0.45, "max_strategy_cap": 0.3, "target_vol": 0.06}`
- best_metric: `[5.554642913152415, 5.491680313882841, 0.023342754441814106, 0.004428250167404557, 0.2659166044834364]`
- bundle_path: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_session_gated_residual_basket_reversion_5m_anchor_latest.json`
