# rolling breakout 30m regime gate

- generated_at: `2026-03-14T12:17:58.607084+00:00`
- candidate: `rolling_breakout_30m_guarded_ls_64_0.002`
- selection_basis: `train_val_only`
- selection_uses_oos: `False`
- selected_rule: `btc_above_ma192_and_breadth_ma192_ge_60`
- label: BTC above 4-day trend + 60% basket breadth above 4-day trend
- conditions: `btc_above_ma192, breadth_ma192_ge_60`
- signal_lag_days: `1`
- survives: `True`
- survives_train_val: `True`
- recommended_action: `activate_conditionally`
- gated_oos_return: `-5.4101%`
- gated_oos_sharpe: `-2.363`
- gated_oos_sortino: `-1.962`
- gated_oos_calmar: `-4.331`
- gated_oos_max_drawdown: `10.2893%`
- gated_oos_trade_count: `37`
- gated_oos_pbo: `0.000`

## evaluated rules
- `basket_vol_ratio_moderate`: return=7.2152% | sharpe=1.714 | pbo=0.500 | activation=68.57% | survives=False
- `btc_above_ma192`: return=-2.6921% | sharpe=-0.986 | pbo=0.125 | activation=45.71% | survives=False
- `btc_above_ma192_and_breadth_ma96_ge_60`: return=-5.7035% | sharpe=-2.765 | pbo=0.000 | activation=34.29% | survives=True
- `btc_above_ma192_and_breadth_ma192_ge_60`: return=-5.4101% | sharpe=-2.363 | pbo=0.000 | activation=42.86% | survives=True
- `btc_above_ma192_and_breadth_ma96_ge_60_and_vol_expansion`: return=-3.9081% | sharpe=-2.596 | pbo=0.000 | activation=17.14% | survives=False
- `btc_above_ma336_and_breadth_ma192_ge_60`: return=-4.9617% | sharpe=-2.286 | pbo=0.000 | activation=31.43% | survives=False
- `btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos`: return=-3.7440% | sharpe=-1.978 | pbo=0.000 | activation=25.71% | survives=False
