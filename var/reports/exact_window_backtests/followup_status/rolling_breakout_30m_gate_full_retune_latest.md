# rolling breakout 30m regime gate

- generated_at: `2026-03-14T13:26:43.450287+00:00`
- candidate: `rolling_breakout_30m_guarded_ls_64_0.0015`
- selection_basis: `train_val_only`
- selection_uses_oos: `False`
- selected_rule: `btc_above_ma192_and_breadth_ma192_ge_60`
- label: BTC above 4-day trend + 60% basket breadth above 4-day trend
- conditions: `btc_above_ma192, breadth_ma192_ge_60`
- signal_lag_days: `1`
- survives: `True`
- survives_train_val: `True`
- recommended_action: `activate_conditionally`
- gated_oos_return: `-6.0503%`
- gated_oos_sharpe: `-2.565`
- gated_oos_sortino: `-2.122`
- gated_oos_calmar: `-4.637`
- gated_oos_max_drawdown: `10.4411%`
- gated_oos_trade_count: `42`
- gated_oos_pbo: `0.000`

## evaluated rules
- `basket_vol_ratio_moderate`: return=7.0690% | sharpe=1.671 | pbo=0.625 | activation=68.57% | survives=False
- `btc_above_ma192`: return=-3.0611% | sharpe=-1.098 | pbo=0.125 | activation=45.71% | survives=False
- `btc_above_ma192_and_breadth_ma96_ge_60`: return=-6.6517% | sharpe=-3.167 | pbo=0.000 | activation=34.29% | survives=True
- `btc_above_ma192_and_breadth_ma192_ge_60`: return=-6.0503% | sharpe=-2.565 | pbo=0.000 | activation=42.86% | survives=True
- `btc_above_ma192_and_breadth_ma96_ge_60_and_vol_expansion`: return=-4.3433% | sharpe=-2.758 | pbo=0.000 | activation=17.14% | survives=False
- `btc_above_ma336_and_breadth_ma192_ge_60`: return=-5.2592% | sharpe=-2.327 | pbo=0.000 | activation=31.43% | survives=False
- `btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos`: return=-4.0008% | sharpe=-2.109 | pbo=0.000 | activation=25.71% | survives=True
