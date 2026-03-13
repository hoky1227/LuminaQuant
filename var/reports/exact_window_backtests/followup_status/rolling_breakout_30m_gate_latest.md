# rolling breakout 30m regime gate

- generated_at: `2026-03-13T10:12:56.480901+00:00`
- candidate: `rolling_breakout_30m_guarded_ls_64_0.002`
- selected_rule: `basket_vol_ratio_moderate`
- label: Moderate basket volatility regime
- conditions: `basket_vol_ratio_moderate`
- signal_lag_days: `1`
- survives: `True`
- recommended_action: `activate_conditionally`
- gated_oos_return: `7.2152%`
- gated_oos_sharpe: `1.714`
- gated_oos_sortino: `1.862`
- gated_oos_calmar: `6.679`
- gated_oos_max_drawdown: `16.3720%`
- gated_oos_trade_count: `45`
- gated_oos_pbo: `0.500`

## evaluated rules
- `basket_vol_ratio_moderate`: return=7.2152% | sharpe=1.714 | pbo=0.500 | activation=68.57% | survives=True
- `btc_above_ma192`: return=-2.6921% | sharpe=-0.986 | pbo=0.125 | activation=45.71% | survives=False
- `btc_above_ma192_and_breadth_ma96_ge_60`: return=-5.7035% | sharpe=-2.765 | pbo=0.000 | activation=34.29% | survives=False
- `btc_above_ma192_and_breadth_ma192_ge_60`: return=-5.4101% | sharpe=-2.363 | pbo=0.000 | activation=42.86% | survives=False
- `btc_above_ma192_and_breadth_ma96_ge_60_and_vol_expansion`: return=-3.9081% | sharpe=-2.596 | pbo=0.000 | activation=17.14% | survives=False
- `btc_above_ma336_and_breadth_ma192_ge_60`: return=-4.9617% | sharpe=-2.286 | pbo=0.000 | activation=31.43% | survives=False
- `btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos`: return=-3.7440% | sharpe=-1.978 | pbo=0.000 | activation=25.71% | survives=False
