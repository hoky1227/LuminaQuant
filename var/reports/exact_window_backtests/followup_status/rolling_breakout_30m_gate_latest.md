# rolling breakout 30m regime gate

- generated_at: `2026-03-13T09:39:06.446359+00:00`
- candidate: `rolling_breakout_30m_guarded_ls_64_0.002`
- selected_rule: `btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos`
- label: BTC above 4-day trend + breadth + positive 2-day basket momentum
- conditions: `btc_above_ma192, breadth_ma96_ge_60, basket_ret96_pos`
- gated_oos_return: `1.4907%`
- gated_oos_sharpe: `1.067`
- gated_oos_sortino: `0.667`
- gated_oos_calmar: `5.456`
- gated_oos_max_drawdown: `3.0581%`
- gated_oos_trade_count: `16`

## evaluated rules
- `btc_above_ma192`: return=-8.3226% | sharpe=-4.438 | activation=45.71%
- `btc_above_ma192_and_breadth_ma96_ge_60`: return=-2.5678% | sharpe=-1.458 | activation=34.29%
- `btc_above_ma192_and_breadth_ma192_ge_60`: return=-7.5940% | sharpe=-4.040 | activation=42.86%
- `btc_above_ma192_and_breadth_ma96_ge_60_and_vol_expansion`: return=-0.8990% | sharpe=-0.536 | activation=17.14%
- `btc_above_ma336_and_breadth_ma192_ge_60`: return=-5.9652% | sharpe=-3.207 | activation=31.43%
- `btc_above_ma192_and_breadth_ma96_ge_60_and_ret96_pos`: return=1.4907% | sharpe=1.067 | activation=25.71%
