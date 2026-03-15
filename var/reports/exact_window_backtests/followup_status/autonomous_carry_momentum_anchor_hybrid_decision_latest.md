# autonomous carry+momentum anchor decision

- generated_at: `2026-03-15T05:30:10.149234+00:00`
- status: `discard`
- decision_reason: locked-OOS return, Sharpe, and max-drawdown all worsened versus the incumbent despite a small validation-return lift
- recommended_action: `retain incumbent; queue residual/crash-aware follow-up lanes`

## locked-OOS deltas vs incumbent

- oos_total_return_delta: -1.5158%
- oos_sharpe_delta: -0.760
- oos_max_drawdown_delta: 0.8145%
- val_total_return_delta: 0.0310%
- val_sharpe_delta: -1.568

## best challenger weights

- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | weight=30.00% | oos_ret=2.9547% | oos_sh=2.295
- `topcap_tsmom_1h_balanced_16_4_0.015` | weight=30.00% | oos_ret=3.2379% | oos_sh=1.464
- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | weight=30.00% | oos_ret=7.2375% | oos_sh=4.557
- `perp_crowding_carry_30m_0.25_0.08` | weight=10.00% | oos_ret=0.0574% | oos_sh=3.438

## sources

- `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686988`
- `https://www.nber.org/papers/w10480`
