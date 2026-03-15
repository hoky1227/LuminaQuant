# autonomous fractional kelly blend 10pct probe

- generated_at: `2026-03-15T09:43:22.709382+00:00`
- lane: 10% Kelly blend into incumbent weights
- status: `discard`
- weights: {"composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80": 0.28895124937927275, "pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55": 0.5800000000000001, "topcap_tsmom_1h_balanced_16_4_0.015": 0.1310487506207273}
- train_return_delta: 0.4086%
- train_sharpe_delta: 0.052
- oos_return_delta: -0.0904%
- oos_sharpe_delta: 0.009
- oos_max_drawdown_delta: -0.0734%

## takeaway

- This was the best low-risk Kelly-style blend tested so far, but it still missed the locked-OOS return gate.
- Keep as a near-miss reference only.
