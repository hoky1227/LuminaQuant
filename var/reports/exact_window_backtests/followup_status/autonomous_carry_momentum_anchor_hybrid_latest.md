# autonomous carry+momentum anchor hybrid

- generated_at: `2026-03-15T05:27:08.250696+00:00`
- selection_basis: `autonomous_web_grounded_carry_momentum_anchor_hybrid`
- candidate_count: `4`
- rationale: anchored carry+momentum hybrid probe using incumbent pair+composite+topcap plus the best 30m perp carry sleeve.
- sources:
  - `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686988`
  - `https://www.nber.org/papers/w10480`

## candidates
- `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | PairSpreadZScoreStrategy | tf=1h | val_ret=1.6067% | oos_ret=7.2375%
- `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | CompositeTrendStrategy | tf=30m | val_ret=4.9879% | oos_ret=2.9547%
- `topcap_tsmom_1h_balanced_16_4_0.015` | TopCapTimeSeriesMomentumStrategy | tf=1h | val_ret=1.8372% | oos_ret=3.2379%
- `perp_crowding_carry_30m_0.25_0.08` | PerpCrowdingCarryStrategy | tf=30m | val_ret=0.0000% | oos_ret=0.0574%
