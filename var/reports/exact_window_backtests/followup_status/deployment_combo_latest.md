# deployment combo

- generated_at: `2026-03-10T13:05:22.769204+00:00`
- scenario_id: `experimental_research_watchlist`
- label: `Experimental research watchlist`
- selection_basis: `research_watchlist_equal_weight`
- oos_return: `3.1445%`
- oos_sharpe: `1.745`
- oos_max_drawdown: `3.4453%`
- oos_trade_count: `364`
- oos_pbo: `0.375`

## components
- research watchlist sleeve: `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | tf=30m | weight=50.00% | oos_return=2.9547% | oos_sharpe=2.295 | oos_pbo=0.375
- research watchlist sleeve: `topcap_tsmom_1h_balanced_16_4_0.015` | tf=1h | weight=50.00% | oos_return=3.2379% | oos_sharpe=1.464 | oos_pbo=0.375
