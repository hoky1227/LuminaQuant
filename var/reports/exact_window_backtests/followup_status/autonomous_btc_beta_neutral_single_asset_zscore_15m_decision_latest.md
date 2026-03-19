# Autonomous BTC-beta-neutral single-asset zscore 15m decision

- generated_at: `2026-03-17T11:46:38.054215+00:00`
- status: `discard`
- representative_variant: `mean_reversion_std_15m_resid_btc_guarded_lo_96_2.40`
- peak_rss_mib: `3136.551`

## locked-OOS metrics
- challenger_return: `3.1744%` | delta_vs_incumbent=`-2.5884%`
- challenger_sharpe: `2.414` | delta_vs_incumbent=`-1.092`
- challenger_max_drawdown: `1.6215%` | delta_vs_incumbent=`0.1938%`

## probe metrics
- probe_return: `-3.9982%`
- probe_sharpe: `-2.604`
- probe_max_drawdown: `5.6013%`

## decision
- Fresh 15m exact-window evidence for BTC-beta-neutral single-asset zscore reversion still failed quickly: the best residualized probe (mean_reversion_std_15m_resid_btc_guarded_lo_96_2.40) was negative on validation (-6.8939%, Sharpe -6.202) and locked OOS (-3.9982%, Sharpe -2.604). The anchored challenger also trailed the incumbent on locked-OOS return, Sharpe, and max drawdown.
- next_action: move to shortlist item 2: session-transition liquidity vacuum fade 5m
