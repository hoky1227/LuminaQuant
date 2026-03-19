# Autonomous funding/liquidation crowding fade 30m decision

- generated_at: `2026-03-17T13:03:14.984466+00:00`
- status: `discard`
- representative_variant: `funding_liquidation_crowding_fade_30m_balanced_ls_96_0.85`
- peak_rss_mib: `3594.805`

## locked-OOS metrics
- challenger_return: `4.0743%` | delta_vs_incumbent=`-1.6885%`
- challenger_sharpe: `2.683` | delta_vs_incumbent=`-0.823`
- challenger_max_drawdown: `2.2423%` | delta_vs_incumbent=`0.8145%`

## probe metrics
- probe_return: `0.0000%`
- probe_sharpe: `0.000`
- probe_max_drawdown: `0.0000%`

## decision
- Round-2 item 1 failed at the probe stage: both funding/liquidation crowding fade variants produced zero participation on validation and locked OOS. The anchored challenger therefore only diluted the incumbent and remained below it on locked-OOS return, Sharpe, and drawdown.
- next_action: if continuing round-2 immediately, move to Basis snapback reversion 30m
