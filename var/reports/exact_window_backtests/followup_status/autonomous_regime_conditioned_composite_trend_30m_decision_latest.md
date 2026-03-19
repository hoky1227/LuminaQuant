# Autonomous regime-conditioned composite trend 30m challenger

- generated_at: `2026-03-17T11:27:52.772211+00:00`
- status: `discard`
- representative_variant: `composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80`
- peak_rss_mib: `3219.531`

## locked-OOS metrics
- challenger_return: `4.0998%` | delta_vs_incumbent=`-1.6630%`
- challenger_sharpe: `2.750` | delta_vs_incumbent=`-0.756`
- challenger_max_drawdown: `1.5384%` | delta_vs_incumbent=`0.1107%`

## probe metrics
- probe_return: `2.2110%`
- probe_sharpe: `1.625`
- probe_max_drawdown: `3.4079%`

## decision
- Reused the March 15, 2026 composite-trend 30m exact-window evidence. The best validation-ranked composite variant (composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80) remained positive, but the anchored challenger still underperformed the incumbent on locked-OOS return, Sharpe, and drawdown, so the family did not justify replacing or diluting the existing trend sleeve.
- next_action: advance to topcap-rotation-relative-momentum 1h next; the 30m composite variant remained too close to the incumbent trend sleeve to overcome the locked-OOS gap
