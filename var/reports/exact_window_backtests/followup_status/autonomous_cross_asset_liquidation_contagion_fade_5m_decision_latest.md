# Autonomous cross-asset liquidation contagion fade 5m challenger

- generated_at: `2026-03-17T13:44:21.328121+00:00`
- status: `discard`
- representative_variant: `liquidation_contagion_fade_5m_balanced_ls_64_1.2`
- peak_rss_mib: `2831.934`

## locked-OOS metrics
- challenger_return: `4.0743%` | delta_vs_incumbent=`-1.6885%`
- challenger_sharpe: `2.683` | delta_vs_incumbent=`-0.823`
- challenger_max_drawdown: `2.2423%` | delta_vs_incumbent=`0.8145%`

## probe metrics
- probe_return: `0.0000%`
- probe_sharpe: `0.000`
- probe_max_drawdown: `0.0000%`

## decision
- Round-2 item 4 failed at the probe stage: both contagion variants showed zero participation on validation and locked OOS, so the anchored challenger only diluted the incumbent and remained below it on the locked-OOS comparison.
- next_action: continue to the final round-2 item; this contagion family needs a stronger event definition
