# Autonomous single-asset zscore 15m challenger

- generated_at: `2026-03-17T11:27:52.357344+00:00`
- status: `discard`
- representative_variant: `mean_reversion_std_15m_balanced_ls_64_2.00`
- peak_rss_mib: `2981.020`

## locked-OOS metrics
- challenger_return: `2.7005%` | delta_vs_incumbent=`-3.0623%`
- challenger_sharpe: `2.435` | delta_vs_incumbent=`-1.071`
- challenger_max_drawdown: `1.6926%` | delta_vs_incumbent=`0.2649%`

## probe metrics
- probe_return: `-11.8992%`
- probe_sharpe: `-2.510`
- probe_max_drawdown: `18.8129%`

## decision
- Reused the March 15, 2026 15m intraday exact-window evidence instead of rerunning a duplicate heavy lane. The validation-ranked single-asset probe (mean_reversion_std_15m_balanced_ls_64_2.00) was already deeply negative on both validation (-11.9092%, Sharpe -4.208) and locked OOS (-11.8992%, Sharpe -2.510), and the anchored challenger still trailed the incumbent on return, Sharpe, and max drawdown.
- next_action: advance to liquidity-shock-reversion 5m next; the 15m single-asset reversion probe failed both validation and locked-OOS quality
