# Autonomous liquidity-shock 5m challenger

- generated_at: `2026-03-17T11:27:52.575803+00:00`
- status: `discard`
- representative_variant: `liquidity_shock_reversion_5m_thin_lo_72_0.010`
- peak_rss_mib: `2808.316`

## locked-OOS metrics
- challenger_return: `3.3317%` | delta_vs_incumbent=`-2.4311%`
- challenger_sharpe: `3.363` | delta_vs_incumbent=`-0.143`
- challenger_max_drawdown: `0.9857%` | delta_vs_incumbent=`-0.4421%`

## probe metrics
- probe_return: `-0.3806%`
- probe_sharpe: `-0.245`
- probe_max_drawdown: `4.3842%`

## decision
- Fresh 5m liquidity-shock candidates were evaluated on BTC/ETH/BNB; the best validation-ranked probe (liquidity_shock_reversion_5m_thin_lo_72_0.010) stayed negative on validation and locked OOS. The anchored challenger improved max drawdown to 0.9857% from the incumbent's 1.4277%, but it still lagged on locked-OOS total return (3.3317% vs 5.7628%).
- next_action: advance to regime-conditioned-composite-trend 30m next; the new liquidity sleeve improved drawdown but still missed the incumbent on locked-OOS return
