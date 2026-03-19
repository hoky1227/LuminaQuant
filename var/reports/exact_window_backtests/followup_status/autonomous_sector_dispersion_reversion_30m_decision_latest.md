# Autonomous sector-dispersion-reversion 30m decision

- generated_at: `2026-03-17T11:04:50.316725+00:00`
- status: `discard`
- representative_variant: `pair_spread_30m_sector_btcusdt_trxusdt_2.0_0.50`
- peak_rss_mib: `3541.164`

## locked-OOS metrics
- challenger_return: `1.8952%` | delta_vs_incumbent=`-3.8676%`
- challenger_sharpe: `1.565` | delta_vs_incumbent=`-1.941`
- challenger_max_drawdown: `2.6446%` | delta_vs_incumbent=`1.2168%`

## probe metrics
- probe_return: `-5.2645%`
- probe_sharpe: `-4.848`
- probe_max_drawdown: `9.1448%`

## decision
- Sector-dispersion 30m added eight bounded crypto-pair candidates, but the best validation-ranked probe (pair_spread_30m_sector_btcusdt_trxusdt_2.0_0.50) still flipped to negative locked-OOS performance (return -5.2645%, Sharpe -4.848). The anchored four-sleeve challenger then materially trailed the incumbent on locked-OOS return, Sharpe, and max drawdown.
- next_action: advance to single-asset-zscore-reversion 15m next; the 30m mean-reversion rows piggybacked on this run and were already negative on validation and locked OOS
