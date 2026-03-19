# Conclusion B — New Hypothesis Refresh

- updated_at_utc: `2026-03-17T11:28:39.822818+00:00`
- status: `completed_no_promotion`
- incumbent: `portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- incumbent locked-OOS: return `5.7628%`, Sharpe `3.506`, max drawdown `1.4277%`

## Final queue results

- `sector-dispersion-reversion 30m` → `discard` | OOS return=1.8952% | Sharpe=1.565 | max DD=2.6446% | peak RSS=3541.164 MiB
- `single-asset-zscore-reversion 15m` → `discard` | OOS return=2.7005% | Sharpe=2.435 | max DD=1.6926% | peak RSS=2981.020 MiB
- `liquidity-shock-reversion 5m` → `discard` | OOS return=3.3317% | Sharpe=3.363 | max DD=0.9857% | peak RSS=2808.316 MiB
- `regime-conditioned-composite-trend 30m` → `discard` | OOS return=4.0998% | Sharpe=2.750 | max DD=1.5384% | peak RSS=3219.531 MiB
- `topcap-rotation-relative-momentum 1h` → `discard` | OOS return=2.3650% | Sharpe=1.588 | max DD=3.0478% | peak RSS=3096.969 MiB

## Final conclusion

- None of the five new-hypothesis lanes beat the incumbent on the locked-OOS decision path.
- The strongest partial result was the fresh `liquidity-shock-reversion 5m` lane, which improved drawdown but still missed the incumbent on return.
- The correct next move is to wait for the deferred 2026-03-31 XPT/XPD retune window or invent a genuinely new family instead of recycling the exhausted queue.
