# article-inspired llm alpha research pipeline

- generated_at: `2026-03-15T14:27:31.845280+00:00`
- total_memory_cap_gib: `8.0`
- heavy_run_cap_gib: `6.5`
- heavy_run_parallelism: `1`

## thesis
- Use LLM-style research orchestration for hypothesis discovery only.
- Keep trading execution rule-based and reproducible.
- Expand many partially uncorrelated sleeves and evaluate them with richer metrics plus regime diagnostics.

## strategy families
- `crypto-metal-residual-pairs` | exec=rule_based_pair_spread | tf=30m, 1h, 4h | rationale=Build on current BTC/XAG watchlist progress while keeping execution deterministic.
- `sector-dispersion-reversion` | exec=cross_sectional_residual_reversion | tf=15m, 30m, 1h | rationale=Matches the article's emphasis on many small-capacity ensemble alphas.
- `lead-lag-regime-spillover` | exec=regime_gated_signal | tf=5m, 15m, 30m | rationale=Use LLM-style hypothesis generation but keep final execution rules explicit and auditable.
- `liquidity-shock-reversion` | exec=event_triggered_mean_reversion | tf=5m, 15m | rationale=Targets small-capacity intraday alpha consistent with the screenshots' discussion.
- `metals-lag-convergence` | exec=lagged_momentum_convergence | tf=4h, 1d | rationale=Targets XPT/XPD and XAU/XAG overlap windows where pair z-score can undertrade despite usable directional-relative structure.
- `regime-breakout-thrust` | exec=regime_filtered_breakout | tf=30m, 1h | rationale=Adds trend-following breadth without relying solely on the existing composite trend sleeve.
- `single-asset-zscore-reversion` | exec=single_asset_mean_reversion | tf=15m, 30m | rationale=Provides a low-complexity baseline mean-reversion family for the automated research loop.
- `intraday-vwap-reversion` | exec=vwap_deviation_reversion | tf=5m, 15m | rationale=Adds a classic execution-friendly reversion sleeve aligned with the article's many-small-alpha framing.
- `topcap-rotation-relative-momentum` | exec=cross_sectional_relative_momentum | tf=1h, 4h | rationale=Adds a lower-implementation-risk relative-value sleeve to complement pair-spread and lag-convergence stat-arb families.
- `vol-compression-break-reversion` | exec=volatility_compression_reversion | tf=5m, 15m, 1h | rationale=Extends existing vol-compression ideas with richer diagnostics instead of one-off tuning.
- `regime-conditioned-composite-trend` | exec=composite_trend_with_regime_filter | tf=30m, 1h, 4h | rationale=Preserve the current strict anchor while broadening the surrounding ensemble.

## operating rules
- Count memory against the global across all active sessions/services/workers total, not per process.
- Allow at most one heavy backtest run at a time.
- Record every finished run in the registry before considering a rerun.
- Prefer saved artifacts over recomputation for dashboards and deployment panels.
