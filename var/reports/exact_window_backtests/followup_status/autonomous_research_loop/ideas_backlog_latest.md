# Autonomous Research Ideas Backlog

- Generated at: `2026-03-15T14:27:32.243039+00:00`
- Candidate universe size: `517`
- Backlog timeframes: `5m, 15m, 30m, 1h, 4h, 1d`
- Incumbent strategy classes: `CompositeTrendStrategy, PairSpreadZScoreStrategy, TopCapTimeSeriesMomentumStrategy`

## Highest-priority in-repo experiments

- `PerpCrowdingCarryStrategy` via `perp_crowding_carry_30m_0.25_0.08` | candidates=9 | already_in_incumbent=False | tf=30m
- `LeadLagSpilloverStrategy` via `leadlag_spillover_5m_0.25_lag2` | candidates=18 | already_in_incumbent=False | tf=5m
- `Alpha101FormulaStrategy` via `alpha101_formula_1h_a005_a005_vwap_tuned_dir` | candidates=6 | already_in_incumbent=False | tf=1h
- `VolCompressionVWAPReversionStrategy` via `volcomp_vwap_rev_guarded_5m_guarded_lo_core_2.20_0.12` | candidates=4 | already_in_incumbent=False | tf=5m
- `LagConvergenceStrategy` via `lag_convergence_4h_metals_core_xauusdt_xagusdt_2_0.018` | candidates=8 | already_in_incumbent=False | tf=4h
- `RollingBreakoutStrategy` via `rolling_breakout_30m_loose_lo_48_0.001` | candidates=4 | already_in_incumbent=False | tf=30m
- `RegimeBreakoutCandidateStrategy` via `regime_breakout_30m_trend_guarded_48_0.68` | candidates=4 | already_in_incumbent=False | tf=30m
- `TopCapTimeSeriesMomentumStrategy` via `topcap_tsmom_1h_balanced_16_4_0.015` | candidates=11 | already_in_incumbent=True | tf=1h

## Research-backed methodology upgrades

- Add stronger train-instability penalties or minimum-train gates before portfolio promotion.
- Reuse exact-window validation artifacts as the canonical duplicate/history source instead of adding a parallel scheduler.
- Keep HRP / risk-parity / volatility-managed portfolio variants behind the existing locked-OOS promotion rule.
- Keep dynamic and overlay allocators under the explicit 8 GiB memory contract and single-heavy-lane discipline.

## Pipeline thesis map

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

## Anti-repeat rules

- Reuse `exact_window_run_registry.jsonl` and the canonical registry snapshot before launching a heavy rerun.
- Keep discarded challengers in the ledger; do not silently carry them forward as production candidates.
- Treat recovered log archives as crash context only, not as the canonical duplicate-signature index.
