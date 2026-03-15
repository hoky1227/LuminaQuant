# Web-grounded candidate lanes (2026-03-15)

Anchor incumbent to beat:
- pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55
- composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80
- topcap_tsmom_1h_balanced_16_4_0.015
- locked-OOS: return 5.5960%, Sharpe 3.447, max DD 1.4277%

## Lane A — Residual / factor-neutral topcap momentum
- Source idea: residual momentum / factor-neutral momentum, plus cross-sectional crypto network structure.
- Low-risk implementation sketch:
  - keep current TopCapTimeSeriesMomentumStrategy shell
  - replace raw relative momentum score with residualized score (remove BTC / basket / sector-style common component)
  - optionally add breadth / network spillover filter as a regime gate rather than a new engine
- Likely touchpoints:
  - src/lumina_quant/strategies/
  - src/lumina_quant/strategy_factory/
  - existing topcap candidate generation/tuning surfaces

## Lane B — Carry + momentum hybrid sleeve
- Source idea: carry + time-series momentum literature.
- Low-risk implementation sketch:
  - start from existing PerpCrowdingCarryStrategy and TopCap/Composite momentum features
  - gate carry by trend persistence or blend carry rank + momentum rank
  - avoid creating a full new allocator first; produce a sleeve candidate that can enter exact-window directly
- Likely touchpoints:
  - existing carry strategy candidate factory
  - momentum signal components already used by topcap/composite

## Lane C — Improved pair-state / multivariate market-neutral sleeve
- Source idea: multivariate market-neutral pair trading and adaptive hedge / state filters.
- Low-risk implementation sketch:
  - keep current PairSpreadZScoreStrategy execution skeleton
  - add stronger entry filters: rolling participation threshold, state filter, or alternative hedge stability test
  - prefer deterministic Kalman-like / rolling-regression approximations over full RL
- Likely touchpoints:
  - existing pair spread strategy candidate factory
  - exact-window pair focus / retune scripts

## Lane D — Crash-aware momentum gate
- Source idea: time-series momentum with crash / regime conditioning.
- Low-risk implementation sketch:
  - retain CompositeTrendStrategy core signal
  - add a crash-avoidance gate using realized vol spike / breadth failure / BTC-below-MA state
  - test as a regime-conditioned variant, not as a separate allocator
- Likely touchpoints:
  - CompositeTrendStrategy metadata / regime feature plumbing
  - dynamic/overlay regime helpers for reusable gate features

## Execution order recommendation
1. Lane A
2. Lane B
3. Lane D
4. Lane C

Reason:
- A/B/D reuse the most existing code and can likely be tuned fastest under the 8 GiB budget.
- C is promising but easier to overcomplicate; keep it deterministic.

## Lane E — Execution / risk-management search
- Treat sizing and exit logic as search dimensions, not afterthoughts.
- Feasible low-risk variants inside current code surfaces:
  - cash ratio / cash buffer
  - fractional Kelly / capped Kelly / blended incumbent+Kelly sizing
  - max weight / family cap / target vol changes
  - stop-loss / trailing stop / max-hold / rebalance frequency
  - take-profit / split-exit approximations if current backtest surface supports them cleanly
- Keep deterministic and memory-safe; avoid feature creep unless existing backtest surfaces already support it.
