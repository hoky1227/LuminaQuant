# Post-refresh Round-2 Shortlist

- generated_at: `2026-03-17T12:37:11.122468+00:00`
- purpose: propose the next genuinely new hypothesis families after both the new-hypothesis refresh queue and the first post-refresh shortlist were fully exhausted
- rule: prioritize ideas that are materially different from pair/trend/topcap momentum, generic liquidity shock fade, vol-of-vol fade, breadth failure reversal, and residual basket reversion

## Proposed round-2 families

### 1. Funding / liquidation crowding fade 30m
- thesis: fade crowded positioning only when funding, open-interest expansion, and liquidation imbalance all align as a short-horizon exhaustion event
- why new: distinct from carry because entry depends on the shock regime, not steady carry level; distinct from liquidity-vacuum fade because the signal is derivatives crowding, not session timing
- implementation shape: reuse existing `crowding_score` support series in the exact-window runtime and trade event-driven reversion after extreme crowding

### 2. Basis snapback reversion 30m
- thesis: large mark-vs-index basis dislocations can mean-revert even when outright price trend remains intact
- why new: distinct from VWAP / single-asset zscore because the anchor is derivatives basis rather than spot-like price history
- implementation shape: trigger on basis zscore extremes with bounded stop/timeout rules

### 3. Session-gated residual basket reversion 5m
- thesis: residual basket reversals may work only around repeated liquidity handoff windows, not all day
- why new: combines residual neutralization with the session-transition idea instead of testing each in isolation
- implementation shape: extend `ResidualBasketReversionStrategy` with UTC session gating

### 4. Cross-asset liquidation contagion fade 5m
- thesis: extreme liquidation bursts in one leader coin can transiently spill into related majors and mean-revert shortly after the contagion burst
- why new: distinct from lead-lag and shock fade because the trigger is liquidation spillover, not price momentum alone
- implementation shape: use liquidation notional support fields to detect contagion bursts and fade the secondary basket reaction

### 5. Multi-horizon trend exhaustion fade 30m
- thesis: when short, medium, and long momentum disagree after a final thrust, the exhaustion unwind may diversify the current incumbent better than another breakout/trend filter
- why new: distinct from regime-conditioned composite trend because the trade is a fade of trend exhaustion, not conditional trend continuation
- implementation shape: use the existing composite momentum helpers to detect disagreement / exhaustion states and trade reversion

## Suggested order

1. `Funding / liquidation crowding fade 30m`
2. `Basis snapback reversion 30m`
3. `Session-gated residual basket reversion 5m`
4. `Cross-asset liquidation contagion fade 5m`
5. `Multi-horizon trend exhaustion fade 30m`

## Recommendation

- If continuing immediately with the lowest implementation risk, start with `Funding / liquidation crowding fade 30m`.
- If optimizing for maximum distinctness versus all prior work, start with `Session-gated residual basket reversion 5m`.

## Execution progress

- `Funding / liquidation crowding fade 30m` → `completed_discard`
- `Basis snapback reversion 30m` → `completed_discard`
- `Session-gated residual basket reversion 5m` → `completed_discard`
- `Cross-asset liquidation contagion fade 5m` → `completed_discard`
- `Multi-horizon trend exhaustion fade 30m` → `completed_discard`
- Current round-2 status: `exhausted_without_improvement`
- Recommended next move: generate a round-3 shortlist or wait for the deferred 2026-03-31 retune.
