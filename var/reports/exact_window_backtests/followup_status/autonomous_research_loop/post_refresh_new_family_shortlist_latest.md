# Post-refresh New Family Shortlist

- generated_at: `2026-03-17T11:31:52.901213+00:00`
- goal: propose materially new family directions now that the current new-hypothesis refresh queue is fully exhausted
- selection_rule: lower duplication risk than the exhausted queue, plausible portfolio diversification, implementable under the single-heavy-lane / <8 GiB contract

## Priority shortlist

### 1. BTC-beta-neutral single-asset zscore reversion
- timeframe: `15m` first, then `30m`
- thesis: de-trend each asset versus BTC before applying zscore reversion so trend crashes do not dominate the signal the way plain single-asset reversion just did
- why distinct: different from plain `single-asset-zscore-reversion` because the traded signal is the residual, not the raw close series
- implementation_risk: medium
- duplication_risk: low-medium
- recommended_first_cut: add residualization knobs to `MeanReversionStdStrategy` and rerun on the current topcap crypto basket

### 2. Session-transition liquidity vacuum fade
- timeframe: `5m`
- thesis: micro dislocations around Asia/Europe/US handoff windows revert more reliably than generic shock events because the time-of-day condition filters noise
- why distinct: different from `liquidity-shock-reversion` because it conditions on repeated session structure instead of raw shock magnitude alone
- implementation_risk: medium
- duplication_risk: low
- recommended_first_cut: extend the new liquidity-shock sleeve with session-window gating and rerun on BTC/ETH/BNB

### 3. Vol-of-vol exhaustion fade
- timeframe: `15m`
- thesis: extreme realized-vol spikes often mean-revert after the second-order volatility shock decays, which is different from the already-tested low-vol compression thesis
- why distinct: `vol-compression-break-reversion` traded failed breaks from low volatility; this would fade extreme high-vol events
- implementation_risk: medium
- duplication_risk: low-medium
- recommended_first_cut: derive a realized-vol zscore and trigger bounded reversion entries only after volatility expansion extremes

### 4. Breadth-thrust failure reversal
- timeframe: `30m`
- thesis: when too many basket members thrust in the same direction and then fail to hold, the unwind can diversify the incumbent pair/trend stack better than a plain breakout family
- why distinct: different from the exhausted breakout families because the trigger is basket breadth failure, not single-asset channel breakout
- implementation_risk: medium-high
- duplication_risk: low
- recommended_first_cut: build a breadth ratio / breadth momentum trigger on the current crypto basket and trade mean reversion after failed thrust

### 5. Cross-sectional residual basket reversion
- timeframe: `5m` or `15m`
- thesis: basket-level residual ranking may capture dispersion mean reversion better than the fixed-pair approximation used in the discarded 30m sector lane
- why distinct: unlike the discarded `sector-dispersion-reversion 30m`, this would rank and rebalance across a whole basket rather than choose one static pair
- implementation_risk: high
- duplication_risk: low
- recommended_first_cut: prototype as a dedicated cross-sectional residual engine only if the lighter shortlist ideas above fail

## Suggested order

1. `BTC-beta-neutral single-asset zscore reversion 15m`
2. `Session-transition liquidity vacuum fade 5m`
3. `Vol-of-vol exhaustion fade 15m`
4. `Breadth-thrust failure reversal 30m`
5. `Cross-sectional residual basket reversion 5m/15m`

## Recommendation

- If you want the lightest next implementation, start with `BTC-beta-neutral single-asset zscore reversion 15m`.
- If you want the most differentiated intraday idea from the exhausted queue, start with `Session-transition liquidity vacuum fade 5m`.

## Execution progress

- `BTC-beta-neutral single-asset zscore reversion 15m` → `discard` | OOS return=3.1744% | Sharpe=2.414 | max DD=1.6215%
- `Session-transition liquidity vacuum fade 5m` → `discard` | OOS return=2.9938% | Sharpe=2.675 | max DD=1.4309%
- `Vol-of-vol exhaustion fade 15m` → `discard` | OOS return=4.0743% | Sharpe=2.683 | max DD=2.2423%
- `Breadth-thrust failure reversal 30m` → `discard` | OOS return=4.0743% | Sharpe=2.683 | max DD=2.2423%
- `Cross-sectional residual basket reversion 15m` → `discard` | OOS return=3.2448% | Sharpe=3.113 | max DD=0.7517%

- Current shortlist status: `exhausted_without_improvement`
- Recommended next move: generate a round-2 shortlist or wait for the deferred 2026-03-31 retune.
