# RollingBreakout Watchlist Plan

- generated_at: `2026-03-14T12:30:00Z`
- status: `watchlist_only`
- production_portfolio: `retain current 3-sleeve incumbent`
- review_basis: `anchored four-sleeve challenger remained non-promotable on locked OOS`

## Production posture
- Keep production portfolio at:
  - CompositeTrend 30m: `35%`
  - TopCapTSMom 1h: `35%`
  - RegimeBreakout 1h: `30%`
- Do **not** promote RollingBreakout into production weights yet.

## Watchlist sleeve
- Sleeve: `rolling_breakout_30m_guarded_ls_64_0.002`
- Gate selection basis: `train_val_only`
- Current selected rule: `btc_above_ma192_and_breadth_ma192_ge_60`
- Signal lag: `1 day`
- Mode: `shadow / paper / monitoring only`

## Daily operating rule
Activate the watchlist sleeve only when the gate is true:
1. `BTC above MA192`
2. `breadth_ma192 >= 60%`
3. apply with `1-day lag`

If gate is false:
- no sleeve activation
- no production allocation change
- continue logging only

## Metrics to log each review cycle
- gated total return
- gated Sharpe / Sortino
- max drawdown
- turnover
- activation ratio
- trade count
- rolling 20-day return
- rolling 20-day drawdown

## Promotion checklist for future re-entry
Only reconsider production inclusion if all hold over the next review window:
- gated sleeve remains `survives_train_val = true`
- shadow return stays positive
- shadow Sharpe is meaningfully positive
- drawdown stays below the current unacceptable tail seen in prior 4-sleeve runs
- rerun anchored 4-sleeve search + final decision writer
- new four-sleeve candidate clears incumbent on locked-OOS promotion rule

## Current conclusion
- RollingBreakout is valid as a **gated watchlist sleeve**.
- It is **not yet valid as a production portfolio sleeve**.
- Required next action is monitoring, not promotion.
