# Exact-Window Fail Analysis

- Generated at: `2026-03-15T07:55:34.652159+00:00`
- Source summary generated at: `2026-03-15T07:55:34.572280+00:00`
- Evaluated candidates: 34
- Promoted candidates: 0
- Clamp max timestamp: `2026-03-07T10:00:00+00:00`

## Rejection Counts

| Reason | Count |
|---|---:|
| oos_sharpe | 51 |
| pbo | 49 |
| trade_count | 12 |
| train_hurdle | 7 |

## Suggested Next Steps

- oos_sharpe: Prioritize evidence-backed parameter expansion or new candidate families on 15m, 1h, 30m, 4h where OOS Sharpe dominates failures. (count=51, timeframes=15m, 1h, 30m, 4h)
- pbo: Reduce overfit pressure by simplifying parameter grids and keeping only the most stable families on the affected timeframes. (count=49, timeframes=1h, 30m, 4h)
- trade_count: Increase trade opportunity on the flagged timeframes via broader parameter grids, symbol additions, or lower-frequency entry filters. (count=12, timeframes=1h, 30m, 4h)
