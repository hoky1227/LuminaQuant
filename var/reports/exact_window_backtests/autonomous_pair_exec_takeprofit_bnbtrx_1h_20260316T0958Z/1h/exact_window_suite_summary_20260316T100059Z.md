# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BNB/USDT, TRX/USDT
- Excluded symbols: 
- Candidate count: 18
- Evaluated count: 18

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| PairSpreadZScoreStrategy | pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70 | 1h | 13.283 | 0 | 0 | 11.10% | 7.481 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 100.00% | 11.10% | 7.481 |

## Portfolio Monthly Hurdle

- 2026-02: return=9.23%, btc=-12.99%, threshold=2.00%, pass=True
- 2026-03: return=1.71%, btc=3.40%, threshold=3.40%, pass=False
