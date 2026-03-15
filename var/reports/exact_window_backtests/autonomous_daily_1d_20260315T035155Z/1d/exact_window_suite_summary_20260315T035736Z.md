# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, XRP/USDT, BNB/USDT, SOL/USDT, TRX/USDT, DOGE/USDT, ADA/USDT, TON/USDT, AVAX/USDT
- Excluded symbols: XAU/USDT, XAG/USDT, XPT/USDT, XPD/USDT
- Candidate count: 45
- Evaluated count: 45

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| PairSpreadZScoreStrategy | pair_spread_1d_balanced_btcusdt_trxusdt_1.4_0.30 | 1d | 7.628 | 0 | 1 | 5.67% | 2.269 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| pair_spread_1d_balanced_btcusdt_trxusdt_1.4_0.30 | PairSpreadZScoreStrategy | 100.00% | 5.67% | 2.269 |

## Portfolio Monthly Hurdle

- 2026-02: return=3.98%, btc=-12.99%, threshold=2.00%, pass=True
- 2026-03: return=1.63%, btc=3.40%, threshold=3.40%, pass=False
