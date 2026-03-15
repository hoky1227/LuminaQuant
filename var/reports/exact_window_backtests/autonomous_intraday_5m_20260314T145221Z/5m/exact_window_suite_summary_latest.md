# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT
- Excluded symbols: 
- Candidate count: 13
- Evaluated count: 13

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LeadLagSpilloverStrategy | leadlag_spillover_5m_0.50_lag4 | 5m | -472.472 | 0 | 0 | -56.77% | -193.519 |
| VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_5m_guarded_lo_strict_2.60_0.10 | 5m | -29.285 | 0 | 0 | -0.56% | -1.725 |
| VwapReversionStrategy | vwap_reversion_5m_guarded_lo_64_0.016 | 5m | -6.831 | 0 | 0 | 3.80% | 1.462 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| leadlag_spillover_5m_0.50_lag4 | LeadLagSpilloverStrategy | 35.00% | -56.77% | -193.519 |
| volcomp_vwap_rev_guarded_5m_guarded_lo_strict_2.60_0.10 | VolCompressionVWAPReversionStrategy | 35.00% | -0.56% | -1.725 |
| vwap_reversion_5m_guarded_lo_64_0.016 | VwapReversionStrategy | 30.00% | 3.80% | 1.462 |

## Portfolio Monthly Hurdle

- 2026-02: return=-20.29%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-5.34%, btc=3.40%, threshold=3.40%, pass=False
