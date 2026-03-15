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
- Candidate count: 19
- Evaluated count: 19

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LeadLagSpilloverStrategy | leadlag_spillover_15m_0.50_lag4 | 15m | -60.274 | 0 | 0 | -40.48% | -18.546 |
| MeanReversionStdStrategy | mean_reversion_std_15m_balanced_ls_64_2.00 | 15m | -16.402 | 0 | 0 | -11.90% | -2.510 |
| PairSpreadZScoreStrategy | pair_spread_15m_core_bnbusdt_trxusdt_2.6_0.70 | 15m | -1.495 | 0 | 0 | -2.42% | -2.484 |
| VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | 15m | -7.132 | 0 | 0 | -1.80% | -6.620 |
| VwapReversionStrategy | vwap_reversion_15m_guarded_lo_48_0.014 | 15m | -5.983 | 0 | 0 | -0.56% | 0.077 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| pair_spread_15m_core_bnbusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 35.00% | -2.42% | -2.484 |
| volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | VolCompressionVWAPReversionStrategy | 35.00% | -1.80% | -6.620 |
| leadlag_spillover_15m_0.50_lag4 | LeadLagSpilloverStrategy | 14.38% | -40.48% | -18.546 |
| vwap_reversion_15m_guarded_lo_48_0.014 | VwapReversionStrategy | 9.40% | -0.56% | 0.077 |
| mean_reversion_std_15m_balanced_ls_64_2.00 | MeanReversionStdStrategy | 6.22% | -11.90% | -2.510 |

## Portfolio Monthly Hurdle

- 2026-02: return=-8.01%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-1.25%, btc=3.40%, threshold=3.40%, pass=False
