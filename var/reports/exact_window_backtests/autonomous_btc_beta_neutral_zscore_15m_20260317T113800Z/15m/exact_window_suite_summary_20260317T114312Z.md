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
- Candidate count: 23
- Evaluated count: 23

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LeadLagSpilloverStrategy | leadlag_spillover_15m_0.50_lag4 | 15m | -185.803 | 0 | 0 | -25.37% | -81.211 |
| LiquidityShockReversionStrategy | liquidity_shock_reversion_15m_thin_lo_64_0.015 | 15m | -6.147 | 0 | 0 | -0.50% | -1.884 |
| MeanReversionStdStrategy | mean_reversion_std_15m_resid_btc_guarded_lo_96_2.40 | 15m | -20.297 | 0 | 0 | -4.00% | -2.604 |
| PairSpreadZScoreStrategy | pair_spread_15m_core_bnbusdt_trxusdt_2.6_0.70 | 15m | 0.600 | 0 | 0 | -2.04% | -2.157 |
| VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | 15m | -4.575 | 0 | 0 | -0.33% | -1.817 |
| VwapReversionStrategy | vwap_reversion_15m_guarded_lo_48_0.014 | 15m | -4.556 | 0 | 0 | -1.21% | -0.153 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | VolCompressionVWAPReversionStrategy | 35.00% | -0.33% | -1.817 |
| liquidity_shock_reversion_15m_thin_lo_64_0.015 | LiquidityShockReversionStrategy | 30.92% | -0.50% | -1.884 |
| pair_spread_15m_core_bnbusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 14.75% | -2.04% | -2.157 |
| leadlag_spillover_15m_0.50_lag4 | LeadLagSpilloverStrategy | 13.32% | -25.37% | -81.211 |
| vwap_reversion_15m_guarded_lo_48_0.014 | VwapReversionStrategy | 3.04% | -1.21% | -0.153 |
| mean_reversion_std_15m_resid_btc_guarded_lo_96_2.40 | MeanReversionStdStrategy | 2.97% | -4.00% | -2.604 |

## Portfolio Monthly Hurdle

- 2026-02: return=-3.79%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-0.73%, btc=3.40%, threshold=3.40%, pass=False
