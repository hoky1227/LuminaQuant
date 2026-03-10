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
- Candidate count: 56
- Evaluated count: 56

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| CompositeTrendStrategy | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | 30m | 11.605 | 0 | 1 | 2.21% | 1.625 |
| LeadLagSpilloverStrategy | leadlag_spillover_15m_0.50_lag4 | 15m | -185.803 | 0 | 0 | -25.37% | -81.211 |
| MeanReversionStdStrategy | mean_reversion_std_30m_balanced_ls_48_1.80 | 30m | -21.972 | 0 | 0 | -13.13% | -2.855 |
| PairSpreadZScoreStrategy | pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70 | 1h | -5.692 | 0 | 0 | -2.11% | -3.833 |
| PerpCrowdingCarryStrategy | perp_crowding_carry_1h_0.45_0.15 | 1h | 0.000 | 0 | 0 | 0.00% | 0.000 |
| RegimeBreakoutCandidateStrategy | regime_breakout_1h_trend_ls_48_0.70 | 1h | 10.643 | 0 | 1 | 8.08% | 2.009 |
| RollingBreakoutStrategy | rolling_breakout_30m_guarded_ls_64_0.002 | 30m | 12.248 | 0 | 1 | 4.08% | 1.042 |
| VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | 15m | -4.577 | 0 | 0 | -0.33% | -1.817 |
| VwapReversionStrategy | vwap_reversion_15m_guarded_lo_48_0.014 | 15m | -4.557 | 0 | 0 | -1.21% | -0.153 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| perp_crowding_carry_1h_0.45_0.15 | PerpCrowdingCarryStrategy | 35.00% | 0.00% | 0.000 |
| volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | VolCompressionVWAPReversionStrategy | 35.00% | -0.33% | -1.817 |
| leadlag_spillover_15m_0.50_lag4 | LeadLagSpilloverStrategy | 10.61% | -25.37% | -81.211 |
| pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 6.82% | -2.11% | -3.833 |
| composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | 5.35% | 2.21% | 1.625 |
| vwap_reversion_15m_guarded_lo_48_0.014 | VwapReversionStrategy | 2.41% | -1.21% | -0.153 |
| rolling_breakout_30m_guarded_ls_64_0.002 | RollingBreakoutStrategy | 1.71% | 4.08% | 1.042 |
| regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | 1.56% | 8.08% | 2.009 |
| mean_reversion_std_30m_balanced_ls_48_1.80 | MeanReversionStdStrategy | 1.53% | -13.13% | -2.855 |

## Portfolio Monthly Hurdle

- 2026-02: return=-2.53%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-0.64%, btc=3.40%, threshold=3.40%, pass=False
