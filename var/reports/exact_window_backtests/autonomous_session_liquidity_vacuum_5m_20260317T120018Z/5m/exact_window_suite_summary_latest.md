# Exact-Window Validation Suite

## Windows
- Train: `2025-01-01T00:00:00+00:00` → `2026-01-01T00:00:00+00:00`
- Validation: `2026-01-01T00:00:00+00:00` → `2026-02-01T00:00:00+00:00`
- OOS requested end-exclusive: `2026-03-09T00:00:00+00:00`
- OOS actual end-exclusive: `2026-03-07T10:00:00.001000+00:00`
- Actual max timestamp used: `2026-03-07T10:00:00+00:00`

## Universe
- Eligible symbols: BTC/USDT, ETH/USDT, BNB/USDT
- Excluded symbols: 
- Candidate count: 8
- Evaluated count: 8

## Best Per Strategy

| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| LiquidityShockReversionStrategy | liquidity_shock_reversion_5m_thin_lo_72_0.010 | 5m | -5.432 | 0 | 0 | -0.38% | -0.245 |
| SessionLiquidityVacuumFadeStrategy | session_liquidity_vacuum_fade_5m_utc_guarded_lo_64_0.008 | 5m | -2.655 | 0 | 0 | -1.38% | -1.611 |
| VolCompressionVWAPReversionStrategy | volcomp_vwap_rev_guarded_5m_guarded_lo_strict_2.60_0.10 | 5m | -23.865 | 0 | 0 | -0.84% | -2.622 |
| VwapReversionStrategy | vwap_reversion_5m_balanced_ls_48_0.012 | 5m | -1.410 | 0 | 0 | 0.55% | 0.350 |

## Portfolio

- Construction basis: `best_per_strategy_fallback`

| Name | Strategy | Weight | OOS Return | OOS Sharpe |
|---|---|---:|---:|---:|
| session_liquidity_vacuum_fade_5m_utc_guarded_lo_64_0.008 | SessionLiquidityVacuumFadeStrategy | 35.00% | -1.38% | -1.611 |
| volcomp_vwap_rev_guarded_5m_guarded_lo_strict_2.60_0.10 | VolCompressionVWAPReversionStrategy | 35.00% | -0.84% | -2.622 |
| liquidity_shock_reversion_5m_thin_lo_72_0.010 | LiquidityShockReversionStrategy | 25.88% | -0.38% | -0.245 |
| vwap_reversion_5m_balanced_ls_48_0.012 | VwapReversionStrategy | 4.12% | 0.55% | 0.350 |

## Portfolio Monthly Hurdle

- 2026-02: return=-0.48%, btc=-12.99%, threshold=2.00%, pass=False
- 2026-03: return=-0.36%, btc=3.40%, threshold=3.40%, pass=False
