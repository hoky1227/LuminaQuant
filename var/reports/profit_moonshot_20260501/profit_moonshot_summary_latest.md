# Profit Moonshot Research Summary

Generated: `2026-05-03T07:36:41.813258Z`
Decision: `no_deployment_ready_candidate`

## Current best

- Best new OOS-positive candidate: `profit_moonshot_derivatives_taker_flow_sparse_mode` (`shadow_review_only`).
- Conservative research baseline retained: `profit_moonshot_momentum_hybrid_safe_mode`.
- Deployment-ready: `false` because liquidation replay is absent, OI train/val relies on proxy, and the OOS edge is tiny.
- Latest tail refresh cutoff: `2026-05-03T04:10:03Z`.

## Strict raw-first table

| mode | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | strict status |
|---|---:|---:|---:|---:|---:|---|
| `profit_moonshot_momentum_hybrid_safe_mode` | -1.3551% | +0.2837% | -0.3832% | +3.3917% | -0.003750 | 보수 연구 후보 유지 / OOS 실패 |
| `profit_moonshot_derivatives_taker_flow_mode` | -1.4181% | -0.1302% | -0.0059% | +1.1541% | 0.000136 | 실패: val/OOS 약함 |
| `profit_moonshot_derivatives_taker_flow_sparse_mode` | -0.3765% | +0.0799% | +0.0247% | +0.8590% | 0.001444 | 최고 신규 shadow 후보 / deployment 불가 |

Session report: `var/reports/profit_moonshot_20260501/derivatives_oos/session_derivatives_taker_flow_report_20260503.json`
