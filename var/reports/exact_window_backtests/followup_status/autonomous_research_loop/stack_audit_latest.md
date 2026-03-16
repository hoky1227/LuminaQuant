# Autonomous Research Stack Audit

- Generated at: `2026-03-16T10:21:58.343664+00:00`
- Incumbent train total return: 4.89%
- Incumbent validation total return: 3.14%
- Incumbent locked-OOS total return: 5.76%
- Current promotion winner: `Current one-shot incumbent` (retained_incumbent)
- Exact-window promoted_total: `0` | next_action=`ralplan_team_ralph_required`
- Heavy lock path: `/home/hoky/Quants-agent/LuminaQuant/.omx/team/continue-the-autonomous-portfo/worktrees/worker-1/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit total-memory budget: `8589934592` bytes (8 GiB)

## Highest-impact instability drivers

### Negative train sleeve contributions dominate the incumbent blend.
- TopCapTimeSeriesMomentumStrategy 1h weighted_train_return=-1.57%
- PairSpreadZScoreStrategy 1h weighted_train_return=2.00%
- CompositeTrendStrategy 30m weighted_train_return=2.37%

### Validation-only selection keeps the incumbent despite train fragility.
- train_total_return=4.89%
- val_total_return=3.14%
- oos_total_return=5.76%
- winner_status=retained_incumbent

### Robustness guardrails already exist and should be reused rather than replaced.
- src/lumina_quant/strategy_factory/research_runner.py
- src/lumina_quant/eval/exact_window_decision.py
- src/lumina_quant/eval/exact_window_runtime.py
- src/lumina_quant/portfolio_split_contract.py

## Incumbent sleeve contributors

| Strategy | TF | Weight | Train Return | Weighted Train Return | Train Sharpe | Train Stability | Min Rolling Sharpe | Val Return | OOS Return |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TopCapTimeSeriesMomentumStrategy | 1h | 14.56% | -10.78% | -1.57% | -0.425 | -1.547 | -27.304 | 1.84% | 3.24% |
| PairSpreadZScoreStrategy | 1h | 60.00% | 3.33% | 2.00% | 0.349 | -1.535 | -28.062 | 1.61% | 7.24% |
| CompositeTrendStrategy | 30m | 25.44% | 9.32% | 2.37% | 0.743 | -1.515 | -32.879 | 4.99% | 2.95% |

## File-backed stack map

- `src/lumina_quant/strategy_factory/research_runner.py` — candidate evaluation, instability penalties, train/val/OOS metrics, and robust ranking hooks.
- `src/lumina_quant/strategy_factory/candidate_library.py` — unused strategy/alpha inventory that should feed the autonomous backlog before adding new architecture.
- `scripts/run_portfolio_optimization.py` — validation-fit / locked-OOS-report portfolio constructor used by incumbent and challengers.
- `src/lumina_quant/eval/exact_window_decision.py` — promotion and candidate-pool decision logic for exact-window sweeps.
- `src/lumina_quant/eval/exact_window_runtime.py` + `src/lumina_quant/portfolio_split_contract.py` — single-heavy-lane lock and explicit memory guard surfaces.

## Audit conclusion

- The incumbent is still the locked-OOS winner, but train instability is real and concentrated in the cross-sectional + regime-breakout sleeves.
- The safest next step is not a new scheduler; it is a deterministic artifact index plus focused follow-up experiments that reuse the existing exact-window registry and heavy-lock contract.
