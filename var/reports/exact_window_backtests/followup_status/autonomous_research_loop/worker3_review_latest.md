# Autonomous Portfolio Research Loop Review — Worker 3

- generated_at: `2026-03-14T14:11:13Z`
- scope: review/document current repository state against `.omx/plans/prd-autonomous-portfolio-research-loop.md` and `.omx/plans/test-spec-autonomous-portfolio-research-loop.md`
- reviewer: `worker-3`
- status: `partial_foundations_present_autonomous_loop_missing`

## Executive Summary
Current repo surfaces already contain strong reusable building blocks: research scoring with train-instability penalties, exact-window low-RAM exclusions, an explicit 8 GiB heavy-lock/memory-guard contract, a sequential anchored portfolio search, and a locked-OOS incumbent-vs-challenger promotion writer. The missing gap is orchestration/documentation for the new PRD itself: there is still no autonomous research-loop workflow/module/CLI, no durable experiment ledger/state/backlog/stack-audit artifact set under `autonomous_research_loop/`, no loop-level `keep` / `discard` / `crash` classification path, and no executable private-git milestone/sign-off gate outside the plan docs.

## Requirement Status Matrix
| Area | Status | Evidence |
| --- | --- | --- |
| Current train-instability evidence is available | PASS | `var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json` reports train return -10.1817% / sharpe -0.456 versus val return 5.0180% / sharpe 3.003 and OOS return 4.7899% / sharpe 1.707. |
| Research scoring already penalizes instability | PASS | `src/lumina_quant/strategy_factory/research_runner.py:59-107` defines instability sharpe/return/turnover penalties plus PBO, drawdown, and turnover controls. |
| Exact-window split + low-RAM selection contract exists | PASS | `src/lumina_quant/eval/exact_window_suite.py:23-30,37-65,762-789,1183-1189` fixes 2025-01-01 / 2026-01-01 / 2026-02-01 windows, excludes 1s micro by default, and keeps validation-only portfolio construction. |
| Explicit <8 GiB heavy-lane memory guard exists | PASS | `src/lumina_quant/portfolio_split_contract.py:36-38,92-100,148-183` plus `scripts/research/search_portfolio_four_sleeve_anchored.py:281-375` enforce a shared heavy lock, explicit budget bytes, checkpoint sampling, and finalize-on-failure behavior. |
| Broad strategy/alpha inventory is already reusable | PASS | `src/lumina_quant/strategy_factory/candidate_library.py:1008-1068,1176-1250,1322-1455` wires trend, lead-lag spillover, top-cap TSMOM, alpha101, pair spread, lag convergence, and carry sleeves; `src/lumina_quant/strategies/registry.py:40-120` exposes rare-event/micro/composite/carry strategy classes. |
| Locked-OOS incumbent promotion gate exists | PASS | `scripts/research/write_portfolio_max_performance_decision.py:374-496` aggregates challengers and retains/promotes only on locked-OOS promotion score; current winner is still the incumbent. |
| Autonomous experiment ledger with keep/discard/crash rows | FAIL | `scripts/run_research_candidates.py:662-705` writes point-in-time report/CSV/shortlist artifacts only; no `experiments.tsv` or loop-level lifecycle artifact exists. |
| Autonomous research state/backlog/stack-audit artifacts | FAIL | Required `autonomous_research_loop/` directory and PRD-named files are currently missing. |
| Loop-level crash policy / deterministic rerun surface | FAIL | `find src/lumina_quant/cli src/lumina_quant/workflows scripts -maxdepth 3 -type f | rg "autonomous|research_loop|experiment"` returned no runtime surface for the new loop. |
| Private-git milestone gate + architect sign-off execution surface | FAIL | Code search found references only inside `.omx/plans/*`; no workflow/CLI implementation was found under active runtime paths. |

## Concrete Findings

### 1) The train-instability problem is real and already measurable
- Current one-shot portfolio train metrics are materially weak: total_return=-10.1817%, sharpe=-0.456, max_drawdown=18.2608%.
- The same artifact remains positive on validation/OOS: val total_return=5.0180%, sharpe=3.003; oos total_return=4.7899%, sharpe=1.707.
- `var/reports/exact_window_backtests/followup_status/portfolio_max_performance_decision_latest.json` still retains the incumbent with reason: "No challenger cleared the locked-OOS promotion rule; keep the current one-shot incumbent.".

### 2) Robustness and memory foundations should be reused, not rewritten
- `research_runner` already scores candidates with explicit instability penalties, PBO penalties, drawdown penalties, turnover thresholds, and cost-stress thresholds.
- `exact_window_suite` already enforces a low-RAM profile by excluding `MicroRangeExpansion1sStrategy` / `1s` from the default exact-window lane.
- `portfolio_split_contract` already centralizes the canonical split contract, shared heavy-run lock path, explicit 8 GiB budget, RSS telemetry, and completion/failure summaries.
- The anchored four-sleeve search already demonstrates the desired single-heavy-lane pattern: checkpoint before each run, sample after completion, finalize on success/failure, then release the lock.

### 3) The candidate universe is broader than the current promotion loop is using
- In-repo candidate generation already includes lead-lag spillover, top-cap time-series momentum, alpha101 formulas, pair-spread variants, lag-convergence pair trades, carry/crowding sleeves, and research-only micro-range expansion.
- The centralized strategy registry also exposes rare-event, composite trend, volatility-compression reversion, carry, and micro-range classes for downstream orchestration.
- This means the PRD can mostly wrap existing strategy surfaces instead of inventing a new alpha stack from scratch.

### 4) Reporting currently stops at snapshot artifacts, not experiment lifecycle tracking
- `scripts/run_research_candidates.py` currently writes `candidate_research_latest.json`, `candidate_research_latest.csv`, `strategy_factory_report_latest.json`, and a shortlist markdown file.
- Those outputs are useful inputs, but they do not encode the PRD-required hypothesis ledger fields, explicit status (`keep` / `discard` / `crash`), changed-files tracking, or crash-reason retention.
- Missing paths today: `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/experiments.tsv`, `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/research_state_latest.json`, `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/ideas_backlog_latest.md`, `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/stack_audit_latest.md`.

### 5) Portfolio promotion gating is present, but milestone gating is not
- `write_portfolio_max_performance_decision.py` already performs incumbent-vs-challenger comparison and blocks promotion when locked-OOS thresholds are not clearly exceeded.
- That satisfies the promotion half of the PRD, but there is still no runtime surface that records milestone cleanliness, requires sign-off, or conditionally performs a private-git push.

## Current Challenger Snapshot
- `Causal dynamic challenger` | promotion_score_delta=-1.4898 | promotable=False
- `Causal overlay challenger` | promotion_score_delta=-2.3318 | promotable=False
- `Backbone-preserving triplet search challenger` | promotion_score_delta=-3.6297 | promotable=False
- `Exact-window frozen tuned challenger` | promotion_score_delta=-4.0308 | promotable=False

## Recommended Next Steps
1. Add the `autonomous_research_loop/` artifact contract first: `experiments.tsv`, `research_state_latest.json`, `ideas_backlog_latest.md`, and `stack_audit_latest.md` (or worker-specific precursors folded into those shared files by the lead).
2. Wrap the existing candidate-research, exact-window, and portfolio-comparison surfaces in one serial loop that writes a ledger row for every experiment and crash.
3. Reuse the existing heavy-lock/RSS telemetry and locked-OOS promotion writer instead of building new gating logic.
4. Implement milestone gating separately from portfolio promotion: clean diff check, verification bundle path, sign-off state, then optional private-git push if credentials already work.

## Verification Evidence
- Search: `find src/lumina_quant/cli src/lumina_quant/workflows scripts -maxdepth 3 -type f | rg "autonomous|research_loop|experiment"` → no autonomous runtime surface found.
- Missing-artifact check: `autonomous_research_loop/` directory + PRD-named files are absent before this worker review artifact.
- Tests: `uv run pytest -q tests/test_search_portfolio_four_sleeve_anchored.py tests/test_write_portfolio_max_performance_decision.py tests/test_exact_window_runtime.py tests/test_build_portfolio_exact_window_freeze.py tests/test_run_research_candidates_script.py tests/test_run_portfolio_optimization_script.py` → `38 passed in 15.75s`.
- Lint: `uv run ruff check scripts/run_research_candidates.py scripts/run_portfolio_optimization.py scripts/research/search_portfolio_four_sleeve_anchored.py scripts/research/write_portfolio_max_performance_decision.py src/lumina_quant/eval/exact_window_suite.py src/lumina_quant/portfolio_split_contract.py src/lumina_quant/strategy_factory/candidate_library.py src/lumina_quant/strategy_factory/research_runner.py tests/test_search_portfolio_four_sleeve_anchored.py tests/test_write_portfolio_max_performance_decision.py tests/test_exact_window_runtime.py tests/test_build_portfolio_exact_window_freeze.py tests/test_run_research_candidates_script.py tests/test_run_portfolio_optimization_script.py` → `All checks passed!`.
- Compile: `uv run python -m py_compile ...` on the reviewed Python surfaces → success.

## Superseded Note
- This review was superseded by later repo changes. See `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/worker3_latest_state_addendum_latest.md` for the current state.

## Output
- Markdown review: `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/worker3_review_latest.md`
- JSON review: `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/worker3_review_latest.json`

- Refinement addendum: `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/worker3_refinement_addendum_latest.md`
