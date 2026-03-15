# Portfolio Four-Sleeve Retune Review — Worker 3

- generated_at: `2026-03-14T12:06:13Z`
- scope: review/document current repository state against `.omx/plans/prd-portfolio-four-sleeve-retune.md` and `.omx/plans/test-spec-portfolio-four-sleeve-retune.md`
- reviewer: `worker-3`
- status: `partial foundations present; PRD-specific four-sleeve retune path not yet implemented in this workspace snapshot`

## Executive Summary
Current repo surfaces already contain the incumbent bundle builder, the baseline exact-window freeze builder, the RollingBreakout gate writer, the portfolio memory-guard helper, and the incumbent-first max-performance decision writer. However, the requested four-sleeve retune lane is still incomplete: the RollingBreakout gate is still OOS-coupled, the freeze builder has no incumbent-aware anchor mode, explicit 8 GiB budget plumbing is absent, the anchored search wrapper is missing, anchored comparison wiring is missing, and the new search-wrapper test file is missing.

## Requirement Status Matrix
| Area | Status | Evidence |
| --- | --- | --- |
| Incumbent bundle surface exists | PASS | `scripts/research/build_portfolio_one_shot_incumbent_bundle.py` builds `portfolio_one_shot_incumbent_bundle_latest.*` |
| Freeze builder exists on incumbent/freeze surface | PARTIAL | `scripts/research/build_portfolio_exact_window_freeze.py:619-760` |
| RollingBreakout gate rebuilt on train+val only | FAIL | `scripts/research/rolling_breakout_30m_regime_gate.py:23-31,398-430,507-515` |
| Freeze supports incumbent-aware anchored mode | FAIL | `scripts/research/build_portfolio_exact_window_freeze.py:619-760` has no `selection_mode` / incumbent-anchor path |
| Explicit 8 GiB budget propagation | FAIL | `src/lumina_quant/portfolio_split_contract.py:144-168` has no fixed-budget arg and constructs `RSSGuard(log_path=..., label=...)` |
| Anchored sequential search wrapper | FAIL | `scripts/research/search_portfolio_four_sleeve_anchored.py` is missing |
| Anchored search tests | FAIL | `tests/test_search_portfolio_four_sleeve_anchored.py` is missing |
| Final comparison wired for anchored four-sleeve path | FAIL | `scripts/research/write_portfolio_max_performance_decision.py:15-24,261-417` only loads incumbent/tuned/dynamic/overlay/backbone artifacts |
| Existing targeted regression suite | PASS (current baseline only) | 18 targeted tests currently pass, but they do not lock the new PRD behaviors |

## Concrete Findings

### 1) RollingBreakout gate still leaks OOS into admission
- Threshold constants remain OOS-based: `scripts/research/rolling_breakout_30m_regime_gate.py:23-31` defines `oos_sharpe_min`, `oos_return_min`, and `oos_trade_count_min`.
- Rule scoring is still OOS-weighted: `scripts/research/rolling_breakout_30m_regime_gate.py:398-410` scores with OOS sharpe, OOS return, and OOS activation ratio.
- Survivor blocking still depends on OOS metrics: `scripts/research/rolling_breakout_30m_regime_gate.py:414-430`.
- Final rule selection still tie-breaks on OOS metrics: `scripts/research/rolling_breakout_30m_regime_gate.py:507-515`.
- Artifact payload does not advertise `selection_basis=train_val_only` or a dedicated `survives_train_val` field: `scripts/research/rolling_breakout_30m_regime_gate.py:540-560`.

### 2) Freeze builder does not yet implement incumbent-aware anchoring
- The public build function exposes no `selection_mode` or incumbent-anchor parameters: `scripts/research/build_portfolio_exact_window_freeze.py:619-639`.
- Selection still chooses the best local row per sleeve via the generic validation-only ranking path: `scripts/research/build_portfolio_exact_window_freeze.py:673-699`.
- The artifact still declares `selection_basis=validation_only`, not an incumbent-aware mode: `scripts/research/build_portfolio_exact_window_freeze.py:728-760`.
- No evidence of `rolling_admission_blocked`, 3-vs-4 row gating contract, or incumbent replacement threshold reporting appears in the build payload region.

### 3) Explicit budget propagation is not wired into the portfolio follow-up guard
- `acquire_portfolio_memory_guard()` accepts only `run_name`, `output_dir`, `input_path`, and `metadata`: `src/lumina_quant/portfolio_split_contract.py:144-150`.
- The helper instantiates `RSSGuard` without an explicit `budget_bytes`: `src/lumina_quant/portfolio_split_contract.py:165-168`.
- Existing runtime tests validate `RSSGuard` budget behavior in isolation, but not the portfolio guard handoff required by the PRD/test spec: `tests/test_exact_window_runtime.py:12-99`.

### 4) Anchored search lane is absent
- `scripts/research/search_portfolio_four_sleeve_anchored.py` does not exist.
- `tests/test_search_portfolio_four_sleeve_anchored.py` does not exist.
- No current repo search surfaced a `portfolio_four_sleeve_*` optimizer wrapper or artifact writer.

### 5) Final comparison writer is not yet wired to the anchored four-sleeve candidate set
- Current defaults point to incumbent, tuned, dynamic, overlay, and backbone artifacts only: `scripts/research/write_portfolio_max_performance_decision.py:15-24`.
- Candidate assembly covers only those existing artifact families: `scripts/research/write_portfolio_max_performance_decision.py:261-417`.
- The PRD-required anchored tuned result, prior tuned 4-sleeve result label separation, gate status notes, and anchored coverage are not yet represented in the current comparison scope.

### 6) Existing tests validate the old baseline, not the new PRD contract
- Freeze tests currently cover validation-only ranking and gate supplementation, but not incumbent-anchor thresholds or 3-row/4-row gate outcomes: `tests/test_build_portfolio_exact_window_freeze.py:356-435`.
- Gate tests currently assert OOS-positive behavior instead of train+val-only admission semantics: `tests/test_rolling_breakout_30m_regime_gate.py:164-199`.
- Decision-writer tests currently cover incumbent/tuned/dynamic/overlay/backbone comparisons, not anchored four-sleeve inclusion/gate-blocked retention: `tests/test_write_portfolio_max_performance_decision.py:104-240`.

## What Is Already Reusable
- Existing incumbent bundle builder is a good base for anchor seeding.
- Existing exact-window freeze builder already preserves the “candidate-detail first, gate supplement second” RollingBreakout pattern.
- Existing decision writer already applies an incumbent-first promotion ladder and can likely absorb the anchored candidate with a narrow extension.
- Existing targeted baseline tests are green, so follow-up changes can be evaluated incrementally.

## Verification Evidence
- File existence check: `test -f scripts/research/search_portfolio_four_sleeve_anchored.py` → `missing`
- File existence check: `test -f tests/test_search_portfolio_four_sleeve_anchored.py` → `missing`
- Tests: `uv run pytest -q tests/test_build_portfolio_exact_window_freeze.py tests/test_rolling_breakout_30m_regime_gate.py tests/test_build_portfolio_one_shot_incumbent_bundle.py tests/test_write_portfolio_max_performance_decision.py tests/test_exact_window_runtime.py` → `18 passed in 0.45s`
- Lint: `uv run ruff check scripts/research/build_portfolio_exact_window_freeze.py scripts/research/build_portfolio_one_shot_incumbent_bundle.py scripts/research/rolling_breakout_30m_regime_gate.py scripts/research/write_portfolio_max_performance_decision.py src/lumina_quant/portfolio_split_contract.py tests/test_build_portfolio_exact_window_freeze.py tests/test_build_portfolio_one_shot_incumbent_bundle.py tests/test_rolling_breakout_30m_regime_gate.py tests/test_write_portfolio_max_performance_decision.py tests/test_exact_window_runtime.py` → `All checks passed!`
- Compile: `uv run python -m py_compile scripts/research/build_portfolio_exact_window_freeze.py scripts/research/build_portfolio_one_shot_incumbent_bundle.py scripts/research/rolling_breakout_30m_regime_gate.py scripts/research/write_portfolio_max_performance_decision.py src/lumina_quant/portfolio_split_contract.py tests/test_build_portfolio_exact_window_freeze.py tests/test_build_portfolio_one_shot_incumbent_bundle.py tests/test_rolling_breakout_30m_regime_gate.py tests/test_write_portfolio_max_performance_decision.py tests/test_exact_window_runtime.py` → success

## Recommended Next Steps
1. Rework `scripts/research/rolling_breakout_30m_regime_gate.py` first so the admission artifact exposes train+val-only rule selection and `survives_train_val`.
2. Extend `scripts/research/build_portfolio_exact_window_freeze.py` with an incumbent-anchor mode and explicit `rolling_admission_blocked` handling.
3. Add fixed-budget plumbing to `acquire_portfolio_memory_guard()` and lock it with a focused regression test.
4. Add `scripts/research/search_portfolio_four_sleeve_anchored.py` plus `tests/test_search_portfolio_four_sleeve_anchored.py`.
5. Extend the comparison writer to ingest anchored four-sleeve outputs and gate-status notes.
6. Re-run the PRD-specified verification command set after implementation lands.
