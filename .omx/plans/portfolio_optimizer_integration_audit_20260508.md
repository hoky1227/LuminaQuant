# Portfolio Optimizer Integration Audit — 2026-05-08

## Baseline preserved

- Baseline ref: `private/main e4b63bf80e368af80c8e09404c6d2a4089d8b956`
- Baseline Actions: `private-ci 25501812630` success, `ci 25501812724` success.
- Source plan: `.omx/plans/ralplan-portfolio-optimizer-integration-cleanup-20260507.md`
- Team runtime note: `omx team` launched `luminaquant-portfolio-0d950126`, but all four workers exited before claiming work; shutdown produced `merge_outcome: noop`, so the implementation proceeded under Ralph single-owner fallback.

## Bounded ownership / implementation map

- `AGENTS.md`: marker-bounded LuminaQuant tree/ownership map (`<!-- LQ:TREE:START -->` / `<!-- LQ:TREE:END -->`) inserted before source implementation.
- `src/lumina_quant/portfolio/optimizer_core.py`: shared portfolio stream/metric/objective-policy payload core.
- `src/lumina_quant/portfolio/hybrid_objective.py`: hybrid-governor-specific objective profiles and scoring formulas.
- `scripts/run_portfolio_optimization.py`: generic CLI wrapper; retains memory guard and report schema, delegates stream/metric hot path to shared core.
- `scripts/research/tune_profit_moonshot_fresh_portfolio.py`: reuses shared safe-float helper; selection contract remains train/validation primary with OOS report-only.
- `scripts/research/optuna_tune_profit_moonshot_calendar.py`: reuses shared safe-float helper; existing locked-OOS policy remains intact.
- `scripts/research/tune_hybrid_online_portfolio.py`: emits objective policy metadata and defaults to `locked_train_val`; diagnostic OOS-scored profiles remain opt-in and labeled non-selection-authoritative.
- `scripts/research/optuna_tune_hybrid_online_portfolio.py`: emits objective policy metadata, defaults to `locked_train_val`, keeps `live_guarded` and `train_aware_guarded` as diagnostic profiles only.
- `scripts/research/validate_profit_moonshot_pass_under_8gb.py`: behavior preserved; no source change required. Existing tests continue to cover RSS JSON/JSONL/time parsing, CI evidence, and push-evidence requirements.

## Behavior locks added before extraction

- `tests/test_portfolio_optimizer_core.py`
  - duplicate timestamp aggregation and deterministic sorted stream output;
  - `validation`/`val` split alias compatibility without relying on warmed cache state;
  - objective policy payload locked-OOS labeling;
  - hybrid tuning/Optuna locked objective ignores OOS metric changes while diagnostic `live_guarded` remains OOS-sensitive.
- `tests/test_run_portfolio_optimization_script.py`
  - generic optimizer validates `objective_policy=train_val_only_locked_oos_report` and `oos_is_objective_input=false` when fitting on validation and reporting OOS.

## Shared core extraction

Moved duplicated stream/payload helpers into `optimizer_core.py` and hybrid-specific objective scoring into `hybrid_objective.py`:

- safe finite float coercion;
- split canonicalization and split stream/metrics helpers;
- stream timestamp normalization, duplicate timestamp aggregation, deterministic alignment;
- NumPy array conversion, return metrics, correlation clustering;
- reusable `StreamCache` for aggregate/array reuse across clustering, fitting, and report-stream construction;
- generic objective policy payloads plus hybrid tuning objective profiles in a narrow `hybrid_objective.py` module.

The generic optimizer CLI remains backward compatible: arguments, report filenames, memory guard lifecycle, scoring config, and cap behavior are preserved.

## Locked-OOS policy

- Default objective policy is `train_val_only_locked_oos_report`.
- OOS remains report/gate evidence for default optimizer/tuning flows.
- OOS-scored hybrid profiles are still available only under explicit diagnostic profile names and emit `diagnostic_oos_in_objective_not_selection_authority`.

## Memory guard / 8 GiB policy

- `scripts/run_portfolio_optimization.py` still wraps execution with `acquire_portfolio_memory_guard(...)` and emits `memory_policy_payload(...)`.
- Profit moonshot validator remains unchanged and continues to enforce `EIGHT_GIB_BYTES = 8 * 1024 * 1024 * 1024`.
- Microbenchmark peak RSS: `109727744` bytes (`~104.6 MiB`), under 8 GiB. See `.omx/plans/portfolio_optimizer_perf_20260508.json`.

## Local evidence so far

- `uv run --extra dev pytest -q tests/test_portfolio_optimizer_core.py` → `3 passed`.
- `uv run --extra dev pytest -q tests/test_portfolio_optimizer_core.py tests/test_run_portfolio_optimization_script.py tests/test_profit_moonshot_fresh_portfolio_tuning.py tests/test_optuna_tune_profit_moonshot_calendar.py tests/test_profit_moonshot_pass_under_8gb_validator.py` → `17 passed`.
- Final broad checks still run after deslop pass and before commit/push.

## Final verification addendum

- Full local suite: `uv run --extra dev pytest -q` → `1188 passed, 1262 warnings in 346.81s`.
- Focused gates after deslop: targeted pytest `17 passed in 2.66s`; ruff check passed; py_compile passed; `git diff --check` passed.

## Architect re-check

- Initial architect status: WATCH/no blockers due hybrid scoring living in generic `optimizer_core.py`.
- Corrective action: moved hybrid-specific profiles/scoring to `src/lumina_quant/portfolio/hybrid_objective.py`.
- Re-check status: CLEAR; no architectural blockers.
- Post-split verification: targeted 17-test subset passed in 3.02s; ruff, py_compile, `git diff --check` passed; full pytest `1188 passed, 1262 warnings in 369.57s`.
