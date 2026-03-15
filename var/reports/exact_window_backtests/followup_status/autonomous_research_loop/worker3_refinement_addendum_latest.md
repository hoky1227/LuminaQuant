# Autonomous Research Loop Review Addendum — Worker 3

- generated_at: `2026-03-14T14:15:00Z`
- reviewer: `worker-3`
- scope: leader/critic refinement on registry reuse and explicit 8 GiB budget injection

## Refinement 1 — Reuse the existing registry/locks; do not add a second scheduler
- The canonical exact-window signature registry already exists at `var/reports/exact_window_backtests/exact_window_run_registry.jsonl`.
- `src/lumina_quant/cli/exact_window.py:178-190` resolves and iterates that registry directly.
- `src/lumina_quant/workflows/alpha_research_pipeline.py:237-267` already encodes `duplicate_policy="skip_if_signature_exists_in_exact_window_run_registry_jsonl_unless_forced"` and surfaces the registry as a recommended output.
- Review implication: the new autonomous ledger should be an index over existing run/summary/memory artifacts, not a second scheduler or duplicate queue.

## Refinement 2 — Concrete executable verification for dynamic/overlay 8 GiB budget injection
### Code evidence
- `scripts/research/run_causal_dynamic_portfolio.py:688-692` calls `acquire_portfolio_memory_guard(...)` without `budget_bytes=`.
- `scripts/research/run_causal_dynamic_portfolio.py:735-736` emits `memory_policy_payload()` with no explicit budget argument.
- `scripts/research/run_causal_overlay_portfolio.py:325-330` calls `acquire_portfolio_memory_guard(...)` without `budget_bytes=`.
- `scripts/research/run_causal_overlay_portfolio.py:383-384` emits `memory_policy_payload()` with no explicit budget argument.
- By contrast, `scripts/research/search_portfolio_four_sleeve_anchored.py:281-287` explicitly passes `budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES`.

### Executable verification result
- `uv run pytest -q tests/test_portfolio_followup_memory_guard.py tests/test_search_portfolio_four_sleeve_anchored.py tests/test_causal_dynamic_portfolio.py tests/test_causal_overlay_portfolio.py` → `2 failed, 23 passed in 16.28s`.
- The only failures are:
  - `tests/test_portfolio_followup_memory_guard.py::test_dynamic_report_passes_explicit_8gib_budget_and_emits_it`
  - `tests/test_portfolio_followup_memory_guard.py::test_overlay_report_passes_explicit_8gib_budget_and_emits_it`
- Failure detail: both assertions observed `captured.get("budget_bytes") is None` instead of `8589934592` (`8 * 1024**3`).
- Worker-side executable monkeypatch verification in `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/worker3_budget_verification_latest.json` independently reproduced the same outcome:
  - dynamic: `acquire_budget_bytes=null`, emitted explicit budget bytes = `null`, status=`fail`
  - overlay: `acquire_budget_bytes=null`, emitted explicit budget bytes = `null`, status=`fail`

## Conclusion
- The registry/locking refinement is supported by current repo evidence; no second scheduler/queue is needed.
- The dynamic and overlay portfolio lanes do **not** currently satisfy the explicit 8 GiB injection requirement.
- Review implication: budget-plumbing work remains outstanding specifically for `run_causal_dynamic_portfolio.py` and `run_causal_overlay_portfolio.py` (and optimizer-lane review should keep expecting the same explicit-budget standard).
