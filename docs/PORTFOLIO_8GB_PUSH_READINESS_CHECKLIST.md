# Portfolio 8GB Push-Readiness Checklist

## Guard surfaces to preserve

- `src/lumina_quant/core/memory_budget.py`
  - Global active-memory cap: `8.0 GiB`
  - Heavy-run budget: `6.5 GiB`
  - Light-worker parallelism budget: `2`
- `src/lumina_quant/core/session_memory.py`
  - Shared session-memory lease file:
    `var/reports/exact_window_backtests/followup_status/session_memory_budget.lock`
- `src/lumina_quant/portfolio_split_contract.py`
  - Portfolio follow-up runs acquire both the shared session lease and the heavy-run lock.
  - Repo-root artifacts must win over `.omx/team/*/worktrees/*` fallbacks so worker copies do not shadow committed evidence.
- `scripts/research/refresh_final_portfolio_validation_data.py`
  - Effective budget is clamped to the safe session cap before any refresh fan-out.
  - Parallel worker estimation uses the explicit reserve + per-worker memory model.

## Worktree-safety checks

- Prefer committed repo artifacts over `.omx` worktree copies whenever both exist.
- Only fall back to `.omx/team/*/worktrees/*/...` when the repo-root artifact is absent.
- Keep session-memory lease paths rooted under repo `var/reports/...`, not worker-local temporary paths.
- Avoid concurrent heavy follow-up runs while validating push readiness; the shared lease/lock is the intended isolation boundary.

## Targeted verification commands

```bash
uv run ruff check tests/test_portfolio_split_contract.py

uv run pytest -q \
  tests/test_portfolio_split_contract.py \
  tests/test_exact_window_runtime.py \
  tests/test_portfolio_followup_memory_guard.py \
  tests/test_refresh_final_portfolio_validation_data_script.py \
  tests/test_run_portfolio_optimization_script.py
```

## CI / merge checklist

- [ ] Memory-policy defaults still advertise an 8 GiB global cap and a 6.5 GiB heavy-run budget.
- [ ] Repo artifact resolution still wins over `.omx` worktree fallbacks.
- [ ] Session-memory lease path still resolves to `var/reports/exact_window_backtests/followup_status/session_memory_budget.lock`.
- [ ] No command in the verification plan launches overlapping heavy portfolio runs.
- [ ] Targeted lint + pytest commands pass on the final tree before push/cherry-pick.
