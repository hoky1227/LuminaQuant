# Test Spec: LuminaQuant dashboard approval refactor + Next migration under 8GB (2026-03-25)

## Verification goals
- Prove the touched Next parity slices are stable and backed by explicit Python contracts.
- Prove touched Streamlit/dashboard refactors preserve behavior.
- Prove dashboard CLI/launch compatibility remains intact.
- Prove the wave stays within the repo's low-memory execution policy.

## Expanded test plan
### Unit
- `uv run pytest tests/test_dashboard_exact_window_service.py tests/test_dashboard_bridge_contract.py tests/test_dashboard_overview_service.py tests/test_dashboard_risk_health_service.py tests/test_dashboard_workflow_jobs_payload_service.py -q`
- Touched Streamlit/dashboard service regression files for the extracted seam(s)
- Any new dashboard CLI/bridge contract tests added in this wave

### Integration
- `uv run pytest tests/test_dashboard_cli.py tests/test_dashboard_import_contract.py tests/test_dashboard_view_switch.py tests/test_exact_window_dashboard_loader.py -q`
- Touched dashboard service integration/regression tests for state/query extraction

### Frontend
- `npm --prefix apps/dashboard_web run test`
- `npm --prefix apps/dashboard_web run lint`
- `npm --prefix apps/dashboard_web run typecheck`
- `npm --prefix apps/dashboard_web run build`

### Observability / runtime evidence
- Verify API route presence in build output for `/api/python/dashboard/overview`, `/workflow-jobs`, `/risk-health`, `/exact-window`
- Capture `git status --short` before/after cleanup passes
- Capture team startup/status/shutdown evidence if `$team` is used

### Sequential repo-wide safety gates
- `uv run ruff check .`
- `uv run python scripts/check_architecture.py`
- `uv run python scripts/verify_docs.py`
- `uv run python scripts/audit_hardcoded_params.py`
- `uv run pytest -q`

### 8GB gate
- Run heavy checks sequentially only.
- If the repo's benchmark/8GB baseline gate is still required after touched changes, run:
  - `mkdir -p logs reports/benchmarks`
  - `/usr/bin/time -v uv run python scripts/benchmark_backtest.py --iters 1 --warmup 0 --output reports/benchmarks/ci_smoke.json 2>&1 | tee logs/ci_smoke.time.log`
  - `uv run python scripts/verify_8gb_baseline.py --benchmark reports/benchmarks/ci_smoke.json --time-log logs/ci_smoke.time.log --oom-log logs/ci_smoke.time.log --skip-dmesg --output reports/benchmarks/ci_smoke_8gb_gate.json`

## Failure gates
- Any regression in touched dashboard tests blocks completion.
- Any failing frontend lint/typecheck/build/test blocks completion.
- Any broken `uv run lq dashboard` compatibility behavior blocks completion.
- Any failing repo-wide static/safety gate blocks completion.
- Any 8GB gate regression blocks APPROVE.
- Any post-deslop regression blocks completion.

## Execution order
1. Lock the current touched dashboard behavior with targeted tests.
2. Finish/refactor one dashboard slice at a time with narrow local reruns.
3. Run frontend verification sequentially.
4. Run targeted dashboard Python verification sequentially.
5. Run repo-wide static + regression gates sequentially.
6. Run ai-slop-cleaner on changed files only.
7. Re-run required targeted/frontend/static gates after cleanup.
8. Run final cold reassessment for APPROVE.

## Team verification path
- Team lane may run only lane-local targeted tests.
- Leader/Ralph owns all heavy sequential verification, final cleanup, and approval evidence.
- Team must not run full `uv run pytest -q` concurrently with any other heavy command.

## Approval evidence checklist
- [ ] Next exact-window parity slice integrated and green
- [ ] Touched payload/service seams are explicit and tested
- [ ] Touched `app.py` extraction preserves behavior
- [ ] Dashboard CLI/launch contract preserved
- [ ] Frontend lint/typecheck/test/build green
- [ ] Ruff/architecture/docs/hardcoded-param gates green
- [ ] Full `uv run pytest -q` green
- [ ] 8GB gate green if re-triggered by the touched wave
- [ ] Post-deslop reruns green
- [ ] Final reassessment returns APPROVE
