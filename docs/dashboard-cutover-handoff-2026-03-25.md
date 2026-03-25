# Dashboard Cutover Handoff — 2026-03-25

## Current branch state
- Branch: `private-main`
- Latest pushed work should include:
  - dashboard approval refactor wave
  - cutover-prep wave 1
  - cutover-prep wave 2

## What is already done
### Approval/migration waves completed
1. Exact-window parity and dashboard contract cleanup
2. Workflow-jobs seam extraction
3. Risk/heartbeat seam extraction
4. Added Next parity routes for:
   - `/performance-price`
   - `/execution-analytics`
   - `/report-export`
   - `/market-data`
   - `/optimization-insights`
   - `/raw-data`

### Latest verification evidence
- targeted dashboard/python checks: green
- web test/lint/typecheck/build: green
- architecture/docs checks: green
- latest full suite evidence: `896 passed, 1 skipped`

## Important current conclusion
**Do not run the final retirement wave yet without planning.**
Immediate Streamlit retirement was architect-rejected earlier because a staged approach was required, then prep-wave 2 closed the remaining parity blockers. The next step is to execute the final retirement wave deliberately.

## Next intended wave
### Retirement wave goals
1. Flip `uv run lq dashboard` default launcher to **Next**
2. Retire `apps/dashboard/app.py` from the primary runtime path
3. Remove Streamlit-first assumptions from:
   - `pyproject.toml`
   - README / README_KR / runbooks
   - launcher tests and any obsolete Streamlit dashboard tests

## Planning artifacts prepared locally
These exist locally in `.omx/plans/` and should be consulted in the next session:
- `prd-luminaquant-dashboard-retirement-wave-8gb.md`
- `test-spec-luminaquant-dashboard-retirement-wave-8gb.md`
- `prd-luminaquant-dashboard-full-cutover-8gb.md`
- `test-spec-luminaquant-dashboard-full-cutover-8gb.md`
- `prd-luminaquant-dashboard-cutover-prep-wave-2-8gb.md`
- `test-spec-luminaquant-dashboard-cutover-prep-wave-2-8gb.md`

## Constraints to preserve
- Keep total active execution posture under **8GB**
- Use **<=1 team worker**
- Run heavy verification **sequentially**
- Preserve required functionality during cutover

## Recommended next command shape
Run a fresh deliberate plan/review cycle for the retirement wave, then use a single executor lane for implementation, followed by leader-owned Ralph verification.
