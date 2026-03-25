# PRD: LuminaQuant dashboard approval refactor + Next migration under 8GB (2026-03-25)

## Problem statement
The LuminaQuant dashboard is mid-migration from a large Streamlit monolith (`apps/dashboard/app.py`) to a Next.js frontend (`apps/dashboard_web/`), but approval risk remains in two connected areas: (1) migration parity is still incomplete and uneven across slices, and (2) dashboard responsibilities remain spread between a giant Streamlit entrypoint and a growing set of bridge/service helpers without a clearly consolidated module boundary. The current wave must both preserve functionality and raise cold-review confidence by landing a sharper architectural seam: explicit Python payload services + clearer Next parity surfaces + safer Streamlit extraction on touched paths.

## Desired outcome
- Keep existing dashboard/operator behavior working while continuing the React/Next migration.
- Finish and harden the currently in-progress exact-window parity slice.
- Refactor touched dashboard paths into clearer service/OOP/module boundaries without introducing regressions.
- Preserve `uv run lq dashboard` and explicit launch-contract behavior during the hybrid period.
- Keep the entire execution/verification posture under the 8GB hardware cap.
- End with evidence strong enough for an APPROVE-grade assessment.

## Scope
### In scope
1. Stabilize the Next dashboard parity slices already present (`overview`, `workflow-jobs`, `risk-health`, `exact-window`) around explicit Python payload contracts.
2. Consolidate Python-side dashboard bridge/service responsibilities under `src/lumina_quant/dashboard/` with clear module ownership and test coverage.
3. Extract another safe dashboard state/query seam from `apps/dashboard/app.py` into service modules for touched paths only.
4. Preserve and verify dashboard CLI/launch compatibility (`src/lumina_quant/cli/dashboard.py`).
5. Update docs/tests/verification so the hybrid dashboard path is reproducible and approval-safe.
6. Run sequential low-memory verification and post-cleanup reassessment.

### Out of scope
- Whole-repo refactor outside the dashboard/migration hotspot lane.
- Full one-wave deletion of Streamlit.
- New dependencies beyond the already-introduced minimal Next stack.
- Concurrent heavy verification or >1 active team worker.

## Key evidence
- Existing migration artifacts already exist: `.omx/plans/prd-dashboard-react-nextjs-migration.md` and `.omx/plans/test-spec-dashboard-react-nextjs-migration.md`.
- Existing approval-refactor artifacts already exist: `.omx/plans/prd-luminaquant-full-refactor-approve-8gb.md` and `.omx/plans/test-spec-luminaquant-full-refactor-approve-8gb.md`.
- Current working tree already contains a new exact-window parity slice WIP in:
  - `apps/dashboard_web/app/exact-window/page.tsx`
  - `apps/dashboard_web/components/exact-window-runtime.tsx`
  - `apps/dashboard_web/lib/exact-window-server.ts`
  - `src/lumina_quant/dashboard/exact_window_service.py`
- Dashboard migration touchpoints remain clustered in:
  - `apps/dashboard/app.py`
  - `apps/dashboard/services/*.py`
  - `src/lumina_quant/dashboard/*.py`
  - `apps/dashboard_web/*`
- Memory policy remains explicit in the repo context: keep heavy verification sequential and total execution posture under 8GB.

## Success criteria
- Exact-window Next parity slice is integrated, tested, and documented.
- Touched dashboard payload services live behind explicit Python modules with stable JSON-oriented contracts.
- `apps/dashboard/app.py` is thinner on the touched state/query path after safe service extraction.
- `uv run lq dashboard` launch behavior remains explicit and regression-tested.
- Targeted dashboard tests, frontend lint/typecheck/build/tests, and sequential repo-wide quality gates pass.
- Final reassessment can argue for APPROVE based on reduced hotspot risk plus preserved behavior.

## User stories
### US-001 Next operator parity
As an operator, I want the Next dashboard to expose the existing key parity slices through explicit Python-backed payloads so migration can continue without hidden Streamlit coupling.

**Acceptance criteria**
- `overview`, `workflow-jobs`, `risk-health`, and `exact-window` all resolve through explicit bridge/service payloads.
- Payload failures return clear, stable empty/error states instead of opaque breakage.
- Tests lock route contract expectations.

### US-002 Maintainer-friendly dashboard boundaries
As a maintainer, I want touched dashboard state/query code moved out of `apps/dashboard/app.py` into focused service modules so the monolith keeps shrinking without a flag-day rewrite.

**Acceptance criteria**
- At least one additional state/query seam is extracted from `app.py` into service code.
- `app.py` preserves behavior through thin wrappers/imports.
- Regression tests cover the extracted contract.

### US-003 Compatibility-safe launch path
As an existing user, I want `uv run lq dashboard` and the hybrid migration path to remain explicit and stable so my workflow does not break while the UI is being migrated.

**Acceptance criteria**
- Dashboard CLI tests cover current launch mode behavior.
- Docs/readme explain the hybrid period accurately.
- No regression in current Streamlit fallback behavior for non-migrated surfaces.

### US-004 Approval-safe verification under 8GB
As the repo owner, I want this wave implemented and verified under the 8GB cap so the workflow does not OOM and the final review is defensible.

**Acceptance criteria**
- <=1 active team worker during execution.
- No concurrent heavy verification.
- Sequential frontend + Python + repo-wide verification runs pass.
- Final reassessment explicitly addresses architecture, behavior preservation, and memory safety.
