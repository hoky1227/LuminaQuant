# Dashboard Web Foundation

This workspace contains the first React/Next.js migration slice for the existing Streamlit dashboard.

## Scope

- minimal Next.js + TypeScript app-router scaffold
- shell/navigation that mirrors the current Streamlit information architecture
- overview/workflow/risk/exact-window parity routes backed by Python compatibility payloads
- typed frontend metadata plus a Python-backed compatibility bridge exposed at `/api/python/dashboard/overview`
- first Overview placeholder view with 8GB memory-budget guardrails

## Commands

```bash
npm install
npm run lint
npm run test
npm run build
```

## Compatibility notes

- Legacy Streamlit source remains `apps/dashboard/app.py`
- Exact-window research still runs on the Python side; the Next route reads the latest exported artifact bundle without re-running heavy jobs
- The migration stays SSR-first and Python-contract-backed to stay safe on the 8GB baseline
