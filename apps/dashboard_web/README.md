# Dashboard Web Foundation

This workspace contains the first React/Next.js migration slice for the existing Streamlit dashboard.

## Scope

- minimal Next.js + TypeScript app-router scaffold
- shell/navigation that mirrors the current Streamlit information architecture
- typed Python compatibility bridge exposed at `/api/python-bridge`
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
- Exact-window research remains owned by the Python dashboard until follow-on routes are implemented
- The first slice is SSR-first and metadata-only to stay safe on the 8GB baseline
