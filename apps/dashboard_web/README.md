# Dashboard Web Foundation

This workspace now contains the primary React/Next.js dashboard runtime for LuminaQuant.

## Scope

- minimal Next.js + TypeScript app-router scaffold
- shell/navigation that preserves the former dashboard information architecture in Next.js
- overview/workflow/risk/exact-window parity routes backed by Python compatibility payloads
- typed frontend metadata plus a Python-backed compatibility bridge exposed at `/api/python/dashboard/overview`
- first Overview placeholder view with 8GB memory-budget guardrails

## Commands

```bash
npm install
npm run lint
npm run test
npm run typecheck
npm run build
```

## Compatibility notes

- The retired legacy entry stub remains `src/lumina_quant/dashboard/retired_stub.py` only to direct operators to the Next launcher
- Exact-window research still runs on the Python side; the Next route reads the latest exported artifact bundle without re-running heavy jobs
- The migration stays SSR-first and Python-contract-backed to stay safe on the 8GB baseline
- Use Node 20+ locally and in CI so the dashboard runtime matches the supported Next.js toolchain
