# Frontend

React + TypeScript chat UI, built with Vite. It talks only to the FastAPI backend; it holds no
LLM keys.

```bash
npm ci
npm run dev        # dev server on :3000 (or `make ui` from the repo root)
npm run build      # production build into dist/
npx tsc --noEmit   # typecheck, as CI runs it
```

Vite reads env files from the repo root, not from this folder. The only setting is
`VITE_API_BASE_URL` (in `.env`), the backend's address. Leave it empty to call the API on the
same origin, which is how the nginx image serves it (`nginx.conf` for compose,
`nginx.k8s.conf` for Kubernetes).

How to run the whole stack is in the [root README](../../README.md#getting-started).
