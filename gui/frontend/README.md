# AIWS GUI Frontend

The web UI for the AIWS reconstruction GUI: a React + TypeScript single-page app built with Vite. It talks to the FastAPI backend over a small JSON API and is compiled (`npm run build`) into a static bundle that the backend serves at `/`. For the full system (backend, execution model, deployment), see [`../README.md`](../README.md).

## Stack

- React 19 + TypeScript, bundled by Vite.
- Tailwind CSS with shadcn/ui components (in `src/components/ui`).
- TanStack Query for server state and job polling.
- three.js via `@react-three/fiber` + `drei` for the result mesh and point-cloud viewers.
- i18next for a bilingual UI (English / 中文).
- Vitest + Testing Library for unit tests.

## Structure

```text
src/
├── main.tsx              # app entry
├── App.tsx               # top-level layout; switches between the step views
├── views/                # ConfigureView, LiveView, ResultView (configure → live → result)
├── components/           # feature components (3D viewers, stepper, log console, metric cards, ...)
│   └── ui/               # shadcn/ui primitives
├── api/                  # typed API client (client.ts) + response types (types.ts)
├── hooks/                # custom hooks (e.g. useJobPolling for live job status)
├── lib/                  # format, stages, validation, and utility helpers
├── i18n/                 # i18next setup + en.json / zh.json
└── assets/               # static images and sample inputs
```

## Develop

```bash
npm install            # first time
npm run dev            # Vite dev server on http://localhost:5173
```

The dev server proxies the API routes (`/health`, `/catalog`, `/jobs`, `/preview`, `/outputs`) to the backend at `http://127.0.0.1:18000`, so keep a backend running alongside it. Point at a different backend with `VITE_BACKEND_URL`:

```bash
VITE_BACKEND_URL=http://127.0.0.1:18000 npm run dev
```

## Build

```bash
npm run build          # type-checks (tsc -b), then writes the static bundle to dist/
```

The backend mounts `dist/` at `/`, so a build is required before the backend can serve the UI. Assets use relative URLs (`base: ""` in `vite.config.ts`), so the same bundle works at the server root and behind an SSH tunnel.

## Test, type-check, lint

```bash
npm test               # unit tests (Vitest)
npm run test:watch     # watch mode
npm run typecheck      # tsc --noEmit -p tsconfig.app.json
npm run lint           # eslint
```
