# AIWS End-to-End GUI

The GUI is a **React (Vite) single-page app served by a small FastAPI backend**. The backend orchestrates the heavy SAM3D/Cadrille reconstruction pipeline on the remote GPU server (`RXL`) over SSH, tracks jobs, and serves results back to the browser.

There is no separate frontend server: `npm run build` compiles the React app to `gui/frontend/dist`, and the FastAPI app mounts that bundle at `/`, so the API and the UI are served together from **one port (18000)**.

## What it supports

- Run a single end-to-end reconstruction from an image + mask through SAM3D and Cadrille (PC or IMG mode).
- Track background jobs by remote PID + on-disk status file, with a live stage stepper and streamed logs.
- A body-cleanup review banner and a visible cleanup stage that removes hallucinated bodies after Cadrille selection.
- Cancel a running job (process-group + Docker container teardown).
- Interactive browser viewers for the result mesh and point cloud (three.js / react-three-fiber), plus metric cards and downloads.
- Bilingual UI (English / 中文).

## Architecture

```text
gui/
├── backend/
│   ├── __init__.py
│   ├── app.py                   # FastAPI: orchestrates SSH jobs AND serves the built SPA at /
│   └── simple_reconstruct_job.py
├── frontend/                    # React + TypeScript + Vite SPA
│   ├── src/                     # views/ (Configure, Live, Result), components/, api/ client, i18n/
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt             # backend Python deps
└── README.md
```

The backend exposes `/health`, `/catalog`, `/jobs`, `/preview`, and `/outputs`. The frontend calls them with relative URLs (`base: ""` in `vite.config.ts`), so the same build works in production and through an SSH tunnel.

## Backend setup

```bash
cd /path/to/AIWS
python3 -m venv .venv
source .venv/bin/activate
pip install -r gui/requirements.txt
```

## Build the frontend (required before the backend can serve the UI)

```bash
cd gui/frontend
npm install        # first time only
npm run build      # writes gui/frontend/dist that the backend serves at /
```

## Run

```bash
uvicorn gui.backend.app:app --host 127.0.0.1 --port 18000
```

Then open `http://127.0.0.1:18000`.

## Local frontend development (hot reload)

To iterate on the UI without rebuilding each time, run the Vite dev server next to a running backend:

```bash
cd gui/frontend
npm run dev        # Vite dev server on port 5173
```

It proxies `/health`, `/catalog`, `/jobs`, `/preview`, and `/outputs` to the backend at `http://127.0.0.1:18000` (override with `VITE_BACKEND_URL`).

## Execution model and assumptions

- The heavy SAM3D/Cadrille pipeline runs on the remote server `RXL`, so the backend shells out over SSH. `ssh RXL` must work from the machine running the backend.
- Remote project root is `/ssd1/rxl/zhankaiming/AIWS`.
- Reconstruction jobs run inside the `cadrille:latest` Docker image on RXL. The backend launches each job detached and tracks it from its on-disk `status.json`, so jobs survive a backend restart.
- To reach a backend running on RXL from your local machine, forward the port over SSH (`ssh -N -L 18000:127.0.0.1:18000 RXL`) and open `http://127.0.0.1:18000`.
