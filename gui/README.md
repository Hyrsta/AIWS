# AIWS End-to-End GUI

The GUI is a **React (Vite) single-page app served by a small FastAPI backend**. The backend runs on the GPU server (`RXL`), where it spawns the heavy SAM3D/Cadrille reconstruction pipeline as local detached subprocesses, tracks jobs, and serves results back to the browser.

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

The frontend calls the backend with relative URLs (`base: ""` in `vite.config.ts`), so the same build works served directly or through an SSH tunnel. The SPA uses `/health`, `/catalog`, and the `/jobs` family (`/jobs`, `/jobs/simple-reconstruct`, `/jobs/{id}` and its `logs`/`metrics`/`inputs`/`file`/`terminate` sub-routes). The `/preview/*`, `/outputs/*`, `/jobs/full-run`, and `/jobs/e2e` endpoints are legacy remote-dispatch helpers the SPA does not call.

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

- The backend runs **on `RXL` itself** (project root `/ssd1/rxl/zhankaiming/AIWS`). The production path (`POST /jobs/simple-reconstruct`, the only job endpoint the SPA calls) spawns the SAM3D/Cadrille pipeline as **local detached subprocesses** (`ssh_host="local"`), not over SSH. Each job is tracked from its on-disk `status.json`, so jobs survive a backend restart.
- SAM3D runs in the `sam3d-objects` conda env; only the Cadrille stage runs inside the `cadrille:latest` Docker image. Jobs are spawned with `LD_PRELOAD` unset, because a base-conda MKL preload otherwise breaks SAM3D's MoGe FFT.
- `GET /catalog` reads the workpiece-dimension catalog from `docs/workpiece-dimensions.md` in the deployment checkout, so that file must exist on the box running the backend.
- The legacy `/jobs/full-run` and `/jobs/e2e` endpoints (not used by the SPA) dispatch to a remote host over SSH; only those require `ssh <host>` to work from the backend.
- To reach the backend from your local machine, forward the port over SSH (`ssh -N -L 18000:127.0.0.1:18000 RXL`) and open `http://127.0.0.1:18000`.
