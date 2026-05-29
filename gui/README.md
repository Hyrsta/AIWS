# AIWS End-to-End GUI (v1)

V1 uses:
- **FastAPI** as a small orchestration backend
- **Streamlit** as the interactive frontend
- **SSH to `RXL`** as the default execution path, because the heavy SAM3D/Cadrille pipeline currently lives on the remote server

## What v1 supports

- Launch a **full multi-GPU Cadrille batch run** with `cadrille_batch.py`
- Launch a **single e2e run** with `e2e_sam3d_to_cadrille.py`
- Track background jobs by remote PID + status file
- Tail job logs from the GUI
- Summarize output roots and per-shard progress
- Preview remote **STL meshes interactively** in the Outputs tab

## Current scope

This first cut is still **orchestration-first**, but it now includes a practical
STL viewer for remote results. The current preview path focuses on:

- `selected_mesh/*.stl`
- `tmp_mesh/*.stl`

STEP preview is still a later step.

## File layout

```text
gui/
├── backend/
│   ├── __init__.py
│   └── app.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Setup

```bash
cd /Users/hyrsta/.openclaw/workspaces/welding-algorithm
python3 -m venv .venv
source .venv/bin/activate
pip install -r gui/requirements.txt
```

## Start backend

```bash
uvicorn gui.backend.app:app --reload --port 8000
```

## Start frontend

```bash
streamlit run gui/streamlit_app.py
```

If your backend is not on `http://127.0.0.1:8000`, set:

```bash
export AIWS_GUI_BACKEND=http://127.0.0.1:8000
```

## Assumptions

- `ssh RXL` works from the machine running the backend
- Remote project root is `/ssd1/rxl/zhankaiming/AIWS`
- Remote Python is `/home/rxl/anaconda3/envs/sam3d-objects/bin/python`
- Current SAM3D output root defaults to:
  `/ssd1/rxl/zhankaiming/AIWS/outputs/sam3d-aiws52-clean-mesh-stl-20260410-193527`

## Current preview implementation

- Backend lists remote mesh files over SSH
- Backend reads the selected remote STL file and converts it to a JSON mesh payload
- Streamlit renders it with `Plotly Mesh3d`

Large meshes are reduced to a configurable preview face budget before rendering.

## Next recommended step

If we want a richer viewer after this, the next good upgrades are:
- side-by-side input/output comparison
- candidate switching (`tmp_mesh` vs `selected_mesh`)
- STEP/BRep preview pathway
- a browser-side Three.js viewer for richer interaction
