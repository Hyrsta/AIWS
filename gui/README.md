# AIWS End-to-End GUI (v1)

V1 uses:
- **FastAPI** as a small orchestration backend
- **Streamlit** as the interactive frontend
- **SSH to `RXL`** as the default execution path, because the heavy SAM3D/Cadrille pipeline currently lives on the remote server

## What v1 supports

- Launch a **4-GPU full run** with `run_cadrille_full_modalities_4gpu.py`
- Launch a **single e2e run** with `run_sam3d_to_cadrille_e2e.py`
- Track background jobs by remote PID + status file
- Tail job logs from the GUI
- Summarize output roots and per-shard progress

## Current scope

This first cut is **orchestration-first**, not geometry-first.
It focuses on starting jobs, monitoring progress, and inspecting shard outputs.
A real 3D STL/STEP viewer can be added next in the Outputs tab.

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

## Next recommended step

Add **STL/STEP preview** in the Outputs tab, likely with one of:
- `pyvista` + Streamlit embedding
- `trimesh` + Plotly mesh rendering
- or a small Three.js panel if we later want a richer browser-side viewer
