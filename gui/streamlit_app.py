from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.environ.get("AIWS_GUI_BACKEND", "http://127.0.0.1:8000")


def api_get(endpoint: str, **params: Any) -> Any:
    response = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(endpoint: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="AIWS E2E GUI", page_icon="🧩", layout="wide")
st.title("AIWS End-to-End GUI")
st.caption("Streamlit frontend + FastAPI backend, remote-first for RXL orchestration")


try:
    health = api_get("/health")
    defaults = health["defaults"]
    st.success(f"Backend connected: {BACKEND_URL}")
except Exception as exc:  # pragma: no cover - UI only
    st.error(f"Backend unavailable: {exc}")
    st.stop()


def default_output_root(prefix: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{defaults['remote_workdir']}/outputs/{prefix}-{stamp}"


def ensure_state_default(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


ensure_state_default("full_output_root", default_output_root("cadrille-sft-gui-full"))
ensure_state_default("single_cadrille_output_root", default_output_root("cadrille-sft-gui-single"))
ensure_state_default("summary_root", default_output_root("cadrille-sft-gui-full"))


def render_job_summary(summary: dict[str, Any]) -> None:
    modalities = summary.get("modalities", {})
    cols = st.columns(2)
    for idx, mode in enumerate(("pc", "img")):
        payload = modalities.get(mode, {})
        with cols[idx]:
            st.subheader(mode.upper())
            st.write(
                {
                    "exists": payload.get("exists", False),
                    "done_shards": payload.get("done_shards", 0),
                    "total_shards": payload.get("total_shards", 0),
                }
            )
            for shard in payload.get("shards", []):
                label = f"{shard['name']} | done={shard['pipeline_summary_exists']} | tmp_py={shard['tmp_py_count']} | selected={shard['selected_py_count']}"
                with st.expander(label):
                    st.json(shard)


def render_job_table(jobs: list[dict[str, Any]]) -> None:
    rows = []
    for job in jobs:
        rows.append(
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": job["status"],
                "host": job["ssh_host"],
                "output_root": job["output_root"],
                "remote_pid": job.get("remote_pid"),
                "exit_code": job.get("exit_code"),
            }
        )
    st.dataframe(rows, use_container_width=True)


def render_mesh_preview(mesh_payload: dict[str, Any]) -> None:
    vertices = mesh_payload.get("vertices") or []
    faces = mesh_payload.get("faces") or []
    if not vertices or not faces:
        st.warning("No previewable mesh data returned")
        return

    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    z = [vertex[2] for vertex in vertices]
    i = [face[0] for face in faces]
    j = [face[1] for face in faces]
    k = [face[2] for face in faces]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color="#4F8BF9",
                opacity=0.9,
                flatshading=True,
                lighting={"ambient": 0.55, "diffuse": 0.7, "roughness": 0.8},
            )
        ]
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={"aspectmode": "data"},
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_preview_browser(*, summary: dict[str, Any], ssh_host: str, root: str, state_prefix: str) -> None:
    st.subheader("3D mesh preview")
    st.caption("V1 preview supports remote STL meshes, typically from selected_mesh or tmp_mesh")

    modalities = summary.get("modalities", {})
    mode_options = [mode for mode in ("pc", "img") if modalities.get(mode, {}).get("exists")]
    if not mode_options:
        st.info("No previewable modalities found")
        return

    mode_key = f"{state_prefix}_mode"
    shard_key = f"{state_prefix}_shard"
    folder_key = f"{state_prefix}_folder"
    dir_key = f"{state_prefix}_directory"
    files_key = f"{state_prefix}_files"
    mesh_key = f"{state_prefix}_mesh"
    file_key = f"{state_prefix}_file"

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_mode = st.selectbox("Preview mode", mode_options, key=mode_key)
    shard_options = [shard["name"] for shard in modalities.get(selected_mode, {}).get("shards", [])]
    with col2:
        selected_shard = st.selectbox("Preview shard", shard_options, key=shard_key)
    with col3:
        selected_folder = st.selectbox("Mesh folder", ["selected_mesh", "tmp_mesh"], key=folder_key)

    current_directory = f"{root.rstrip('/')}/{selected_mode}/{selected_shard}/{selected_folder}"
    if st.session_state.get(dir_key) != current_directory:
        st.session_state[dir_key] = current_directory
        st.session_state.pop(files_key, None)
        st.session_state.pop(mesh_key, None)

    st.code(current_directory)

    if st.button("Load mesh file list", key=f"{state_prefix}_load_files"):
        try:
            payload = api_get("/preview/files", ssh_host=ssh_host, directory=current_directory, pattern="*.stl")
            st.session_state[files_key] = payload["files"]
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)

    files = st.session_state.get(files_key, [])
    if not files:
        st.info("Load a file list to browse meshes in this folder")
        return

    selected_file = st.selectbox("Mesh file", files, key=file_key)
    max_faces = st.slider("Preview max faces", min_value=1000, max_value=40000, value=15000, step=1000, key=f"{state_prefix}_max_faces")

    if st.button("Render 3D preview", key=f"{state_prefix}_render_preview"):
        mesh_path = f"{current_directory.rstrip('/')}/{selected_file}"
        try:
            st.session_state[mesh_key] = api_get("/preview/mesh", ssh_host=ssh_host, path=mesh_path, max_faces=max_faces)
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)

    mesh_payload = st.session_state.get(mesh_key)
    if mesh_payload:
        meta_cols = st.columns(4)
        meta_cols[0].metric("Preview faces", mesh_payload.get("face_count", 0))
        meta_cols[1].metric("Original faces", mesh_payload.get("original_face_count", 0))
        meta_cols[2].metric("Vertices", mesh_payload.get("vertex_count", 0))
        extents = mesh_payload.get("extents") or [0, 0, 0]
        meta_cols[3].metric("Extent max", round(max(extents), 4) if extents else 0)
        st.caption(mesh_payload.get("path", ""))
        render_mesh_preview(mesh_payload)


tab_full, tab_single, tab_jobs, tab_outputs = st.tabs(
    ["Launch full run", "Launch single e2e", "Jobs", "Outputs"]
)

with tab_full:
    st.subheader("Launch 4-GPU full run")
    st.caption("Default preset targets the current SFT baseline. Switch checkpoint preset to ckpt/cadrille_rl when launching the RL comparison run.")
    with st.form("full_run_form"):
        col1, col2 = st.columns(2)
        with col1:
            ssh_host = st.text_input("SSH host", defaults["ssh_host"])
            remote_workdir = st.text_input("Remote workdir", defaults["remote_workdir"])
            remote_python = st.text_input("Remote python", defaults["remote_python"])
            sam3d_output_root = st.text_input("SAM3D output root", defaults["sam3d_output_root"])
            output_root = st.text_input("Output root", key="full_output_root")
            split_prefix = st.text_input("Split prefix", "sam3d_bridge_sft_gui")
            modalities = st.text_input("Modalities", "pc,img")
            checkpoint_preset = st.selectbox("Checkpoint preset", ["ckpt/cadrille_sft", "ckpt/cadrille_rl"], index=0)
            custom_checkpoint = st.text_input("Custom checkpoint path (optional)", "")
            processor_path = st.text_input("Processor path", defaults["cadrille_processor_path"])
        with col2:
            gpus = st.text_input("GPU list", "0,1,2,3")
            pc_n_samples = st.number_input("PC n_samples", min_value=1, value=5)
            img_n_samples = st.number_input("IMG n_samples", min_value=1, value=1)
            batch_size = st.number_input("Cadrille batch size", min_value=1, value=64)
            selection_mode = st.selectbox("Selection mode", ["evaluate", "index"])
            cadrille_runtime = st.selectbox("Cadrille runtime", ["docker", "auto", "host"])
            docker_image = st.text_input("Cadrille docker image", defaults["cadrille_docker_image"])
            docker_extra_args = st.text_input("Docker extra args", defaults["cadrille_docker_extra_args"])
            allow_selection_fallback = st.checkbox("Allow selection fallback", value=False)
            export_brep = st.checkbox("Export BRep / STEP", value=True)
            force = st.checkbox("Force overwrite", value=False)
            dry_run = st.checkbox("Dry run", value=False)
        submitted = st.form_submit_button("Launch full run", type="primary")

    if submitted:
        payload = {
            "ssh_host": ssh_host,
            "remote_workdir": remote_workdir,
            "remote_python": remote_python,
            "sam3d_output_root": sam3d_output_root,
            "output_root": output_root,
            "split_prefix": split_prefix,
            "modalities": modalities,
            "gpus": gpus,
            "pc_n_samples": int(pc_n_samples),
            "img_n_samples": int(img_n_samples),
            "cadrille_batch_size": int(batch_size),
            "selection_mode": selection_mode,
            "allow_selection_fallback": allow_selection_fallback,
            "cadrille_runtime": cadrille_runtime,
            "cadrille_docker_image": docker_image,
            "cadrille_docker_extra_args": docker_extra_args,
            "cadrille_checkpoint": custom_checkpoint.strip() or checkpoint_preset,
            "cadrille_processor_path": processor_path,
            "export_brep": export_brep,
            "force": force,
            "dry_run": dry_run,
        }
        try:
            result = api_post("/jobs/full-run", payload)
            st.success(f"Started job: {result['job_id']}")
            st.json(result)
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)

with tab_single:
    st.subheader("Launch single e2e run")
    st.caption("Single-run launch now matches the current e2e bridge CLI instead of the older split-run wrapper interface.")
    with st.form("single_e2e_form"):
        col1, col2 = st.columns(2)
        with col1:
            ssh_host = st.text_input("SSH host ", defaults["ssh_host"], key="single_ssh_host")
            remote_workdir = st.text_input("Remote workdir ", defaults["remote_workdir"], key="single_remote_workdir")
            remote_python = st.text_input("Remote python ", defaults["remote_python"], key="single_remote_python")
            sam3d_output_root = st.text_input("SAM3D output root ", defaults["sam3d_output_root"], key="single_sam3d_output_root")
            dataset_root = st.text_input("Dataset root", defaults["dataset_root"])
            cadrille_root = st.text_input("Cadrille root", defaults["cadrille_root"])
            cadrille_output_root = st.text_input("Cadrille output root", key="single_cadrille_output_root")
            split_name = st.text_input("Bridge split name", "sam3d_bridge_sft_gui_single")
            checkpoint_preset = st.selectbox("Checkpoint preset ", ["ckpt/cadrille_sft", "ckpt/cadrille_rl"], index=0)
            custom_checkpoint = st.text_input("Custom checkpoint path (optional) ", "")
            processor_path = st.text_input("Processor path ", defaults["cadrille_processor_path"])
        with col2:
            cadrille_mode = st.selectbox("Cadrille mode", ["pc", "img"])
            cadrille_n_samples = st.number_input("Cadrille n_samples", min_value=1, value=5)
            batch_size = st.number_input("Cadrille batch size ", min_value=1, value=64)
            limit = st.number_input("Sample limit (0 = none)", min_value=0, value=0)
            selection_mode = st.selectbox("Selection mode ", ["evaluate", "index"])
            selected_candidate_index = st.number_input("Selected candidate index", min_value=0, value=0)
            cadrille_runtime = st.selectbox("Cadrille runtime ", ["docker", "auto", "host"], index=0)
            docker_gpus = st.text_input("Docker GPU selector", "device=0")
            docker_image = st.text_input("Docker image", defaults["cadrille_docker_image"], key="single_docker_image")
            docker_extra_args = st.text_input("Docker extra args ", defaults["cadrille_docker_extra_args"])
            skip_sam3d = st.checkbox("Skip SAM3D", value=True)
            allow_selection_fallback = st.checkbox("Allow selection fallback ", value=False)
            export_brep = st.checkbox("Export BRep / STEP ", value=True)
            force = st.checkbox("Force overwrite ", value=False)
            dry_run = st.checkbox("Dry run ", value=False)
        submitted = st.form_submit_button("Launch single e2e", type="primary")

    if submitted:
        payload = {
            "ssh_host": ssh_host,
            "remote_workdir": remote_workdir,
            "remote_python": remote_python,
            "sam3d_output_root": sam3d_output_root,
            "dataset_root": dataset_root,
            "cadrille_root": cadrille_root,
            "cadrille_output_root": cadrille_output_root,
            "bridge_split_name": split_name,
            "skip_sam3d": skip_sam3d,
            "cadrille_runtime": cadrille_runtime,
            "cadrille_docker_image": docker_image,
            "cadrille_docker_extra_args": docker_extra_args,
            "cadrille_docker_gpus": docker_gpus,
            "cadrille_checkpoint": custom_checkpoint.strip() or checkpoint_preset,
            "cadrille_processor_path": processor_path,
            "cadrille_mode": cadrille_mode,
            "cadrille_n_samples": int(cadrille_n_samples),
            "cadrille_batch_size": int(batch_size),
            "limit": int(limit) if limit > 0 else None,
            "selection_mode": selection_mode,
            "allow_selection_fallback": allow_selection_fallback,
            "selected_candidate_index": int(selected_candidate_index),
            "export_brep": export_brep,
            "force": force,
            "dry_run": dry_run,
        }
        try:
            result = api_post("/jobs/e2e", payload)
            st.success(f"Started job: {result['job_id']}")
            st.json(result)
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)

with tab_jobs:
    st.subheader("Jobs")
    refresh_jobs = st.button("Refresh jobs")
    if refresh_jobs or "jobs_cache" not in st.session_state:
        try:
            st.session_state["jobs_cache"] = api_get("/jobs")
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)
            st.session_state["jobs_cache"] = []

    jobs = st.session_state.get("jobs_cache", [])
    render_job_table(jobs)

    if jobs:
        job_ids = [job["job_id"] for job in jobs]
        selected_job_id = st.selectbox("Select job", job_ids)
        selected_job = next(job for job in jobs if job["job_id"] == selected_job_id)
        st.json(selected_job)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Refresh selected job"):
                selected_job = api_get(f"/jobs/{selected_job_id}")
                st.session_state["selected_job"] = selected_job
        with col2:
            if st.button("Terminate selected job"):
                try:
                    api_post(f"/jobs/{selected_job_id}/terminate", {})
                    st.warning("Termination signal sent")
                except Exception as exc:  # pragma: no cover - UI only
                    st.error(exc)

        log_lines = st.slider("Log tail lines", 20, 500, 120)
        try:
            log_payload = api_get(f"/jobs/{selected_job_id}/logs", tail_lines=log_lines)
            st.text_area("Job log tail", log_payload["log"], height=320)
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)

        if st.button("Load job output summary"):
            try:
                summary = api_get(f"/jobs/{selected_job_id}/summary")
                st.session_state["selected_summary"] = summary
            except Exception as exc:  # pragma: no cover - UI only
                st.error(exc)

        if "selected_summary" in st.session_state:
            render_job_summary(st.session_state["selected_summary"])

with tab_outputs:
    st.subheader("Browse an output root")
    with st.form("output_summary_form"):
        ssh_host = st.text_input("SSH host  ", defaults["ssh_host"], key="summary_ssh_host")
        root = st.text_input("Remote output root", key="summary_root")
        submitted = st.form_submit_button("Load output summary")

    if submitted:
        try:
            summary = api_get("/outputs/summary", ssh_host=ssh_host, root=root)
            st.session_state["manual_summary"] = summary
            st.session_state["manual_summary_ssh_host"] = ssh_host
            st.session_state["manual_summary_root"] = root
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)

    if "manual_summary" in st.session_state:
        st.json({"root": st.session_state["manual_summary"].get("root")})
        render_job_summary(st.session_state["manual_summary"])
        render_preview_browser(
            summary=st.session_state["manual_summary"],
            ssh_host=st.session_state.get("manual_summary_ssh_host", defaults["ssh_host"]),
            root=st.session_state.get("manual_summary_root", st.session_state["manual_summary"].get("root", "")),
            state_prefix="manual_preview",
        )

st.divider()
st.caption(
    "Current GUI can launch runs, monitor jobs, inspect shard outputs, and preview remote STL meshes."
)
