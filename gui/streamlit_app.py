from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.environ.get("AIWS_GUI_BACKEND", "http://127.0.0.1:8000")


def api_get(path: str, **params: Any) -> Any:
    response = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=30)
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


tab_full, tab_single, tab_jobs, tab_outputs = st.tabs(
    ["Launch full run", "Launch single e2e", "Jobs", "Outputs"]
)

with tab_full:
    st.subheader("Launch 4-GPU full run")
    with st.form("full_run_form"):
        col1, col2 = st.columns(2)
        with col1:
            ssh_host = st.text_input("SSH host", defaults["ssh_host"])
            remote_workdir = st.text_input("Remote workdir", defaults["remote_workdir"])
            remote_python = st.text_input("Remote python", defaults["remote_python"])
            sam3d_output_root = st.text_input("SAM3D output root", defaults["sam3d_output_root"])
            output_root = st.text_input("Output root", default_output_root("cadrille-gui-full"))
            split_prefix = st.text_input("Split prefix", "sam3d_bridge_gui")
            modalities = st.text_input("Modalities", "pc,img")
        with col2:
            gpus = st.text_input("GPU list", "0,1,2,3")
            pc_n_samples = st.number_input("PC n_samples", min_value=1, value=5)
            img_n_samples = st.number_input("IMG n_samples", min_value=1, value=1)
            batch_size = st.number_input("Cadrille batch size", min_value=1, value=64)
            selection_mode = st.selectbox("Selection mode", ["evaluate", "index"])
            cadrille_runtime = st.selectbox("Cadrille runtime", ["docker", "auto", "host"])
            docker_image = st.text_input("Cadrille docker image", defaults["cadrille_docker_image"])
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
    with st.form("single_e2e_form"):
        col1, col2 = st.columns(2)
        with col1:
            ssh_host = st.text_input("SSH host ", defaults["ssh_host"], key="single_ssh_host")
            remote_workdir = st.text_input("Remote workdir ", defaults["remote_workdir"], key="single_remote_workdir")
            remote_python = st.text_input("Remote python ", defaults["remote_python"], key="single_remote_python")
            sam3d_output_root = st.text_input("SAM3D output root ", defaults["sam3d_output_root"], key="single_sam3d_output_root")
            dataset_root = st.text_input("Dataset root", defaults["dataset_root"])
            cadrille_root = st.text_input("Cadrille root", defaults["cadrille_root"])
            cadrille_output_root = st.text_input("Cadrille output root", default_output_root("cadrille-gui-single"))
            split_name = st.text_input("Cadrille split name", "sam3d_bridge_gui_single")
        with col2:
            cadrille_mode = st.selectbox("Cadrille mode", ["pc", "img"])
            cadrille_input_source = st.selectbox("Input source", ["mesh", "point_cloud", "multi_view"])
            cadrille_n_samples = st.number_input("Cadrille n_samples", min_value=1, value=5)
            batch_size = st.number_input("Cadrille batch size ", min_value=1, value=64)
            sample_offset = st.number_input("Sample offset", min_value=0, value=0)
            max_samples = st.number_input("Max samples (0 = none)", min_value=0, value=0)
            selection_mode = st.selectbox("Selection mode ", ["evaluate", "index"])
            selected_candidate_index = st.number_input("Selected candidate index", min_value=0, value=0)
            docker_gpus = st.text_input("Docker GPU selector", "device=0")
            docker_image = st.text_input("Docker image", defaults["cadrille_docker_image"], key="single_docker_image")
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
            "cadrille_split_name": split_name,
            "skip_sam3d": skip_sam3d,
            "cadrille_runtime": "docker",
            "cadrille_docker_image": docker_image,
            "cadrille_docker_gpus": docker_gpus,
            "cadrille_mode": cadrille_mode,
            "cadrille_input_source": cadrille_input_source,
            "cadrille_n_samples": int(cadrille_n_samples),
            "cadrille_batch_size": int(batch_size),
            "sample_offset": int(sample_offset),
            "max_samples": int(max_samples) if max_samples > 0 else None,
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
        root = st.text_input("Remote output root", default_output_root("cadrille-gui-full"))
        submitted = st.form_submit_button("Load output summary")

    if submitted:
        try:
            summary = api_get("/outputs/summary", ssh_host=ssh_host, root=root)
            st.session_state["manual_summary"] = summary
        except Exception as exc:  # pragma: no cover - UI only
            st.error(exc)

    if "manual_summary" in st.session_state:
        st.json({"root": st.session_state["manual_summary"].get("root")})
        render_job_summary(st.session_state["manual_summary"])

st.divider()
st.caption(
    "V1 is orchestration-first: launch runs, monitor jobs, inspect shard outputs. "
    "Next step can be STL/STEP preview in the Outputs tab."
)
