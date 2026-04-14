from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st


BACKEND_URL = os.environ.get("AIWS_GUI_BACKEND", "http://127.0.0.1:8000")
POLL_INTERVAL_SEC = 1.0
PIPELINE_STAGES = [
    "SAM3D: Loading checkpoints",
    "SAM3D: Generating mesh",
    "Cadrille: Preparing input",
    "Cadrille: Generating CAD result",
]
TERMINAL_STATUSES = {"completed", "failed", "terminated"}


def api_get(endpoint: str, **params: Any) -> Any:
    response = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post_multipart(
    endpoint: str,
    *,
    files: dict[str, tuple[str, bytes, str]],
    data: dict[str, str] | None = None,
) -> Any:
    response = requests.post(f"{BACKEND_URL}{endpoint}", files=files, data=data, timeout=120)
    response.raise_for_status()
    return response.json()


def _set_query_job_id(job_id: str) -> None:
    try:
        st.query_params["job_id"] = job_id
    except Exception:
        pass
    try:
        st.experimental_set_query_params(job_id=job_id)
    except Exception:
        pass


def _get_query_job_id() -> str | None:
    try:
        query_job_id = st.query_params.get("job_id")
        if query_job_id:
            if isinstance(query_job_id, list):
                return str(query_job_id[0])
            return str(query_job_id)
    except Exception:
        pass
    try:
        params = st.experimental_get_query_params()
        query_job_id = params.get("job_id")
        if query_job_id:
            if isinstance(query_job_id, list):
                return str(query_job_id[0])
            return str(query_job_id)
    except Exception:
        pass
    return None


def set_active_job_id(job_id: str) -> None:
    st.session_state["active_job_id"] = job_id
    st.session_state.pop("suppress_auto_resume", None)
    _set_query_job_id(job_id)


def get_active_job_id() -> str | None:
    query_job_id = _get_query_job_id()
    if query_job_id:
        return query_job_id
    return st.session_state.get("active_job_id")


def clear_active_job() -> None:
    st.session_state.pop("active_job_id", None)
    st.session_state["suppress_auto_resume"] = True
    try:
        st.query_params.clear()
    except Exception:
        pass
    try:
        st.experimental_set_query_params()
    except Exception:
        pass


def get_latest_simple_job() -> dict[str, Any] | None:
    try:
        jobs = api_get("/jobs")
    except Exception:
        return None
    for job in jobs:
        if job.get("kind") == "simple_reconstruct":
            return job
    return None


def stage_message(job: dict[str, Any]) -> tuple[str, str]:
    status = job.get("status")
    stage = job.get("stage") or "queued"
    if status == "completed":
        return "success", "Reconstruction finished"
    if status == "failed":
        return "error", "Reconstruction failed"
    if stage == "sam3d":
        return "info", "Generating SAM3D mesh..."
    if stage == "cadrille":
        return "info", "Generating Cadrille result..."
    return "info", "Queued..."


def stage_progress(job: dict[str, Any]) -> float:
    status = job.get("status")
    if status in TERMINAL_STATUSES:
        return 1.0
    stage_label = job.get("stage_label") or ""
    if stage_label in PIPELINE_STAGES:
        return (PIPELINE_STAGES.index(stage_label) + 0.5) / len(PIPELINE_STAGES)
    return 0.05


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration_words(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    if secs or not parts:
        parts.append(f"{secs} second" + ("s" if secs != 1 else ""))
    return " ".join(parts)


def format_output_path(path: str | None, output_root: str | None) -> str | None:
    if not path:
        return None
    if not output_root:
        return path
    try:
        return str(Path(path).resolve().relative_to(Path(output_root).resolve()))
    except Exception:
        return path


def get_cadrille_settings(job: dict[str, Any]) -> tuple[str | None, str | None]:
    request = job.get("request") or {}
    model = request.get("cadrille_checkpoint_preset")
    modality = request.get("cadrille_mode_label")
    if not modality and request.get("cadrille_mode"):
        modality = str(request.get("cadrille_mode")).upper()

    command = job.get("command") or []
    if not model and "--cadrille-checkpoint" in command:
        checkpoint_value = command[command.index("--cadrille-checkpoint") + 1]
        checkpoint_text = str(checkpoint_value).lower()
        if "sft" in checkpoint_text:
            model = "SFT"
        elif "rl" in checkpoint_text:
            model = "RL"
    if not modality and "--cadrille-mode" in command:
        modality = str(command[command.index("--cadrille-mode") + 1]).upper()

    return model, modality


def render_cadrille_settings(job: dict[str, Any]) -> None:
    model, modality = get_cadrille_settings(job)
    if not model and not modality:
        return
    parts = []
    if model:
        parts.append(f"Model: {model}")
    if modality:
        parts.append(f"Input modality: {modality}")
    st.markdown("**Cadrille settings**")
    st.caption(" | ".join(parts))


def elapsed_seconds(job: dict[str, Any]) -> float | None:
    started_at = job.get("started_at") or job.get("created_at")
    if started_at is None:
        return None
    if job.get("status") in TERMINAL_STATUSES:
        end_time = job.get("ended_at") or job.get("updated_at") or started_at
    else:
        end_time = time.time()
    return max(float(end_time) - float(started_at), 0.0)


def sync_pipeline_timings(job: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    job_id = str(job.get("job_id") or "active-job")
    tracker_store = st.session_state.setdefault("_pipeline_stage_timings", {})
    tracker = tracker_store.setdefault(job_id, {"stage_timings": {}, "current_stage_label": None})

    merged: dict[str, dict[str, float | None]] = {}
    remote_timings = job.get("stage_timings") or {}
    for label, timing in remote_timings.items():
        if isinstance(timing, dict):
            merged[label] = dict(timing)

    for label, timing in tracker.get("stage_timings", {}).items():
        entry = merged.setdefault(label, {})
        for key in ("started_at", "ended_at"):
            if entry.get(key) is None and timing.get(key) is not None:
                entry[key] = timing[key]

    now = time.time()
    current_label = job.get("stage_label")
    previous_label = tracker.get("current_stage_label")

    if current_label in PIPELINE_STAGES:
        current_entry = merged.setdefault(current_label, {})
        default_started_at = job.get("started_at") if current_label == PIPELINE_STAGES[0] else now
        current_entry.setdefault("started_at", default_started_at or now)

    if previous_label in PIPELINE_STAGES and previous_label != current_label:
        previous_entry = merged.setdefault(previous_label, {})
        previous_entry.setdefault("started_at", now)
        previous_entry.setdefault("ended_at", now)

    if job.get("status") in TERMINAL_STATUSES:
        active_label = previous_label if previous_label in PIPELINE_STAGES else current_label
        if active_label in PIPELINE_STAGES:
            active_entry = merged.setdefault(active_label, {})
            active_entry.setdefault("started_at", job.get("started_at") or now)
            active_entry.setdefault("ended_at", job.get("ended_at") or job.get("updated_at") or now)
        tracker["current_stage_label"] = None
    else:
        tracker["current_stage_label"] = current_label

    tracker["stage_timings"] = merged
    tracker_store[job_id] = tracker
    return merged


def stage_duration_seconds(job: dict[str, Any], stage_label: str, stage_timings: dict[str, dict[str, float | None]]) -> float | None:
    entry = stage_timings.get(stage_label) or {}
    started_at = entry.get("started_at")
    ended_at = entry.get("ended_at")
    if started_at is None:
        return None
    if ended_at is None:
        if job.get("status") in TERMINAL_STATUSES or stage_label != job.get("stage_label"):
            return None
        ended_at = time.time()
    return max(float(ended_at) - float(started_at), 0.0)


def _pipeline_stage_style(job: dict[str, Any], stage_label: str, stage_timings: dict[str, dict[str, float | None]]) -> tuple[str, str, str, str]:
    current_label = job.get("stage_label")
    status = job.get("status")
    entry = stage_timings.get(stage_label) or {}
    if status == "failed" and stage_label == current_label:
        return "❌", "#fff1f2", "#ef4444", "Failed"
    if entry.get("ended_at") is not None:
        return "✅", "#f0fdf4", "#22c55e", "Done"
    if stage_label == current_label and status not in TERMINAL_STATUSES:
        return "🔄", "#eff6ff", "#3b82f6", "Running"
    return "⏳", "#f8fafc", "#cbd5e1", "Waiting"


def _pipeline_stage_title(stage_label: str) -> str:
    return stage_label.replace(": ", "<br>")


def render_pipeline(job: dict[str, Any]) -> None:
    stage_timings = sync_pipeline_timings(job)
    current_label = job.get("stage_label")
    status = job.get("status")

    st.markdown("**Pipeline**")
    column_spec = []
    for index in range(len(PIPELINE_STAGES)):
        column_spec.append(4)
        if index < len(PIPELINE_STAGES) - 1:
            column_spec.append(1)
    columns = st.columns(column_spec)

    for index, stage_label in enumerate(PIPELINE_STAGES):
        stage_col = columns[index * 2]
        icon, bg_color, border_color, state_text = _pipeline_stage_style(job, stage_label, stage_timings)
        duration_seconds = stage_duration_seconds(job, stage_label, stage_timings)
        if duration_seconds is not None:
            if stage_label == current_label and status not in TERMINAL_STATUSES:
                time_text = f"Elapsed: {format_duration_words(duration_seconds)}"
            else:
                time_text = format_duration_words(duration_seconds)
        else:
            time_text = state_text

        stage_col.markdown(
            f"""
            <div style="
                text-align:center;
                padding:0.75rem 0.5rem;
                min-height:130px;
                border:1.5px solid {border_color};
                border-radius:12px;
                background:{bg_color};
            ">
                <div style="font-size:1.15rem; margin-bottom:0.25rem;">{icon}</div>
                <div style="font-weight:600; line-height:1.35; margin-bottom:0.45rem;">{_pipeline_stage_title(stage_label)}</div>
                <div style="font-size:0.82rem; color:#475569;">{time_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if index < len(PIPELINE_STAGES) - 1:
            arrow_col = columns[index * 2 + 1]
            connector = "→"
            connector_color = "#3b82f6" if stage_label == current_label and status not in TERMINAL_STATUSES else "#94a3b8"
            arrow_col.markdown(
                f"<div style='text-align:center; padding-top:3rem; font-size:1.4rem; color:{connector_color};'>{connector}</div>",
                unsafe_allow_html=True,
            )

    total_elapsed = elapsed_seconds(job)
    if total_elapsed is not None:
        total_label = "Total processing time" if status in TERMINAL_STATUSES else "Total elapsed time"
        st.caption(f"{total_label}: {format_duration_words(total_elapsed)}")


def show_mesh_preview(job: dict[str, Any], *, title: str, path: str | None, color: str) -> None:
    st.markdown(f"**{title}**")
    if not path:
        st.info("Preview unavailable: mesh file not found.")
        return
    try:
        mesh_payload = api_get("/preview/mesh", ssh_host=job.get("ssh_host", "local"), path=path, max_faces=12000)
        st.plotly_chart(
            build_mesh_figure(mesh_payload, color=color),
            config={"displaylogo": False},
            use_container_width=True,
        )
        st.caption(
            f"Preview uses a simplified display mesh: {mesh_payload.get('vertex_count')} vertices, "
            f"{mesh_payload.get('face_count')} faces "
            f"(from {mesh_payload.get('original_face_count')} original faces)."
        )
        st.code(path)
    except Exception as exc:
        st.info(f"Preview unavailable: {exc}")


def build_mesh_figure(payload: dict[str, Any], *, color: str = "#4f8bf9") -> go.Figure:
    vertices = payload.get("vertices") or []
    faces = payload.get("faces") or []
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
                color=color,
                opacity=1.0,
                flatshading=True,
                lighting={"ambient": 0.6, "diffuse": 0.8, "roughness": 0.9, "specular": 0.1},
            )
        ]
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        height=420,
    )
    return fig


def show_completed_result(job: dict[str, Any]) -> None:
    result_paths = job.get("result_paths") or {}
    output_root = result_paths.get("results_root") or result_paths.get("job_root") or job.get("output_root")
    sam3d_mesh = result_paths.get("sam3d_mesh_stl") or result_paths.get("sam3d_mesh_glb")
    cadrille_mesh = result_paths.get("selected_mesh")

    st.subheader("Results")
    st.caption("Processing is finished. The generated files have been saved to the paths below.")

    render_cadrille_settings(job)

    if output_root:
        st.markdown("**Saved output folder**")
        st.code(output_root)

    if sam3d_mesh or cadrille_mesh:
        st.markdown("**Mesh previews**")
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            show_mesh_preview(job, title="SAM3D reconstructed STL mesh", path=sam3d_mesh, color="#35b779")
        with preview_col2:
            show_mesh_preview(job, title="Cadrille selected STL mesh", path=cadrille_mesh, color="#4f8bf9")

    st.markdown("**SAM3D outputs**")
    for label, path in [
        ("SAM3D GLB", result_paths.get("sam3d_mesh_glb")),
        ("SAM3D STL", result_paths.get("sam3d_mesh_stl")),
    ]:
        display_path = format_output_path(path, output_root)
        if display_path:
            st.markdown(f"**{label}**")
            st.code(display_path)

    st.markdown("**Cadrille outputs**")
    for label, path in [
        ("Selected STEP", result_paths.get("selected_brep")),
        ("Selected STL", result_paths.get("selected_mesh")),
        ("Selected Python", result_paths.get("selected_py")),
    ]:
        display_path = format_output_path(path, output_root)
        if display_path:
            st.markdown(f"**{label}**")
            st.code(display_path)


st.set_page_config(page_title="AIWS Reconstruction GUI", page_icon="🧩", layout="centered")
st.title("AIWS Reconstruction GUI")
st.caption("Choose one photo and its mask, then start reconstruction.")

try:
    api_get("/health")
except Exception as exc:  # pragma: no cover - UI only
    st.error(f"Backend unavailable: {exc}")
    st.stop()

active_job_id = get_active_job_id()
if not active_job_id and not st.session_state.get("suppress_auto_resume"):
    latest_job = get_latest_simple_job()
    if latest_job and (time.time() - float(latest_job.get("updated_at") or latest_job.get("created_at") or 0) < 12 * 3600):
        active_job_id = latest_job["job_id"]
        st.session_state["active_job_id"] = active_job_id

if not active_job_id:
    image_file = st.file_uploader(
        "Choose the photo to reconstruct",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )
    mask_file = st.file_uploader(
        "Choose the corresponding mask",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )
    checkpoint_preset = st.radio(
        "Cadrille checkpoint",
        options=["RL", "SFT"],
        horizontal=True,
        help="Choose which Cadrille checkpoint to use for reconstruction.",
    )
    cadrille_mode = st.radio(
        "Cadrille input modality",
        options=["PC", "IMG"],
        horizontal=True,
        help="Choose which Cadrille modality to run on the SAM3D mesh input.",
    )

    if image_file and mask_file:
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.markdown("**Photo**")
            st.image(image_file.getvalue(), use_column_width=True)
        with preview_col2:
            st.markdown("**Mask**")
            st.image(mask_file.getvalue(), use_column_width=True)

    if st.button("Start reconstruction", type="primary", disabled=not (image_file and mask_file)):
        if image_file is None or mask_file is None:
            st.error("Please select both the photo and the mask again, then retry.")
        else:
            try:
                result = api_post_multipart(
                    "/jobs/simple-reconstruct",
                    data={"cadrille_checkpoint_preset": checkpoint_preset, "cadrille_mode": cadrille_mode},
                    files={
                        "image": (
                            image_file.name or "image.png",
                            image_file.getvalue(),
                            image_file.type or "application/octet-stream",
                        ),
                        "mask": (
                            mask_file.name or "mask.png",
                            mask_file.getvalue(),
                            mask_file.type or "application/octet-stream",
                        ),
                    },
                )
                set_active_job_id(result["job_id"])
                st.rerun()
            except Exception as exc:  # pragma: no cover - UI only
                st.error(exc)
else:
    try:
        message_placeholder = st.empty()
        progress_placeholder = st.empty()
        pipeline_placeholder = st.empty()
        refresh_placeholder = st.empty()

        while True:
            job = api_get(f"/jobs/{active_job_id}")
            level, message = stage_message(job)
            if level == "success":
                message_placeholder.success(message)
            elif level == "error":
                message_placeholder.error(message)
            else:
                message_placeholder.info(message)

            progress_placeholder.progress(stage_progress(job))
            with pipeline_placeholder.container():
                render_cadrille_settings(job)
                render_pipeline(job)

            if job.get("status") in TERMINAL_STATUSES:
                refresh_placeholder.empty()
                break

            refresh_placeholder.caption("Updating live...")
            time.sleep(POLL_INTERVAL_SEC)

        if job.get("status") == "completed":
            show_completed_result(job)

        if job.get("status") == "failed" and job.get("error"):
            st.error(job["error"])

        if st.button("Start another reconstruction"):
            clear_active_job()
            st.rerun()
    except Exception as exc:  # pragma: no cover - UI only
        st.error(exc)
        if st.button("Reset GUI"):
            clear_active_job()
            st.rerun()

st.divider()
st.caption("This GUI intentionally keeps all runtime settings in code and only exposes the user-facing reconstruction flow.")
