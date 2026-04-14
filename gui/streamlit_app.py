from __future__ import annotations

import os
import time
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


BACKEND_URL = os.environ.get("AIWS_GUI_BACKEND", "http://127.0.0.1:8000")
AUTO_REFRESH_MS = 3000


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
    stage = job.get("stage") or "queued"
    if status == "completed":
        return 1.0
    if status == "failed":
        return 1.0
    if stage == "sam3d":
        return 0.45
    if stage == "cadrille":
        return 0.8
    return 0.1


def enable_auto_refresh(interval_ms: int = AUTO_REFRESH_MS) -> None:
    components.html(
        f"""
        <script>
        window.setTimeout(function () {{
            window.parent.location.reload();
        }}, {interval_ms});
        </script>
        """,
        height=0,
    )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def elapsed_seconds(job: dict[str, Any]) -> float | None:
    created_at = job.get("created_at")
    if created_at is None:
        return None
    if job.get("status") in {"completed", "failed", "terminated"} and job.get("ended_at") is not None:
        end_time = job.get("ended_at")
    else:
        end_time = time.time()
    return max(float(end_time) - float(created_at), 0.0)


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
    output_root = result_paths.get("job_root") or job.get("output_root")
    sam3d_mesh = result_paths.get("sam3d_mesh_stl") or result_paths.get("sam3d_mesh_glb")
    cadrille_mesh = result_paths.get("selected_mesh")

    st.subheader("Results")
    st.caption("Processing is finished. The generated files have been saved to the paths below.")

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
        if path:
            st.markdown(f"**{label}**")
            st.code(path)

    st.markdown("**Cadrille outputs**")
    for label, path in [
        ("Selected STEP", result_paths.get("selected_brep")),
        ("Selected STL", result_paths.get("selected_mesh")),
        ("Selected Python", result_paths.get("selected_py")),
        ("Cadrille output folder", result_paths.get("cadrille_output_root")),
    ]:
        if path:
            st.markdown(f"**{label}**")
            st.code(path)


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
                    data={"cadrille_checkpoint_preset": checkpoint_preset},
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
        job = api_get(f"/jobs/{active_job_id}")
        level, message = stage_message(job)
        if level == "success":
            st.success(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)

        st.progress(stage_progress(job))
        stage_label = job.get("stage_label") or job.get("stage") or "Queued"
        elapsed = elapsed_seconds(job)
        if job.get("status") == "completed":
            st.caption(f"Current stage: {stage_label} | Generation ready in {format_duration(elapsed)}")
        else:
            st.caption(f"Current stage: {stage_label} | Elapsed time: {format_duration(elapsed)}")

        if job.get("status") not in {"completed", "failed", "terminated"}:
            enable_auto_refresh()
            st.caption("Refreshing automatically...")

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
