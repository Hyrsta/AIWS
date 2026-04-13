from __future__ import annotations

import os
from typing import Any

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


def set_active_job_id(job_id: str) -> None:
    st.session_state["active_job_id"] = job_id
    st.query_params["job_id"] = job_id


def get_active_job_id() -> str | None:
    query_job_id = st.query_params.get("job_id")
    if query_job_id:
        return str(query_job_id)
    return st.session_state.get("active_job_id")


def clear_active_job() -> None:
    st.session_state.pop("active_job_id", None)
    st.query_params.clear()


def stage_message(job: dict[str, Any]) -> tuple[str, str]:
    status = job.get("status")
    stage = job.get("stage") or "queued"
    if status == "completed":
        return "success", "Reconstruction finished"
    if status == "failed":
        return "error", "Reconstruction failed"
    if stage == "sam3d":
        return "info", "Processing SAM3D..."
    if stage == "cadrille":
        return "info", "Processing Cadrille..."
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


st.set_page_config(page_title="AIWS Reconstruction GUI", page_icon="🧩", layout="centered")
st.title("AIWS Reconstruction GUI")
st.caption("Choose one photo and its mask, then start reconstruction.")

try:
    api_get("/health")
except Exception as exc:  # pragma: no cover - UI only
    st.error(f"Backend unavailable: {exc}")
    st.stop()

active_job_id = get_active_job_id()

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
        try:
            result = api_post_multipart(
                "/jobs/simple-reconstruct",
                data={"cadrille_checkpoint_preset": checkpoint_preset},
                files={
                    "image": (
                        image_file.name,
                        image_file.getvalue(),
                        image_file.type or "application/octet-stream",
                    ),
                    "mask": (
                        mask_file.name,
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
        st.caption(f"Current stage: {stage_label}")

        if job.get("status") not in {"completed", "failed", "terminated"}:
            enable_auto_refresh()
            st.caption("Refreshing automatically...")

        if job.get("status") == "completed":
            result_paths = job.get("result_paths") or {}
            selected_mesh = result_paths.get("selected_mesh")
            selected_brep = result_paths.get("selected_brep")
            selected_py = result_paths.get("selected_py")
            if selected_mesh:
                st.code(selected_mesh)
            elif selected_brep:
                st.code(selected_brep)
            elif selected_py:
                st.code(selected_py)

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
