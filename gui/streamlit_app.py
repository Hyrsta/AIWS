from __future__ import annotations

import html
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


BACKEND_URL = os.environ.get("AIWS_GUI_BACKEND", "http://127.0.0.1:8000")
POLL_INTERVAL_SEC = 1.0
POSTSCALE_STAGE_LABEL = "Post-scaling: Aligning CAD to catalog (mm)"
PIPELINE_STAGES = [
    "SAM3D: Loading checkpoints",
    "SAM3D: Generating mesh",
    "Cadrille: Preparing input",
    "Cadrille: Generating CAD result",
    POSTSCALE_STAGE_LABEL,
]
TERMINAL_STATUSES = {"completed", "failed", "terminated"}

# Fallback catalog used only if the /catalog backend endpoint is unreachable.
# Normally the workpiece + model dropdowns are populated by `load_catalog()`
# which fetches docs/workpiece-dimensions.md via the backend.
WORKPIECE_CATALOG_FALLBACK: dict[str, list[str]] = {
    "cover_plate": ["G90", "G93", "G113", "G140"],
    "square_tube": ["F101", "F120", "F150"],
    "bellmouth": ["L75", "L148"],
    "h_beam": ["(default)"],
}
POSTSCALE_SKIP_OPTION = "(skip post-scaling)"


@st.cache_data(ttl=60)
def load_catalog() -> dict[str, dict[str, Any]]:
    """Fetch the catalog from the backend; fall back to the static mirror.
    Returns {class_name: {model_code: {bbox_m, bbox_mm}}} or empty entries."""
    try:
        payload = api_get("/catalog")
        out: dict[str, dict[str, Any]] = {}
        for cls_name, cls_info in (payload.get("classes") or {}).items():
            entries = cls_info.get("entries") or {}
            if cls_name == "h_beam" and "default" in entries:
                out[cls_name] = {"(default)": entries["default"]}
            else:
                out[cls_name] = {k: v for k, v in entries.items()}
        return out
    except Exception:
        return {cls: {m: {} for m in models} for cls, models in WORKPIECE_CATALOG_FALLBACK.items()}


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        .stApp {
            background:
                radial-gradient(circle at top, rgba(37, 99, 235, 0.18), transparent 28%),
                linear-gradient(180deg, #0f172a 0%, #020617 58%, #020617 100%);
            color: #e2e8f0;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2rem;
            padding-bottom: 2.75rem;
        }

        h1 {
            color: #f8fafc;
            font-size: 2.35rem;
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        h2, h3 {
            color: #f8fafc;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        p, label, [data-testid="stMarkdownContainer"] p {
            color: #cbd5e1;
        }

        .stCaption,
        [data-testid="stCaptionContainer"] {
            color: #94a3b8;
        }

        [data-testid="stFileUploader"] section {
            border: 1px dashed #334155;
            background: rgba(15, 23, 42, 0.78);
            border-radius: 18px;
        }

        [data-testid="stFileUploader"] small {
            color: #94a3b8;
        }

        [data-testid="stRadio"] {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 0.75rem 0.9rem;
        }

        .stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 14px;
            border: none;
            background: linear-gradient(135deg, #f97316, #ea580c);
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 14px 30px rgba(249, 115, 22, 0.24);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            cursor: pointer;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 34px rgba(249, 115, 22, 0.28);
        }

        .stButton > button:focus {
            outline: 2px solid #93c5fd;
            outline-offset: 2px;
        }

        .stButton > button:disabled {
            background: #334155;
            color: #94a3b8;
            box-shadow: none;
            cursor: not-allowed;
        }

        [data-testid="stCodeBlock"],
        pre {
            border-radius: 16px !important;
            border: 1px solid #1e293b;
            background: rgba(15, 23, 42, 0.92) !important;
        }

        [data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid #1e293b;
        }

        [data-testid="stPlotlyChart"] > div {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid #1e293b;
            background: rgba(2, 6, 23, 0.66);
        }

        hr {
            border-color: #1e293b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_intro_banner() -> None:
    st.markdown(
        """
        <div style="
            padding:1.4rem 1.5rem;
            border:1px solid #1e293b;
            border-radius:22px;
            background:linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.76));
            box-shadow:0 20px 45px rgba(2, 6, 23, 0.35);
            margin-bottom:1.25rem;
        ">
            <div style="font-size:0.78rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:#60a5fa; margin-bottom:0.7rem;">
                AIWS offline pipeline
            </div>
            <div style="font-size:2rem; line-height:1.12; font-weight:800; color:#f8fafc; margin-bottom:0.45rem;">
                CAD reconstruction from a photo and mask
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:0.55rem; margin-top:0.95rem;">
                <span style="padding:0.36rem 0.78rem; border-radius:999px; border:1px solid rgba(59, 130, 246, 0.28); background:rgba(37, 99, 235, 0.12); color:#bfdbfe; font-size:0.84rem; font-weight:600;">Single reconstruction flow</span>
                <span style="padding:0.36rem 0.78rem; border-radius:999px; border:1px solid rgba(34, 197, 94, 0.28); background:rgba(22, 163, 74, 0.12); color:#bbf7d0; font-size:0.84rem; font-weight:600;">Live stage tracking</span>
                <span style="padding:0.36rem 0.78rem; border-radius:999px; border:1px solid rgba(249, 115, 22, 0.28); background:rgba(249, 115, 22, 0.12); color:#fdba74; font-size:0.84rem; font-weight:600;">Immediate result preview</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: str | None = None) -> None:
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="font-size:0.96rem; color:#94a3b8; line-height:1.55; margin-top:0.18rem;">{subtitle}</div>'
    st.markdown(
        f"""
        <div style="margin:0 0 0.85rem 0;">
            <div style="font-size:1.12rem; font-weight:700; color:#f8fafc; line-height:1.3;">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_output_group(
    title: str,
    entries: list[tuple[str, str | None]],
    output_root: str | None,
    *,
    job_id: str | None = None,
) -> None:
    """Render a group of output entries as a tight stack of download buttons.

    Each button's label is the friendly name (e.g. "Scaled STEP (mm)"); the
    file's actual on-disk filename is used for the download itself, and the
    relative path is tucked into the button's hover tooltip so the location
    information isn't lost from the UI."""
    render_section_heading(title)
    rendered_any = False
    for label, path in entries:
        if not path:
            continue
        # When job_id is set we serve the file via /jobs/{id}/file and
        # expose a download button. Without job_id we can't fetch the file,
        # so just show a tiny caption with the relative path as a fallback.
        if job_id:
            file_bytes = fetch_file_bytes(job_id, path)
            if file_bytes is None:
                continue
            rendered_any = True
            st.download_button(
                label=label,
                data=file_bytes,
                file_name=Path(path).name,
                key=f"dl_{job_id}_{label}",
                help=format_output_path(path, output_root) or path,
                use_container_width=True,
            )
        else:
            display_path = format_output_path(path, output_root)
            if display_path:
                rendered_any = True
                st.caption(f"{label}: {display_path}")
    if not rendered_any:
        st.caption("No files available.")


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
    # If the user explicitly cleared the active job (clicked "Start another
    # reconstruction"), ignore both the URL leftover and any stale session
    # value until they actually start a new job (which clears the flag in
    # set_active_job_id). Otherwise a rerun triggered by file uploads would
    # re-read the old ?job_id=… from the URL and jump back to the old job.
    if st.session_state.get("suppress_auto_resume"):
        return None
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


def image_dimensions(uploaded_bytes: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(uploaded_bytes)) as img:
            return img.size  # (w, h)
    except Exception:
        return None


def fetch_log_tail(job_id: str, n_lines: int = 0) -> str:
    """Fetch the job log. `n_lines=0` (the default now) asks the backend for
    the full file so the scrollable log container retains every line. We
    used to pass `40` and then clip to the last 4000 chars in the GUI,
    which silently dropped earlier output on long-running jobs."""
    try:
        payload = api_get(f"/jobs/{job_id}/logs", tail_lines=n_lines)
        return payload.get("log") or ""
    except Exception as exc:
        return f"(log fetch failed: {exc})"


def fetch_job_metrics(job_id: str) -> dict[str, Any] | None:
    try:
        return api_get(f"/jobs/{job_id}/metrics")
    except Exception:
        return None


def fetch_postscale_metadata(job: dict[str, Any]) -> dict[str, Any] | None:
    """Load the postscale metadata.json via /jobs/{id}/file so the details
    expander can show the matrix M, axis pairing, and after-scale bbox."""
    result_paths = job.get("result_paths") or {}
    meta_path = result_paths.get("scaled_metadata")
    if not meta_path:
        return None
    try:
        response = requests.get(
            f"{BACKEND_URL}/jobs/{job['job_id']}/file",
            params={"path": meta_path},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def fetch_file_bytes(job_id: str, file_path: str) -> bytes | None:
    """Download a file from the job's output root via the backend."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/jobs/{job_id}/file",
            params={"path": file_path},
            timeout=60,
        )
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def cancel_job(job_id: str) -> bool:
    try:
        response = requests.post(f"{BACKEND_URL}/jobs/{job_id}/terminate", timeout=10)
        response.raise_for_status()
        return True
    except Exception:
        return False


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

    badges = []
    if model:
        badges.append(
            f'<span style="display:inline-block; padding:0.42rem 0.88rem; border-radius:999px; border:1px solid rgba(59, 130, 246, 0.28); background:rgba(37, 99, 235, 0.14); color:#bfdbfe; font-weight:700; font-size:0.98rem;">Model: {model}</span>'
        )
    if modality:
        badges.append(
            f'<span style="display:inline-block; padding:0.42rem 0.88rem; border-radius:999px; border:1px solid rgba(34, 197, 94, 0.28); background:rgba(22, 163, 74, 0.14); color:#bbf7d0; font-weight:700; font-size:0.98rem;">Input modality: {modality}</span>'
        )

    badge_html = "".join(badges)
    render_section_heading("Cadrille settings")
    st.markdown(
        f'<div style="display:flex; flex-wrap:wrap; gap:0.6rem; align-items:center; margin:0.1rem 0 0.8rem 0;">{badge_html}</div>',
        unsafe_allow_html=True,
    )


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
    status = job.get("status")
    request = job.get("request") or {}
    postscale_enabled = bool(request.get("postscale_enabled") or request.get("workpiece_class"))

    if current_label in PIPELINE_STAGES:
        current_entry = merged.setdefault(current_label, {})
        default_started_at = job.get("started_at") if current_label == PIPELINE_STAGES[0] else now
        current_entry.setdefault("started_at", default_started_at or now)

    if previous_label in PIPELINE_STAGES and previous_label != current_label:
        previous_entry = merged.setdefault(previous_label, {})
        previous_entry.setdefault("started_at", now)
        previous_entry.setdefault("ended_at", now)

    # Back-fill every stage earlier than the current stage as Done. Otherwise
    # a stage that was too fast to be observed as `current_stage_label` (e.g.
    # SAM3D "Loading checkpoints" finishing between polls) would stay on
    # "Waiting" even though the pipeline has clearly moved past it.
    if current_label in PIPELINE_STAGES:
        current_idx = PIPELINE_STAGES.index(current_label)
        for earlier_label in PIPELINE_STAGES[:current_idx]:
            earlier_entry = merged.setdefault(earlier_label, {})
            earlier_entry.setdefault("started_at", job.get("started_at") or now)
            earlier_entry.setdefault("ended_at", now)

    if status in TERMINAL_STATUSES:
        active_label = previous_label if previous_label in PIPELINE_STAGES else current_label
        if active_label in PIPELINE_STAGES:
            active_entry = merged.setdefault(active_label, {})
            active_entry.setdefault("started_at", job.get("started_at") or now)
            active_entry.setdefault("ended_at", job.get("ended_at") or job.get("updated_at") or now)
        # When the job has ended, mark every stage Done (skipping post-scaling
        # if the user didn't pick a workpiece — that stage is rendered as
        # "Skipped" elsewhere). Without this, any stage whose `ended_at`
        # wasn't recorded inline would stay on "Waiting" forever on the final
        # page.
        terminal_end = job.get("ended_at") or job.get("updated_at") or now
        terminal_start = job.get("started_at") or terminal_end
        for label in PIPELINE_STAGES:
            if label == POSTSCALE_STAGE_LABEL and not postscale_enabled:
                continue
            entry = merged.setdefault(label, {})
            entry.setdefault("started_at", terminal_start)
            entry.setdefault("ended_at", terminal_end)
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


def _pipeline_stage_style(job: dict[str, Any], stage_label: str, stage_timings: dict[str, dict[str, float | None]]) -> tuple[str, str, str, str, str]:
    current_label = job.get("stage_label")
    status = job.get("status")
    entry = stage_timings.get(stage_label) or {}
    has_any_timing = any((stage_timings.get(label) or {}).get("started_at") is not None for label in PIPELINE_STAGES)
    request = job.get("request") or {}
    postscale_enabled = bool(request.get("postscale_enabled") or request.get("workpiece_class"))

    # Special: when the user skipped post-scaling, mark the postscale card
    # as "Skipped" once the job terminates (so it doesn't look like a stuck
    # "Waiting" card forever).
    if stage_label == POSTSCALE_STAGE_LABEL and not postscale_enabled and status in TERMINAL_STATUSES:
        return "Skipped", "rgba(15, 23, 42, 0.55)", "#475569", "Skipped (no workpiece chosen)", "#94a3b8"

    if status == "failed" and stage_label == current_label:
        return "Failed", "rgba(127, 29, 29, 0.35)", "#ef4444", "Failed", "#fecaca"
    if entry.get("ended_at") is not None:
        return "Done", "rgba(20, 83, 45, 0.35)", "#22c55e", "Done", "#bbf7d0"
    if stage_label == current_label and status not in TERMINAL_STATUSES:
        return "Live", "rgba(30, 64, 175, 0.32)", "#3b82f6", "Running", "#bfdbfe"
    if status == "completed" and not has_any_timing:
        return "Done", "rgba(20, 83, 45, 0.35)", "#22c55e", "Done", "#bbf7d0"
    # Safety net: any stage earlier than the currently-live stage should be
    # rendered as Done even if its `ended_at` somehow wasn't back-filled
    # (sync_pipeline_timings normally handles this, but this guard prevents
    # stale "Waiting" cards in any edge case).
    if (
        stage_label in PIPELINE_STAGES
        and current_label in PIPELINE_STAGES
        and PIPELINE_STAGES.index(stage_label) < PIPELINE_STAGES.index(current_label)
    ):
        return "Done", "rgba(20, 83, 45, 0.35)", "#22c55e", "Done", "#bbf7d0"
    # When the job has ended, every stage that wasn't explicitly skipped
    # should look Done.
    if status == "completed":
        return "Done", "rgba(20, 83, 45, 0.35)", "#22c55e", "Done", "#bbf7d0"
    return "Waiting", "rgba(15, 23, 42, 0.78)", "#334155", "Waiting", "#cbd5e1"


def _pipeline_stage_title(stage_label: str) -> str:
    return stage_label.replace(": ", "<br>")


def _format_seconds_zh(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v >= 60:
        m, s = divmod(v, 60.0)
        return f"{int(m)} 分 {s:.1f} 秒"
    return f"{v:.2f} 秒"


def _format_mb(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v >= 1024:
        return f"{v / 1024.0:.2f} GB"
    return f"{v:.0f} MB"


def _format_iou(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _format_cd(value: Any) -> str:
    try:
        return f"{float(value):.4e}"
    except (TypeError, ValueError):
        return "—"


def _metric_card_html(label: str, value: str, hint: str | None = None) -> str:
    hint_html = (
        f'<div style="font-size:0.74rem; color:#94a3b8; margin-top:0.25rem;">{html.escape(hint)}</div>'
        if hint
        else ""
    )
    return (
        '<div style="background:rgba(15,23,42,0.65); border:1px solid #1e293b; '
        'border-radius:14px; padding:0.85rem 1.0rem; min-height:96px; '
        'display:flex; flex-direction:column; justify-content:center;">'
        f'<div style="font-size:0.74rem; color:#94a3b8; letter-spacing:0.02em; '
        f'text-transform:uppercase; font-weight:600;">{html.escape(label)}</div>'
        f'<div style="font-size:1.35rem; color:#f8fafc; font-weight:700; '
        f'margin-top:0.25rem;">{html.escape(value)}</div>'
        f"{hint_html}"
        "</div>"
    )


def render_metrics_panel(metrics: dict[str, Any] | None) -> None:
    """Render SAM3D + Cadrille runtime metrics under the pipeline strip.

    Shows: SAM3D 时间 + 显存 reserved 最大值; Cadrille 平均 IoU + 中位 CD +
    时间 + 显存 reserved 最大值. Each section is hidden until the backend
    has data for it (so the panel doesn't flash a row of placeholders the
    moment a job starts)."""
    if not metrics:
        return
    sam3d = (metrics or {}).get("sam3d") or {}
    cadrille = (metrics or {}).get("cadrille") or {}
    if not sam3d.get("available") and not cadrille.get("available"):
        return

    render_section_heading(
        "Runtime metrics",
        "Stage-level timing and GPU memory readings emitted by the pipeline.",
    )

    if sam3d.get("available"):
        st.markdown(
            '<div style="font-weight:700; color:#cbd5e1; margin:0.3rem 0 0.4rem 0; font-size:0.92rem;">'
            "SAM3D · Mesh generation"
            "</div>",
            unsafe_allow_html=True,
        )
        sam3d_init = sam3d.get("model_init_sec")
        sam3d_dur = sam3d.get("duration_sec")
        sam3d_reserved = sam3d.get("peak_memory_reserved_mb")
        sam3d_alloc = sam3d.get("peak_memory_allocated_mb")
        cards = []
        cards.append(_metric_card_html("时间 (推理耗时)", _format_seconds_zh(sam3d_dur)))
        cards.append(
            _metric_card_html(
                "显存 reserved 最大值",
                _format_mb(sam3d_reserved),
                hint=f"allocated 峰值 {_format_mb(sam3d_alloc)}" if sam3d_alloc is not None else None,
            )
        )
        if sam3d_init is not None:
            cards.append(_metric_card_html("模型加载耗时", _format_seconds_zh(sam3d_init)))
        st.markdown(
            f'<div style="display:grid; grid-template-columns:repeat({len(cards)}, 1fr); '
            f'gap:0.7rem; margin-bottom:1.0rem;">{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )

    if cadrille.get("available"):
        st.markdown(
            '<div style="font-weight:700; color:#cbd5e1; margin:0.3rem 0 0.4rem 0; font-size:0.92rem;">'
            "Cadrille · CAD generation"
            "</div>",
            unsafe_allow_html=True,
        )
        c_iou = cadrille.get("mean_iou")
        c_cd = cadrille.get("median_cd")
        c_dur = cadrille.get("duration_sec")
        c_reserved = cadrille.get("peak_memory_reserved_mb")
        c_alloc = cadrille.get("peak_memory_allocated_mb")
        c_n = cadrille.get("n_samples")
        device = cadrille.get("device_name")
        cards = [
            _metric_card_html(
                "平均 IoU",
                _format_iou(c_iou),
                hint=f"{c_n} candidates" if c_n else None,
            ),
            _metric_card_html(
                "中位 CD",
                _format_cd(c_cd),
                hint="Chamfer distance (lower is better)",
            ),
            _metric_card_html("时间 (生成耗时)", _format_seconds_zh(c_dur)),
            _metric_card_html(
                "显存 reserved 最大值",
                _format_mb(c_reserved),
                hint=(
                    f"allocated 峰值 {_format_mb(c_alloc)}"
                    + (f" · {device}" if device else "")
                    if c_alloc is not None
                    else (device or None)
                ),
            ),
        ]
        st.markdown(
            f'<div style="display:grid; grid-template-columns:repeat(4, 1fr); '
            f'gap:0.7rem; margin-bottom:0.6rem;">{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )


def render_log_viewer(job_id: str) -> None:
    """Render a fixed-height scrollable log container that retains the full
    job log. Replaces the old `st.expander` + `st.code(log[-4000:])` block
    which truncated to the last 4000 characters."""
    log_text = fetch_log_tail(job_id, n_lines=0)  # 0 → full file
    raw = log_text if log_text.strip() else "(log file empty so far)"
    body = html.escape(raw)
    line_count = raw.count("\n") + (0 if raw.endswith("\n") else 1)
    byte_count = len(raw.encode("utf-8"))
    size_label = f"{byte_count / 1024:.1f} KB" if byte_count >= 1024 else f"{byte_count} B"

    with st.expander(f"Live log ({line_count} lines · {size_label})", expanded=False):
        # `components.html` runs the markup inside an iframe, which lets the
        # small auto-scroll script work without leaking globals into the host
        # page. Height is fixed so the container scrolls instead of pushing
        # later sections off-screen.
        html_block = f"""
<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin:0; padding:0; height:100%; background:#0b1220; }}
  #log {{
    height:100%; overflow:auto;
    padding:14px 16px;
    font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    font-size:12px; line-height:1.5; color:#e2e8f0;
    white-space:pre-wrap; word-break:break-word;
    box-sizing:border-box;
  }}
  #log::-webkit-scrollbar {{ width:10px; }}
  #log::-webkit-scrollbar-thumb {{ background:#334155; border-radius:5px; }}
  #log::-webkit-scrollbar-track {{ background:#0b1220; }}
</style></head>
<body>
<div id="log">{body}</div>
<script>
  // Auto-scroll to the latest output on every rerun so users always see
  // the freshest lines without losing the ability to scroll back manually
  // mid-render.
  const el = document.getElementById('log');
  if (el) {{ el.scrollTop = el.scrollHeight; }}
</script>
</body></html>"""
        components.html(html_block, height=480, scrolling=False)


def render_pipeline(job: dict[str, Any]) -> None:
    stage_timings = sync_pipeline_timings(job)
    current_label = job.get("stage_label")
    status = job.get("status")

    render_section_heading("Pipeline status", "Track each stage while the current job moves from SAM3D reconstruction to Cadrille CAD selection.")
    column_spec = []
    for index in range(len(PIPELINE_STAGES)):
        column_spec.append(4)
        if index < len(PIPELINE_STAGES) - 1:
            column_spec.append(1)
    columns = st.columns(column_spec)

    for index, stage_label in enumerate(PIPELINE_STAGES):
        stage_col = columns[index * 2]
        badge_text, bg_color, border_color, state_text, badge_color = _pipeline_stage_style(job, stage_label, stage_timings)
        duration_seconds = stage_duration_seconds(job, stage_label, stage_timings)
        if duration_seconds is not None:
            if stage_label == current_label and status not in TERMINAL_STATUSES:
                time_text = f"Elapsed: {format_duration_words(duration_seconds)}"
            else:
                time_text = format_duration_words(duration_seconds)
        elif status == "completed":
            time_text = "Completed"
        else:
            time_text = state_text

        stage_col.markdown(
            f"""
            <div style="
                width:100%;
                display:flex;
                flex-direction:column;
                justify-content:space-between;
                align-items:flex-start;
                text-align:left;
                box-sizing:border-box;
                padding:0.95rem 0.9rem;
                height:168px;
                border:1.5px solid {border_color};
                border-radius:18px;
                background:{bg_color};
                box-shadow:0 10px 24px rgba(2, 6, 23, 0.18);
            ">
                <div style="display:inline-flex; align-items:center; gap:0.45rem; padding:0.24rem 0.64rem; border-radius:999px; border:1px solid {border_color}; background:rgba(2, 6, 23, 0.18); color:{badge_color}; font-size:0.76rem; font-weight:700; margin-bottom:0.85rem;">
                    <span style="width:0.48rem; height:0.48rem; border-radius:50%; background:{border_color}; display:inline-block;"></span>
                    {badge_text}
                </div>
                <div style="font-weight:700; line-height:1.35; margin-bottom:0.5rem; color:#f8fafc; min-height:58px; display:flex; align-items:center;">{_pipeline_stage_title(stage_label)}</div>
                <div style="font-size:0.84rem; color:#cbd5e1; min-height:36px; display:flex; align-items:center;">{time_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if index < len(PIPELINE_STAGES) - 1:
            arrow_col = columns[index * 2 + 1]
            connector = "→"
            connector_color = "#60a5fa" if stage_label == current_label and status not in TERMINAL_STATUSES else "#475569"
            arrow_col.markdown(
                f"<div style='text-align:center; padding-top:3.5rem; font-size:1.4rem; color:{connector_color};'>{connector}</div>",
                unsafe_allow_html=True,
            )

    total_elapsed = elapsed_seconds(job)
    if total_elapsed is not None:
        total_label = "Total processing time" if status in TERMINAL_STATUSES else "Total elapsed time"
        st.caption(f"{total_label}: {format_duration_words(total_elapsed)}")


def show_mesh_preview(
    job: dict[str, Any],
    *,
    title: str,
    path: str | None,
    color: str,
    units: str | None = None,
) -> None:
    """Render a mesh preview. `units` controls how the extents caption is shown
    (e.g. 'mm' or '(canonical units)')."""
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
        vertex_count = mesh_payload.get("vertex_count")
        face_count = mesh_payload.get("face_count")
        original_face_count = mesh_payload.get("original_face_count")
        extents = mesh_payload.get("extents")  # [xlen, ylen, zlen]
        # Build a two-line caption: extents on top, geometry counts below.
        if extents and len(extents) == 3:
            xlen, ylen, zlen = extents
            unit_str = f" {units}" if units else ""
            ext_line = f"Extents: {xlen:.3g} × {ylen:.3g} × {zlen:.3g}{unit_str}"
            st.caption(ext_line)
        if original_face_count and face_count and int(face_count) < int(original_face_count):
            st.caption(
                f"Preview uses simplified display mesh: {vertex_count} vertices, {face_count} faces "
                f"(from {original_face_count} original faces)."
            )
        else:
            st.caption(f"Preview mesh: {vertex_count} vertices, {face_count} faces.")
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
        paper_bgcolor="rgba(2, 6, 23, 0)",
        plot_bgcolor="rgba(2, 6, 23, 0)",
        scene={
            "aspectmode": "data",
            "bgcolor": "#020617",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        height=440,
    )
    return fig


def show_completed_result(job: dict[str, Any]) -> None:
    result_paths = job.get("result_paths") or {}
    output_root = result_paths.get("results_root") or result_paths.get("job_root") or job.get("output_root")
    sam3d_mesh = result_paths.get("sam3d_mesh_stl") or result_paths.get("sam3d_mesh_glb")
    cadrille_mesh = result_paths.get("selected_mesh")
    scaled_mesh = result_paths.get("scaled_mesh_stl")
    wclass = result_paths.get("workpiece_class")
    mcode = result_paths.get("model_code")

    render_section_heading("Results", "Processing finished. Review the preview meshes and exported files below.")

    render_cadrille_settings(job)

    if sam3d_mesh or cadrille_mesh or scaled_mesh:
        render_section_heading("Mesh previews")
        # If post-scaling ran, show three columns (SAM3D, Cadrille canonical, Cadrille scaled mm).
        # Otherwise keep the original 2-column layout.
        if scaled_mesh:
            preview_col1, preview_col2, preview_col3 = st.columns(3)
            with preview_col1:
                show_mesh_preview(job, title="SAM3D reconstructed STL mesh",
                                  path=sam3d_mesh, color="#35b779",
                                  units="(mesh units)")
            with preview_col2:
                show_mesh_preview(job, title="Cadrille canonical STL (before scaling)",
                                  path=cadrille_mesh, color="#4f8bf9",
                                  units="(canonical units)")
            with preview_col3:
                scale_title = (
                    f"Cadrille metric STL ({wclass}/{mcode or 'default'}, mm)"
                    if wclass else "Cadrille metric STL (mm)"
                )
                show_mesh_preview(job, title=scale_title,
                                  path=scaled_mesh, color="#f97316",
                                  units="mm")
        else:
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                show_mesh_preview(job, title="SAM3D reconstructed STL mesh",
                                  path=sam3d_mesh, color="#35b779",
                                  units="(mesh units)")
            with preview_col2:
                show_mesh_preview(job, title="Cadrille selected STL mesh",
                                  path=cadrille_mesh, color="#4f8bf9",
                                  units="(canonical units)")

    # Post-scaling transparency: show the affine M, axis pairing, det note,
    # and a catalog-target vs after-scale bbox match table.
    if scaled_mesh:
        meta = fetch_postscale_metadata(job)
        if meta:
            with st.expander("Post-scaling details (matrix, axis pairing, bbox match)", expanded=False):
                cat = meta.get("catalog") or {}
                target = cat.get("bbox_mm") or [None, None, None]
                after = meta.get("after_scale_bbox_mm") or {}
                actual = [after.get("xlen"), after.get("ylen"), after.get("zlen")]
                canonical = meta.get("canonical_bbox") or {}
                scale = meta.get("scale") or {}
                matrix = scale.get("matrix_3x3") or []
                axis_map = scale.get("axis_map") or []
                det_note = scale.get("det_note") or ""

                # Per-axis match table
                st.markdown("**Per-axis bbox: catalog target vs actual after scaling**")
                rows = ["| Axis | Target (mm) | Actual (mm) | Δ (rel.) | Match |",
                        "|---|---:|---:|---:|:---:|"]
                axes = "XYZ"
                for i in range(3):
                    t = target[i]
                    a = actual[i]
                    if t is not None and a is not None and t != 0:
                        rel = abs(a - t) / t
                        ok = "✅" if rel < 1e-3 else "❌"
                        rows.append(f"| {axes[i]} | {t:.4f} | {a:.4f} | {rel:.2e} | {ok} |")
                    else:
                        rows.append(f"| {axes[i]} | {t} | {a} | — | — |")
                st.markdown("\n".join(rows))

                # Axis pairing
                if axis_map:
                    st.markdown("**Sorted-extent axis pairing (canonical → catalog)**")
                    pair_lines = ["| Rank | Canonical axis | Catalog axis |",
                                  "|---:|:---:|:---:|"]
                    for entry in axis_map:
                        pair_lines.append(f"| {entry.get('rank')} | {entry.get('canonical_axis')} | {entry.get('catalog_axis')} |")
                    st.markdown("\n".join(pair_lines))

                # Affine M
                if matrix and len(matrix) == 3:
                    st.markdown("**3×3 affine M (rotation + per-axis scale, applied to centered canonical solid)**")
                    m_rows = ["| | X col | Y col | Z col |", "|---|---:|---:|---:|"]
                    for ridx, row in enumerate(matrix):
                        cells = "".join(f" {v:+.6f} |" for v in row)
                        m_rows.append(f"| {axes[ridx]} row |{cells}")
                    st.markdown("\n".join(m_rows))

                if det_note:
                    st.caption(f"Determinant note: {det_note}")
                if canonical:
                    cb = canonical
                    st.caption(
                        f"Canonical bbox (Cadrille code units): "
                        f"{cb.get('xlen', '?'):.3g} × {cb.get('ylen', '?'):.3g} × {cb.get('zlen', '?'):.3g}; "
                        f"center ({(cb.get('xmin', 0)+cb.get('xmax', 0))/2:.3g}, "
                        f"{(cb.get('ymin', 0)+cb.get('ymax', 0))/2:.3g}, "
                        f"{(cb.get('zmin', 0)+cb.get('zmax', 0))/2:.3g})"
                    )

    if output_root:
        render_section_heading("Saved output folder")
        st.code(output_root)

    job_id_str = str(job.get("job_id") or "")
    if scaled_mesh:
        output_col1, output_col2, output_col3 = st.columns(3)
    else:
        output_col1, output_col2 = st.columns(2)
        output_col3 = None
    with output_col1:
        render_output_group(
            "SAM3D outputs",
            [
                ("SAM3D GLB", result_paths.get("sam3d_mesh_glb")),
                ("SAM3D STL", result_paths.get("sam3d_mesh_stl")),
            ],
            output_root,
            job_id=job_id_str,
        )
    with output_col2:
        render_output_group(
            "Cadrille outputs (canonical)",
            [
                ("Selected STEP", result_paths.get("selected_brep")),
                ("Selected STL", result_paths.get("selected_mesh")),
                ("Selected Python", result_paths.get("selected_py")),
            ],
            output_root,
            job_id=job_id_str,
        )
    if output_col3 is not None:
        with output_col3:
            render_output_group(
                f"Post-scaling outputs (mm — {wclass}/{mcode or 'default'})",
                [
                    ("Scaled STEP (mm)", result_paths.get("scaled_brep_step")),
                    ("Scaled STL (mm)", result_paths.get("scaled_mesh_stl")),
                    ("Scaled Python (mm)", result_paths.get("scaled_py")),
                    ("Scaled metadata.json", result_paths.get("scaled_metadata")),
                ],
                output_root,
                job_id=job_id_str,
            )


st.set_page_config(page_title="AIWS offline pipeline CAD Reconstruction", page_icon="🧩", layout="wide")
inject_custom_styles()
render_intro_banner()

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
    input_col, preview_col = st.columns([0.95, 1.05], gap="large")

    with input_col:
        render_section_heading("Start a new reconstruction", "Upload the required inputs, then choose the Cadrille settings for this run.")
        image_file = st.file_uploader(
            "Choose the photo to reconstruct",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
        )
        mask_file = st.file_uploader(
            "Choose the corresponding mask",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
        )

        settings_col1, settings_col2 = st.columns(2)
        with settings_col1:
            checkpoint_preset = st.radio(
                "Cadrille checkpoint",
                options=["RL", "SFT"],
                horizontal=True,
                help="Choose which Cadrille checkpoint to use for reconstruction.",
            )
        with settings_col2:
            cadrille_mode = st.radio(
                "Cadrille input modality",
                options=["PC", "IMG"],
                horizontal=True,
                help="Choose which Cadrille modality to run on the SAM3D mesh input.",
            )

        # Workpiece selection drives the optional post-scaling stage.
        catalog = load_catalog()
        catalog_classes = list(catalog.keys())
        render_section_heading(
            "Workpiece for metric post-scaling",
            "Pick the catalog workpiece so the reconstructed CAD can be rewritten in millimeters. "
            f"Leave class as '{POSTSCALE_SKIP_OPTION}' to skip post-scaling.",
        )
        class_col, model_col = st.columns(2)
        with class_col:
            workpiece_class_choice = st.selectbox(
                "Workpiece class",
                options=[POSTSCALE_SKIP_OPTION] + catalog_classes,
                index=0,
            )
        with model_col:
            if workpiece_class_choice in catalog:
                model_entries = catalog[workpiece_class_choice]
                model_options = list(model_entries.keys())
                model_code_choice = st.selectbox("Model code", options=model_options, index=0)
                # Show the catalog target bbox so user can verify before submitting.
                meta = model_entries.get(model_code_choice) or {}
                if meta and "bbox_mm" in meta:
                    bx, by, bz = meta["bbox_mm"]
                    st.caption(f"Target bbox (mm): {bx:.2f} × {by:.2f} × {bz:.2f}")
            else:
                model_code_choice = None
                st.selectbox("Model code", options=["—"], index=0, disabled=True)

        postscale_active = workpiece_class_choice in catalog

        # Client-side validation: image and mask must have the same dimensions.
        validation_error: str | None = None
        if image_file and mask_file:
            img_dims = image_dimensions(image_file.getvalue())
            mask_dims = image_dimensions(mask_file.getvalue())
            if img_dims and mask_dims and img_dims != mask_dims:
                validation_error = (
                    f"Photo size {img_dims[0]}×{img_dims[1]} does not match mask "
                    f"size {mask_dims[0]}×{mask_dims[1]}. SAM3D requires matching dimensions."
                )
            if validation_error:
                st.warning(validation_error)

        button_label = (
            "Start reconstruction + metric scaling"
            if postscale_active
            else "Start reconstruction (no scaling)"
        )
        button_disabled = not (image_file and mask_file) or validation_error is not None

        if st.button(button_label, type="primary", disabled=button_disabled):
            if image_file is None or mask_file is None:
                st.error("Please select both the photo and the mask again, then retry.")
            else:
                form_data: dict[str, str] = {
                    "cadrille_checkpoint_preset": checkpoint_preset,
                    "cadrille_mode": cadrille_mode,
                }
                if postscale_active:
                    form_data["workpiece_class"] = workpiece_class_choice
                    if model_code_choice and model_code_choice not in ("(default)", "—"):
                        form_data["model_code"] = model_code_choice
                try:
                    result = api_post_multipart(
                        "/jobs/simple-reconstruct",
                        data=form_data,
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

    with preview_col:
        render_section_heading("Input preview", "Double-check the uploaded image pair before sending the job to the pipeline.")
        if image_file and mask_file:
            img_dims = image_dimensions(image_file.getvalue())
            mask_dims = image_dimensions(mask_file.getvalue())
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                lbl = f"Photo · {img_dims[0]}×{img_dims[1]}" if img_dims else "Photo"
                st.caption(lbl)
                st.image(image_file.getvalue(), use_column_width=True)
            with preview_col2:
                lbl = f"Mask · {mask_dims[0]}×{mask_dims[1]}" if mask_dims else "Mask"
                st.caption(lbl)
                st.image(mask_file.getvalue(), use_column_width=True)
        else:
            st.markdown(
                """
                <div style="
                    min-height:320px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    padding:1.5rem;
                    border:1px solid #1e293b;
                    border-radius:20px;
                    background:rgba(15, 23, 42, 0.6);
                    color:#94a3b8;
                    line-height:1.65;
                ">
                    Upload both files to see a side-by-side preview here before starting the reconstruction.
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    try:
        # Single-pass render + st.rerun() polling — this is the streamlit
        # idiom for "live" pages with control widgets. The previous version
        # used `while True + sleep` which re-registered the cancel button
        # under the same key on every iteration, tripping streamlit's
        # duplicate-key guard.
        job = api_get(f"/jobs/{active_job_id}")
        message_level, message_text = stage_message(job)

        render_section_heading(
            "Live job status",
            "The page refreshes automatically while this reconstruction is in progress.",
        )
        if message_level == "success":
            st.success(message_text)
        elif message_level == "error":
            st.error(message_text)
        else:
            st.info(message_text)

        render_cadrille_settings(job)
        render_pipeline(job)
        render_metrics_panel(fetch_job_metrics(active_job_id))

        is_running = job.get("status") not in TERMINAL_STATUSES

        # Cancel button — rendered once per script run while the job is alive.
        if is_running:
            cancel_col, _ = st.columns([1, 4])
            with cancel_col:
                if st.button(
                    "Cancel job",
                    key=f"cancel_{active_job_id}",
                    type="secondary",
                    use_container_width=True,
                ):
                    if cancel_job(active_job_id):
                        st.warning("Cancel request sent. The job will stop shortly.")
                    else:
                        st.error("Failed to send cancel request.")
                    # Re-poll immediately so the UI reflects the new state.
                    time.sleep(0.5)
                    st.rerun()

        # Full log in a fixed-height scrollable container (retains every
        # line — previously this was truncated to ~40 lines from the
        # backend and the last 4000 chars in the GUI).
        render_log_viewer(active_job_id)

        if is_running:
            st.caption("Updating live...")
            time.sleep(POLL_INTERVAL_SEC)
            st.rerun()

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
