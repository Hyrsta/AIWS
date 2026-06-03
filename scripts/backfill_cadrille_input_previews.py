#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import skimage.transform
import torch
import trimesh
from PIL import Image, ImageOps
from pytorch3d.ops import sample_farthest_points


POINT_COUNT = 256
PRE_POINT_COUNT = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill Cadrille input previews for completed historical GUI jobs."
    )
    p.add_argument("--repo-root", type=Path, default=Path("/ssd1/rxl/zhankaiming/AIWS"))
    p.add_argument("--jobs-root", type=Path, default=None)
    p.add_argument("--job", action="append", default=[], help="Specific job_id to backfill; repeatable")
    p.add_argument("--force", action="store_true", help="Rewrite existing preview artifacts")
    p.add_argument("--dry-run", action="store_true", help="Print actions without writing artifacts or tracker files")
    return p.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] unreadable JSON {path}: {exc}")
        return None


def bridge_stl_for(job_root: Path) -> Path | None:
    manifest = job_root / "bridge" / "input_manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = row.get("cadrille_stl_path")
            if raw and Path(raw).exists():
                return Path(raw)
    candidates = sorted((job_root / "bridge").glob("data/**/*.stl"))
    return candidates[0] if candidates else None


def selected_candidate(job_root: Path) -> str | None:
    for path in (
        job_root / "cadrille" / "reselect.json",
        job_root / "results" / "cadrille_reselect.json",
    ):
        if path.exists():
            data = load_json(path)
            best = data.get("best") if isinstance(data, dict) else None
            if best:
                return str(best)

    summary = job_root / "cadrille" / "pipeline_summary.json"
    data = load_json(summary) if summary.exists() else None
    if isinstance(data, dict):
        rows = data.get("selected_rows")
        if isinstance(rows, list) and rows:
            cand = rows[0].get("candidate_stem")
            if cand:
                return str(cand)
    return None


def stable_seed(job_id: str, candidate: str | None) -> int:
    token = f"{job_id}:{candidate or ''}:cadrille-input-preview-backfill"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def backfill_point_cloud(job: dict[str, Any], bridge_stl: Path, out_path: Path) -> dict[str, Any]:
    job_id = str(job["job_id"])
    candidate = selected_candidate(Path(job["output_root"]))
    seed = stable_seed(job_id, candidate)
    np.random.seed(seed)

    mesh = trimesh.load(str(bridge_stl), force="mesh")
    vertices, _faces = trimesh.sample.sample_surface(mesh, PRE_POINT_COUNT)
    _sampled, ids = sample_farthest_points(torch.tensor(vertices, dtype=torch.float32).unsqueeze(0), K=POINT_COUNT)
    ids_np = ids[0].cpu().numpy()
    points = (vertices[ids_np] - 0.5) * 2.0

    payload = {
        "mode": "pc",
        "backfilled": True,
        "provenance": "regenerated_from_saved_bridge_stl",
        "source_bridge_stl": str(bridge_stl),
        "source_candidate": candidate,
        "n_points": int(points.shape[0]),
        "n_pre_points": PRE_POINT_COUNT,
        "sampling_seed": seed,
        "points": [[float(x), float(y), float(z)] for x, y, z in points],
        "note": (
            "Historical job did not persist batch['point_clouds']; this preview reruns "
            "Cadrille's mesh_to_point_cloud path from the saved normalized bridge STL "
            "and is not guaranteed bit-for-bit identical to the original stochastic sample."
        ),
    }
    atomic_write_json(out_path, payload)
    return payload


def mesh_to_image(mesh, camera_distance=-0.9, front=(1, 1, 1), width=500, height=500, img_size=128) -> Image.Image:
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    front_array = np.array(front, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    eye = lookat + front_array * camera_distance
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(front_array, right)
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    arr = np.asarray(image)
    arr = (arr * 255).astype(np.uint8)
    arr = skimage.transform.resize(
        arr,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.uint8)
    return Image.fromarray(arr)


def backfill_render_grid(bridge_stl: Path, out_path: Path) -> None:
    import open3d as o3d

    mesh_tm = trimesh.load(str(bridge_stl), force="mesh")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_tm.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_tm.faces))
    mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
    mesh.compute_vertex_normals()

    fronts = [(1, 1, 1), (-1, -1, -1), (-1, 1, -1), (1, -1, 1)]
    images = [mesh_to_image(mesh, front=front) for front in fronts]
    images = [ImageOps.expand(image, border=3, fill="black") for image in images]
    grid = Image.fromarray(
        np.vstack((
            np.hstack((np.array(images[0]), np.array(images[1]))),
            np.hstack((np.array(images[2]), np.array(images[3]))),
        ))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def update_result_path(job_file: Path, status_file: Path, key: str, value: str, dry_run: bool) -> None:
    for path in (job_file, status_file):
        if not path.exists():
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        rp = data.setdefault("result_paths", {})
        if not isinstance(rp, dict):
            data["result_paths"] = rp = {}
        rp[key] = value
        if not dry_run:
            atomic_write_json(path, data)


def iter_job_files(jobs_root: Path, wanted: set[str]) -> list[Path]:
    files = sorted(jobs_root.glob("*.json"))
    if wanted:
        files = [p for p in files if p.stem in wanted]
    return files


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    jobs_root = (args.jobs_root or (repo_root / "outputs" / "gui-jobs")).resolve()
    wanted = set(args.job)

    changed: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []

    for job_file in iter_job_files(jobs_root, wanted):
        job = load_json(job_file)
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or job_file.stem)
        mode = (job.get("request") or {}).get("cadrille_mode")
        status = job.get("status")
        rp = job.get("result_paths") or {}
        job_root_raw = job.get("output_root") or rp.get("job_root")
        if status != "completed" or not isinstance(rp, dict) or not job_root_raw:
            skipped.append((job_id, "not completed or no result_paths"))
            continue
        job_root = Path(str(job_root_raw))
        bridge_stl = bridge_stl_for(job_root)
        if bridge_stl is None:
            skipped.append((job_id, "missing bridge STL"))
            continue
        results_root = Path(str(rp.get("results_root") or (job_root / "results")))
        status_file = job_root / "status.json"

        if mode == "pc":
            out = results_root / "cadrille_input_points.json"
            if out.exists() and not args.force:
                skipped.append((job_id, "point preview already exists"))
                continue
            print(f"[pc] {job_id}: {bridge_stl} -> {out}")
            if not args.dry_run:
                payload = backfill_point_cloud(job, bridge_stl, out)
                if payload.get("n_points") != POINT_COUNT:
                    raise RuntimeError(f"{job_id}: expected {POINT_COUNT} points, got {payload.get('n_points')}")
            update_result_path(job_file, status_file, "cadrille_input_points", str(out), args.dry_run)
            changed.append((job_id, mode, str(out)))
        elif mode == "img":
            out = results_root / "cadrille_input_render_grid.png"
            if out.exists() and not args.force:
                skipped.append((job_id, "render grid already exists"))
                continue
            print(f"[img] {job_id}: {bridge_stl} -> {out}")
            if not args.dry_run:
                backfill_render_grid(bridge_stl, out)
            update_result_path(job_file, status_file, "cadrille_input_render_grid", str(out), args.dry_run)
            changed.append((job_id, mode, str(out)))
        else:
            skipped.append((job_id, f"unsupported mode {mode!r}"))

    print(json.dumps({"changed": changed, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
