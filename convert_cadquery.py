#!/usr/bin/env python3
"""
Batch-convert CadQuery .py scripts to STL meshes (and optional B-Rep STEP files).

Usage:
    python convert_cadquery.py --src work_dirs/cad/ --mesh-out work_dirs/stl/
    python convert_cadquery.py --src work_dirs/cad/ --mesh-out work_dirs/stl/ --export-brep
"""

import argparse
import traceback
from multiprocessing import Process
from pathlib import Path

import cadquery as cq
import trimesh
from tqdm import tqdm


def compound_to_mesh(compound):
    """Tessellate a CadQuery compound into a trimesh object."""
    vertices, faces = compound.tessellate(0.001, 0.1)
    pts = [(v.x, v.y, v.z) for v in vertices]
    return trimesh.Trimesh(pts, faces)


def convert_single(py_path, mesh_path, brep_path=None, export_brep=False):
    """Convert one CadQuery .py file to STL (and optionally STEP)."""
    try:
        namespace = {}
        exec(py_path.read_text(), namespace)
        if "r" not in namespace:
            raise ValueError("Script must define variable 'r'")

        compound = namespace["r"].val()
        mesh = compound_to_mesh(compound)
        if len(mesh.faces) <= 2:
            raise ValueError("Degenerate mesh (<= 2 faces)")

        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(mesh_path)

        if export_brep and brep_path is not None:
            brep_path.parent.mkdir(parents=True, exist_ok=True)
            cq.exporters.export(compound, str(brep_path))

    except Exception:
        print(f"[convert] Failed: {py_path}")
        traceback.print_exc()


def convert_with_timeout(py_path, mesh_path, brep_path, export_brep, timeout):
    """Run conversion in a subprocess with a timeout."""
    proc = Process(target=convert_single,
                   args=(py_path, mesh_path, brep_path, export_brep))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        print(f"[convert] Timeout: {py_path}")
        proc.terminate()
        proc.join()


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert CadQuery scripts to STL/STEP")
    parser.add_argument("--src", type=Path, default=Path("work_dirs/cad"),
                        help="Directory with CadQuery .py files")
    parser.add_argument("--mesh-out", type=Path, default=Path("work_dirs/stl"),
                        help="Output directory for .stl files")
    parser.add_argument("--brep-out", type=Path, default=Path("work_dirs/brep"),
                        help="Output directory for B-Rep files")
    parser.add_argument("--export-brep", action="store_true",
                        help="Also export STEP files")
    parser.add_argument("--brep-ext", default="step",
                        help="B-Rep file extension (default: step)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout per script in seconds")
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Source directory {args.src} not found")

    args.mesh_out.mkdir(parents=True, exist_ok=True)
    if args.export_brep:
        args.brep_out.mkdir(parents=True, exist_ok=True)

    py_files = sorted(args.src.glob("*.py"))
    if not py_files:
        print(f"No .py files in {args.src}")
        return

    for py_path in tqdm(py_files, desc="Converting"):
        mesh_path = args.mesh_out / f"{py_path.stem}.stl"
        brep_path = args.brep_out / f"{py_path.stem}.{args.brep_ext}" if args.export_brep else None
        convert_with_timeout(py_path, mesh_path, brep_path, args.export_brep, args.timeout)


if __name__ == "__main__":
    main()
