#!/usr/bin/env python3
"""Convert CadQuery programs to STL meshes (and optional B-Rep files)."""

import argparse
import traceback
from multiprocessing import Process
from pathlib import Path
from typing import Optional

import cadquery as cq
import trimesh
from tqdm import tqdm


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    pts = [(v.x, v.y, v.z) for v in vertices]
    return trimesh.Trimesh(pts, faces)


def convert_py_file(py_path: Path, mesh_path: Path, brep_path: Optional[Path], export_brep: bool):
    """Convert a single CadQuery file to STL (and optionally B-Rep)."""
    try:
        namespace = {}
        exec(py_path.read_text(), namespace)
        if "r" not in namespace:
            raise ValueError("CadQuery script must define variable `r` with a CadQuery object")
        compound = namespace["r"].val()
        mesh = compound_to_mesh(compound)
        if len(mesh.faces) <= 2:
            raise ValueError("Degenerate mesh with <= 2 faces")
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(mesh_path)
        if export_brep and brep_path is not None:
            brep_path.parent.mkdir(parents=True, exist_ok=True)
            cq.exporters.export(compound, str(brep_path))
    except Exception:
        print(f"[convert] Failed to convert {py_path}")
        traceback.print_exc()


def convert_py_file_safe(py_path: Path, mesh_path: Path, brep_path: Optional[Path], export_brep: bool, timeout: float):
    process = Process(target=convert_py_file, args=(py_path, mesh_path, brep_path, export_brep))
    process.start()
    process.join(timeout)
    if process.is_alive():
        print(f"[convert] Timeout while converting {py_path}")
        process.terminate()
        process.join()


def main():
    parser = argparse.ArgumentParser(description="Convert CadQuery .py programs into STL/B-Rep files.")
    parser.add_argument("--src", type=Path, default=Path("./work_dirs/tmp_py"),
                        help="Directory containing CadQuery .py files")
    parser.add_argument("--mesh-out", type=Path, default=Path("./work_dirs/tmp_mesh"),
                        help="Output directory for generated .stl files")
    parser.add_argument("--brep-out", type=Path, default=Path("./work_dirs/tmp_brep"),
                        help="Output directory for generated B-Rep files")
    parser.add_argument("--export-brep", action="store_true",
                        help="Enable exporting B-Rep (.step) files in addition to STL meshes")
    parser.add_argument("--brep-ext", type=str, default="step",
                        help="Extension to use for exported B-Rep files (default: step)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout (seconds) per CadQuery program execution")
    args = parser.parse_args()

    src_dir = args.src
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory {src_dir} does not exist")

    args.mesh_out.mkdir(parents=True, exist_ok=True)
    if args.export_brep:
        args.brep_out.mkdir(parents=True, exist_ok=True)

    py_files = sorted([p for p in src_dir.iterdir() if p.suffix == ".py"])
    if not py_files:
        print(f"No .py files found in {src_dir}")
        return

    for py_path in tqdm(py_files, desc="Converting"):
        mesh_path = args.mesh_out / (py_path.stem + ".stl")
        brep_path = None
        if args.export_brep:
            brep_path = args.brep_out / (py_path.stem + f".{args.brep_ext}")
        convert_py_file_safe(py_path, mesh_path, brep_path, args.export_brep, args.timeout)


if __name__ == "__main__":
    main()
