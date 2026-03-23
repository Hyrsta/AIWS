#!/usr/bin/env python3
"""
AIWS end-to-end pipeline: multi-view RGB images -> mesh (SAM3D) -> CAD model (Cadrille).

Usage:
    python pipeline.py --input-dir data/workpiece_01 --output-dir output/workpiece_01
    python pipeline.py --input-dir data/workpiece_01 --output-dir output/workpiece_01 --skip-sam3d --mesh output/workpiece_01/mesh/reconstructed.ply
"""

import argparse
import os
import sys
import time

from sam3d_wrapper import SAM3DReconstructor
from cadrille_wrapper import CadrilleConverter


def main():
    parser = argparse.ArgumentParser(
        description="AIWS: multi-view images to weld-ready CAD models")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with multi-view RGB images of the workpiece")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for all outputs (mesh, CAD, logs)")
    parser.add_argument("--mesh", default=None,
                        help="Skip SAM3D and use this mesh file directly")
    parser.add_argument("--skip-sam3d", action="store_true",
                        help="Skip mesh reconstruction (requires --mesh)")
    parser.add_argument("--skip-cadrille", action="store_true",
                        help="Stop after mesh reconstruction")
    parser.add_argument("--export-brep", action="store_true",
                        help="Also export B-Rep STEP file from CadQuery output")
    parser.add_argument("--sam3d-config", default=None,
                        help="Path to SAM3D pipeline config YAML")
    parser.add_argument("--cadrille-checkpoint", default="maksimko123/cadrille-rl",
                        help="HuggingFace checkpoint for Cadrille model")
    parser.add_argument("--cadrille-mode", default="pc",
                        choices=["pc", "img"],
                        help="Cadrille input mode: point cloud or multi-view images")
    parser.add_argument("--num-points", type=int, default=8192,
                        help="Number of points to sample from mesh for Cadrille")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (cuda or cpu)")
    args = parser.parse_args()

    if args.skip_sam3d and args.mesh is None:
        parser.error("--skip-sam3d requires --mesh")

    os.makedirs(args.output_dir, exist_ok=True)
    mesh_dir = os.path.join(args.output_dir, "mesh")
    cad_dir = os.path.join(args.output_dir, "cad")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(cad_dir, exist_ok=True)

    print("=" * 60)
    print("AIWS Pipeline")
    print("=" * 60)

    # ── Stage 1: multi-view images -> mesh ──────────────────────
    if args.skip_sam3d:
        mesh_path = args.mesh
        print(f"\n[Stage 1] Skipped. Using existing mesh: {mesh_path}")
    else:
        print(f"\n[Stage 1] Reconstructing mesh from images in {args.input_dir}")
        t0 = time.time()

        reconstructor = SAM3DReconstructor(
            config_path=args.sam3d_config,
            device=args.device,
        )
        mesh_path = reconstructor.reconstruct(
            image_dir=args.input_dir,
            output_dir=mesh_dir,
        )

        elapsed = time.time() - t0
        print(f"[Stage 1] Done in {elapsed:.1f}s. Mesh saved to {mesh_path}")

    if args.skip_cadrille:
        print(f"\nPipeline stopped after Stage 1. Mesh at {mesh_path}")
        return

    # ── Stage 2: mesh -> CAD model ──────────────────────────────
    print(f"\n[Stage 2] Converting mesh to CAD model")
    t0 = time.time()

    converter = CadrilleConverter(
        checkpoint=args.cadrille_checkpoint,
        device=args.device,
    )
    cad_output = converter.convert(
        mesh_path=mesh_path,
        output_dir=cad_dir,
        mode=args.cadrille_mode,
        num_points=args.num_points,
        export_brep=args.export_brep,
    )

    elapsed = time.time() - t0
    print(f"[Stage 2] Done in {elapsed:.1f}s.")
    print(f"  CadQuery script: {cad_output['script']}")
    print(f"  STL mesh:        {cad_output['stl']}")
    if cad_output.get("step"):
        print(f"  STEP B-Rep:      {cad_output['step']}")

    print(f"\nPipeline complete. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
