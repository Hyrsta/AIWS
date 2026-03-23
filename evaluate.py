#!/usr/bin/env python3
"""
Evaluate CAD reconstruction quality by comparing predicted STL meshes
against ground truth meshes or point clouds.

Metrics:
    - IoU (Intersection over Union) via voxelization
    - Chamfer Distance (bidirectional, L2)
    - Invalidity ratio (fraction of CadQuery scripts that failed to produce a mesh)

Usage:
    python evaluate.py --pred-dir output/workpiece_01/cad --gt-path data/workpiece_01/gt.stl
    python evaluate.py --pred-dir work_dirs/stl --gt-path data/gt_meshes/
"""

import argparse
import os
import glob
import json

import numpy as np
import trimesh


def sample_points(mesh_or_path, n=10000):
    """Load a mesh and sample points from its surface."""
    if isinstance(mesh_or_path, str):
        mesh = trimesh.load(mesh_or_path, force="mesh")
    else:
        mesh = mesh_or_path
    return mesh.sample(n).astype(np.float32)


def chamfer_distance(pts_a, pts_b):
    """Compute bidirectional Chamfer distance between two point sets."""
    from scipy.spatial import cKDTree

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)

    return float(np.mean(dist_a ** 2) + np.mean(dist_b ** 2))


def voxel_iou(mesh_a, mesh_b, pitch=0.01):
    """Compute IoU between two meshes via voxelization."""
    try:
        vox_a = mesh_a.voxelized(pitch).fill()
        vox_b = mesh_b.voxelized(pitch).fill()

        # Align to same grid
        enc_a = set(map(tuple, vox_a.sparse_indices))
        enc_b = set(map(tuple, vox_b.sparse_indices))

        intersection = len(enc_a & enc_b)
        union = len(enc_a | enc_b)
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_pair(pred_path, gt_path, num_points=10000):
    """Evaluate a single prediction against ground truth."""
    try:
        pred_mesh = trimesh.load(pred_path, force="mesh")
        gt_mesh = trimesh.load(gt_path, force="mesh")
    except Exception as e:
        return {"error": str(e), "valid": False}

    pred_pts = pred_mesh.sample(num_points).astype(np.float32)
    gt_pts = gt_mesh.sample(num_points).astype(np.float32)

    cd = chamfer_distance(pred_pts, gt_pts)
    iou = voxel_iou(pred_mesh, gt_mesh)

    return {
        "chamfer_distance": cd,
        "iou": iou,
        "pred_faces": len(pred_mesh.faces),
        "gt_faces": len(gt_mesh.faces),
        "valid": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAD reconstruction quality")
    parser.add_argument("--pred-dir", required=True,
                        help="Directory with predicted .stl files")
    parser.add_argument("--gt-path", required=True,
                        help="Ground truth mesh file or directory")
    parser.add_argument("--num-points", type=int, default=10000,
                        help="Points to sample for Chamfer distance")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.stl")))
    if not pred_files:
        print(f"No .stl files in {args.pred_dir}")
        return

    # Match predictions to ground truth
    if os.path.isdir(args.gt_path):
        gt_files = {os.path.splitext(os.path.basename(f))[0]: f
                    for f in glob.glob(os.path.join(args.gt_path, "*.stl"))}
    else:
        # Single GT file, use for all predictions
        gt_files = None

    results = []
    valid_count = 0

    for pred_path in pred_files:
        stem = os.path.splitext(os.path.basename(pred_path))[0]

        if gt_files is not None:
            gt_path = gt_files.get(stem)
            if gt_path is None:
                print(f"  No GT for {stem}, skipping")
                continue
        else:
            gt_path = args.gt_path

        metrics = evaluate_pair(pred_path, gt_path, args.num_points)
        metrics["name"] = stem
        results.append(metrics)

        if metrics["valid"]:
            valid_count += 1
            print(f"  {stem}: CD={metrics['chamfer_distance']:.6f}, "
                  f"IoU={metrics['iou']:.4f}")
        else:
            print(f"  {stem}: INVALID - {metrics.get('error', 'unknown')}")

    # Summary
    total = len(results)
    invalid = total - valid_count
    valid_results = [r for r in results if r["valid"]]

    print(f"\nSummary ({valid_count}/{total} valid)")
    if valid_results:
        avg_cd = np.mean([r["chamfer_distance"] for r in valid_results])
        avg_iou = np.mean([r["iou"] for r in valid_results])
        print(f"  Mean Chamfer Distance: {avg_cd:.6f}")
        print(f"  Mean IoU:              {avg_iou:.4f}")
        print(f"  Invalidity ratio:      {invalid}/{total} ({100*invalid/total:.1f}%)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
