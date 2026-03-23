"""
Wrapper around Cadrille for mesh-to-CAD conversion.

Cadrille is a multi-modal CAD reconstruction model that takes
point clouds, images, or text and generates CadQuery programs.
This wrapper handles mesh-to-point-cloud sampling, model inference,
CadQuery execution, and STL/STEP export.

Requires: third_party/cadrille to be installed.
See setup instructions in the main README.
"""

import os
import sys
import traceback
from pathlib import Path
from multiprocessing import Process

import numpy as np

# Add Cadrille to path
CADRILLE_DIR = os.path.join(os.path.dirname(__file__), "third_party", "cadrille")
sys.path.insert(0, CADRILLE_DIR)


def _sample_points_from_mesh(mesh_path, num_points=8192):
    """Sample a point cloud from a mesh file."""
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if hasattr(mesh, "sample"):
        points = mesh.sample(num_points)
    else:
        # Fallback for point cloud files
        points = np.asarray(mesh.vertices)
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            indices = np.random.choice(len(points), num_points, replace=True)
            points = points[indices]
    return points.astype(np.float32)


def _execute_cadquery_script(script_path, stl_path, step_path=None, timeout=10.0):
    """
    Execute a CadQuery script and export the result to STL (and optionally STEP).
    Runs in a subprocess to handle timeouts and crashes.
    """
    def _run(script_path, stl_path, step_path):
        try:
            import cadquery as cq
            import trimesh

            namespace = {}
            exec(Path(script_path).read_text(), namespace)

            if "r" not in namespace:
                raise ValueError(
                    "CadQuery script must define variable 'r' with a CadQuery object"
                )

            compound = namespace["r"].val()

            # Export STL via tessellation
            vertices, faces = compound.tessellate(0.001, 0.1)
            pts = [(v.x, v.y, v.z) for v in vertices]
            mesh = trimesh.Trimesh(pts, faces)
            if len(mesh.faces) <= 2:
                raise ValueError("Degenerate mesh with <= 2 faces")

            Path(stl_path).parent.mkdir(parents=True, exist_ok=True)
            mesh.export(stl_path)

            # Export STEP if requested
            if step_path:
                Path(step_path).parent.mkdir(parents=True, exist_ok=True)
                cq.exporters.export(compound, str(step_path))

        except Exception:
            print(f"[Cadrille] Failed to execute {script_path}")
            traceback.print_exc()

    proc = Process(target=_run, args=(script_path, stl_path, step_path))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        print(f"[Cadrille] Timeout ({timeout}s) executing {script_path}")
        proc.terminate()
        proc.join()


class CadrilleConverter:
    """Convert a 3D mesh to a parametric CAD model using Cadrille."""

    def __init__(self, checkpoint="maksimko123/cadrille-rl", device="cuda"):
        """
        Args:
            checkpoint: HuggingFace model ID or local path.
            device: 'cuda' or 'cpu'.
        """
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from cadrille import Cadrille
            from transformers import AutoProcessor

            self._model = Cadrille.from_pretrained(
                self.checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            self._processor = AutoProcessor.from_pretrained(
                "ckpt/Qwen2-VL-2B-Instruct",
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
                padding_side="left",
            )
            print(f"[Cadrille] Model loaded from {self.checkpoint}")
        except ImportError:
            raise ImportError(
                "Cadrille not found. Run:\n"
                "  cd third_party/cadrille && pip install -e .\n"
                "and download checkpoints per their README."
            )

    def convert(self, mesh_path, output_dir, mode="pc",
                num_points=8192, export_brep=False):
        """
        Convert a mesh to a parametric CAD model.

        Args:
            mesh_path: path to input mesh (.ply, .stl, .obj).
            output_dir: directory for output files.
            mode: 'pc' (point cloud) or 'img' (multi-view images).
            num_points: number of points to sample from mesh.
            export_brep: if True, also export a STEP file.

        Returns:
            dict with keys: 'script' (CadQuery .py), 'stl', 'step' (if requested).
        """
        self._load_model()
        import torch
        from cadrille import collate

        os.makedirs(output_dir, exist_ok=True)
        stem = Path(mesh_path).stem

        # Sample point cloud from mesh
        print(f"[Cadrille] Sampling {num_points} points from {mesh_path}")
        points = _sample_points_from_mesh(mesh_path, num_points)

        # Prepare input batch
        batch = [{
            "point_cloud": torch.tensor(points, dtype=torch.float32),
            "stem": stem,
        }]

        # Run inference
        print(f"[Cadrille] Running inference (mode={mode})")
        with torch.no_grad():
            inputs = collate(batch, self._processor, mode=mode)
            inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            outputs = self._model.generate(**inputs, max_new_tokens=1024)

        # Decode CadQuery script
        script_text = self._processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

        # Save CadQuery script
        script_path = os.path.join(output_dir, f"{stem}.py")
        with open(script_path, "w") as f:
            f.write(script_text)
        print(f"[Cadrille] CadQuery script saved to {script_path}")

        # Execute CadQuery script to get STL/STEP
        stl_path = os.path.join(output_dir, f"{stem}.stl")
        step_path = os.path.join(output_dir, f"{stem}.step") if export_brep else None

        print(f"[Cadrille] Executing CadQuery script to generate mesh")
        _execute_cadquery_script(script_path, stl_path, step_path)

        result = {"script": script_path, "stl": stl_path}
        if step_path and os.path.exists(step_path):
            result["step"] = step_path

        return result
