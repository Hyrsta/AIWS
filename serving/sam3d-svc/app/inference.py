import glob
import os
import subprocess


class Runner:
    def __init__(self):
        self.ready = False

    def warmup(self, settings) -> None:
        if not os.path.exists(settings.sam3d_ckpt):
            raise RuntimeError(f"sam3d checkpoint not found: {settings.sam3d_ckpt}")
        self.ready = True

    def run(self, settings, req: dict) -> dict:
        input_dir = req["input_dir"]
        mask_path = req["mask_path"]
        job_dir = os.path.dirname(input_dir)
        mesh_dir = os.path.join(job_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_path = os.path.join(mesh_dir, "sam3d_mesh.ply")
        image_path = self._first_image(input_dir)
        cmd = [
            "python", settings.sam3d_entry,
            "--input-image", image_path,
            "--input-mask", mask_path,
            "--seed", str(settings.__dict__.get("seed", 42)),
            "--out-mesh", mesh_path,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=settings.stage_timeout_s,
                           capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError("sam3d inference timed out")
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "")[-2000:]
            raise RuntimeError(f"sam3d inference failed: {tail}")
        if not os.path.exists(mesh_path):
            raise RuntimeError("sam3d produced no mesh")
        return {"mesh_path": mesh_path}

    def _first_image(self, input_dir):
        candidates = sorted(
            p for p in glob.glob(os.path.join(input_dir, "*"))
            if os.path.basename(p) != "mask.png"
        )
        if not candidates:
            raise RuntimeError("no input image found")
        return candidates[0]
