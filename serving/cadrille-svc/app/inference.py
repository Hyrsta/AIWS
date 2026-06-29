import json
import os
import subprocess


class Runner:
    """Shells out to the cadrille flow under a hard timeout.

    The hard timeout converts the known OCC materialization deadlock into a
    clean failure instead of a hung request.
    """

    def __init__(self):
        self.ready = False

    def warmup(self, settings) -> None:
        if not os.path.exists(settings.cadrille_ckpt):
            raise RuntimeError(f"cadrille checkpoint not found: {settings.cadrille_ckpt}")
        self.ready = True

    def run(self, settings, req: dict) -> dict:
        mesh_path = req["mesh_path"]
        job_dir = os.path.dirname(os.path.dirname(mesh_path))
        # CADRILLE_ENTRY is a thin CLI on the image that wraps the existing
        # cadrille_infer_wrapper + cadrille_evaluate_wrapper to emit the canonical
        # cad/, preview/, metrics.json layout. Confirm flag names against the live
        # wrappers in repos/cadrille on RXL before the integration smoke.
        cmd = [
            "python", settings.cadrille_entry,
            "--mesh", mesh_path,
            "--mode", req["mode"],
            "--n-candidates", str(req["n_candidates"]),
            "--seed", str(req["seed"]),
            "--ckpt", settings.cadrille_ckpt,
            "--device", settings.device,
            "--out-dir", job_dir,
        ]
        if req.get("cleanup", True):
            cmd.append("--cleanup")
        try:
            subprocess.run(cmd, check=True, timeout=settings.stage_timeout_s,
                           capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError("cadrille inference timed out")
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "")[-2000:]
            raise RuntimeError(f"cadrille inference failed: {tail}")

        metrics_path = os.path.join(job_dir, "metrics.json")
        metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        return {
            "cad_code_path": os.path.join(job_dir, "cad", "model.py"),
            "step_path": os.path.join(job_dir, "cad", "model.step"),
            "preview_path": os.path.join(job_dir, "preview", "model.stl"),
            "metrics": metrics,
        }
