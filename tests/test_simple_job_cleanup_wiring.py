"""Fast wiring checks for the body-cleanup GUI stage (no Docker, Mac-runnable).
Stubs numpy/PIL (imported by the target module at load time but not exercised
here) so this pure test runs anywhere."""
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

# --- stub heavy deps the target module imports at load but this test never uses ---
if "numpy" not in sys.modules:
    try:
        import numpy  # noqa: F401
    except Exception:
        sys.modules["numpy"] = types.ModuleType("numpy")
if "PIL" not in sys.modules:
    try:
        import PIL  # noqa: F401
    except Exception:
        pil = types.ModuleType("PIL")
        img = types.ModuleType("PIL.Image")
        pil.Image = img
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = img

MOD = os.path.join(os.path.dirname(__file__), "..", "gui", "backend", "simple_reconstruct_job.py")
spec = importlib.util.spec_from_file_location("simple_reconstruct_job", MOD)
srj = importlib.util.module_from_spec(spec)
sys.modules["simple_reconstruct_job"] = srj
spec.loader.exec_module(srj)


def test_stage_label_inserted_before_postscale():
    labels = srj.PIPELINE_STAGE_LABELS
    assert srj.BODY_CLEANUP_STAGE_LABEL in labels
    assert labels.index(srj.BODY_CLEANUP_STAGE_LABEL) == labels.index("Cadrille: Generating CAD result") + 1
    assert labels.index(srj.BODY_CLEANUP_STAGE_LABEL) < labels.index(srj.POSTSCALE_STAGE_LABEL)


def test_result_paths_expose_cleanup_keys():
    with tempfile.TemporaryDirectory() as d:
        job_root = Path(d)
        glb = job_root / "m.glb"; glb.write_bytes(b"glb")
        stl = job_root / "m.stl"; stl.write_bytes(b"stl")
        rp = srj.build_simple_result_paths(
            job_root=job_root, sam3d_mesh_glb=glb, sam3d_mesh_stl=stl,
            cadrille_output_root=job_root, selected_mesh=None, selected_py=None, selected_brep=None,
        )
        for k in ("cleaned_brep_step", "cleaned_mesh_stl", "cleanup_metadata", "n_bodies_before", "n_bodies_after"):
            assert k in rp


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} wiring tests passed.")


if __name__ == "__main__":
    _run_all()
