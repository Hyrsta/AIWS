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
        ops = types.ModuleType("PIL.ImageOps")
        pil.Image = img
        pil.ImageOps = ops
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = img
        sys.modules["PIL.ImageOps"] = ops

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


def test_img_result_paths_use_saved_cadrille_input_render_grid():
    with tempfile.TemporaryDirectory() as d:
        job_root = Path(d)
        glb = job_root / "m.glb"; glb.write_bytes(b"glb")
        stl = job_root / "m.stl"; stl.write_bytes(b"stl")
        cadrille_root = job_root / "cadrille"
        selected_py = cadrille_root / "selected_py" / "part.py"
        selected_py.parent.mkdir(parents=True)
        selected_py.write_text("# selected", encoding="utf-8")
        selected_render = cadrille_root / "selected_input_renders" / "part.png"
        selected_render.parent.mkdir(parents=True)
        selected_render.write_bytes(b"saved-render")

        def fail_render(src, dst):
            raise AssertionError("new IMG jobs should reuse the saved Cadrille input render")

        old_render = getattr(srj, "render_cadrille_input_grid", None)
        srj.render_cadrille_input_grid = fail_render
        try:
            rp = srj.build_simple_result_paths(
                job_root=job_root, sam3d_mesh_glb=glb, sam3d_mesh_stl=stl,
                cadrille_output_root=cadrille_root, selected_mesh=None, selected_py=str(selected_py), selected_brep=None,
                cadrille_mode="img",
            )
        finally:
            if old_render is None:
                delattr(srj, "render_cadrille_input_grid")
            else:
                srj.render_cadrille_input_grid = old_render

        out = job_root / "results" / "cadrille_input_render_grid.png"
        assert out.read_bytes() == b"saved-render"
        assert rp["cadrille_input_render_grid"] == str(out)


def test_img_result_paths_use_reselected_candidate_render_grid():
    with tempfile.TemporaryDirectory() as d:
        job_root = Path(d)
        glb = job_root / "m.glb"; glb.write_bytes(b"glb")
        stl = job_root / "m.stl"; stl.write_bytes(b"stl")
        cadrille_root = job_root / "cadrille"
        selected_py = cadrille_root / "tmp_py" / "part+3.py"
        selected_py.parent.mkdir(parents=True)
        selected_py.write_text("# reselected", encoding="utf-8")
        candidate_render = cadrille_root / "input_renders" / "part+3.png"
        candidate_render.parent.mkdir(parents=True)
        candidate_render.write_bytes(b"candidate-render")

        rp = srj.build_simple_result_paths(
            job_root=job_root, sam3d_mesh_glb=glb, sam3d_mesh_stl=stl,
            cadrille_output_root=cadrille_root, selected_mesh=None, selected_py=str(selected_py), selected_brep=None,
            cadrille_mode="img",
        )

        out = job_root / "results" / "cadrille_input_render_grid.png"
        assert out.read_bytes() == b"candidate-render"
        assert rp["cadrille_input_render_grid"] == str(out)


def test_render_grid_uses_xvfb_without_display():
    old_display = os.environ.pop("DISPLAY", None)
    old_which = srj.shutil.which
    old_run = srj.subprocess.run
    calls = []

    def fake_which(name):
        return "/usr/bin/xvfb-run" if name == "xvfb-run" else None

    def fake_run(cmd, check, timeout):
        calls.append((cmd, check, timeout))

    srj.shutil.which = fake_which
    srj.subprocess.run = fake_run
    try:
        srj.render_cadrille_input_grid(Path("/tmp/in.stl"), Path("/tmp/out.png"))
    finally:
        srj.shutil.which = old_which
        srj.subprocess.run = old_run
        if old_display is not None:
            os.environ["DISPLAY"] = old_display

    assert calls
    cmd, check, timeout = calls[0]
    assert cmd[:3] == ["xvfb-run", "-a", sys.executable]
    assert check is True
    assert timeout == 120


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} wiring tests passed.")


if __name__ == "__main__":
    _run_all()
