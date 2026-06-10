"""CadQuery integration tests for cadrille_body_cleanup. Runs in cadrille:latest:
    docker run --rm -v <repo>:/repo:ro cadrille:latest python /repo/tests/test_body_cleanup.py
Uses plain asserts + a __main__ runner so pytest is NOT required in the image.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cadquery as cq
import cadrille_body_cleanup as bc


def _export(solids, path):
    cq.exporters.export(cq.Compound.makeCompound(solids), path)


def _args(in_step, out_dir, **kw):
    base = dict(in_step=in_step, out_dir=out_dir, out_stem=None, epsilon_rel=0.07,
                export_stl=True, stl_linear_deflection=0.001, stl_angular_deflection=0.1,
                confidence_removed_vol_frac=0.05, confidence_runnerup_ratio=0.30)
    base.update(kw)
    return SimpleNamespace(**base)


def _run(solids, **kw):
    d = tempfile.mkdtemp()
    in_step = os.path.join(d, "in.step")
    _export(solids, in_step)
    return bc.run_cleanup(_args(in_step, d, **kw)), d


def test_two_near_boxes_one_cluster_both_kept():
    a = cq.Solid.makeBox(1, 1, 1)
    b = cq.Solid.makeBox(1, 1, 1).moved(cq.Location(cq.Vector(1.05, 0, 0)))  # 0.05 gap
    meta, _ = _run([a, b])
    assert meta["n_bodies_before"] == 2
    assert meta["n_bodies_after"] == 2
    assert meta["noop"] is True


def test_main_plus_far_small_speck_removed():
    main = cq.Solid.makeBox(2, 2, 2)
    speck = cq.Solid.makeBox(0.1, 0.1, 0.1).moved(cq.Location(cq.Vector(8, 0, 0)))
    meta, _ = _run([main, speck])
    assert meta["n_bodies_before"] == 2
    assert meta["n_bodies_after"] == 1
    assert meta["n_bodies_removed"] == 1
    assert meta["noop"] is False


def test_main_plus_far_large_body_guarded():
    main = cq.Solid.makeBox(2, 2, 2)                                              # vol 8
    big = cq.Solid.makeBox(1.8, 1.8, 1.8).moved(cq.Location(cq.Vector(9, 0, 0)))  # vol ~5.8, far
    # Default: the runner-up guard refuses to blind-delete a comparable-volume
    # cluster (the dominant amputation failure in the 2026-06-10 epsilon study).
    meta, _ = _run([main, big])
    assert meta["n_bodies_after"] == 2
    assert meta["selection_mode"] == "epsilon_guarded"
    assert meta["confidence_flag"] is False       # nothing removed, nothing to warn about
    # Legacy epsilon-only behavior, preserved behind guard_runnerup_ratio=0.
    meta0, _ = _run([main, big], guard_runnerup_ratio=0.0)
    assert meta0["n_bodies_after"] == 1           # far ⇒ removed despite size
    assert meta0["selection_mode"] == "epsilon"
    assert meta0["confidence_flag"] is True       # large runner-up should warn


def test_three_near_boxes_all_kept():
    a = cq.Solid.makeBox(1, 1, 1)
    b = cq.Solid.makeBox(1, 1, 1).moved(cq.Location(cq.Vector(1.05, 0, 0)))
    c = cq.Solid.makeBox(1, 1, 1).moved(cq.Location(cq.Vector(2.10, 0, 0)))
    meta, _ = _run([a, b, c])
    assert meta["n_bodies_before"] == 3
    assert meta["n_bodies_after"] == 3 and meta["noop"] is True


def test_single_box_noop_and_outputs_exist():
    meta, d = _run([cq.Solid.makeBox(1, 1, 1)])
    assert meta["noop"] is True and meta["n_bodies_after"] == 1
    assert os.path.exists(os.path.join(d, "in__cleaned.step"))
    assert os.path.exists(os.path.join(d, "in__cleaned.stl"))
    assert os.path.exists(os.path.join(d, "in__cleanup_metadata.json"))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} integration tests passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
