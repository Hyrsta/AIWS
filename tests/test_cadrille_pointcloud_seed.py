"""Determinism of cadrille point-cloud sampling. Pure-python (numpy + trimesh only),
so it runs in CI without torch/cadrille/pytorch3d."""
import importlib.util
import os

import numpy as np

_MOD = os.path.join(os.path.dirname(__file__), "..", "scripts", "cadrille_seeding.py")
_spec = importlib.util.spec_from_file_location("cadrille_seeding", _MOD)
seeding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seeding)


class _StubBase:
    """Stand-in for the upstream dataset: each __getitem__ consumes the global numpy RNG,
    exactly like trimesh.sample.sample_surface does."""
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return float(np.random.rand())


def test_seeded_dataset_is_reproducible_and_distinct_per_index():
    ds = seeding.SeededDataset(_StubBase(), seed=42)
    run1 = [ds[i] for i in range(len(ds))]
    run2 = [ds[i] for i in range(len(ds))]
    assert run1 == run2, "same seed must reproduce the same per-index samples"
    assert len(set(run1)) == len(run1), "different indices must give distinct samples"


def test_seeded_dataset_changes_with_base_seed():
    a = [seeding.SeededDataset(_StubBase(), seed=42)[i] for i in range(4)]
    b = [seeding.SeededDataset(_StubBase(), seed=7)[i] for i in range(4)]
    assert a != b, "a different base seed must change the samples"


def test_trimesh_sample_surface_honors_global_numpy_seed():
    """Characterizes the dependency assumption SeededDataset relies on (trimesh 4.5.3)."""
    import trimesh
    mesh = trimesh.creation.box()
    np.random.seed(0)
    a, _ = trimesh.sample.sample_surface(mesh, 200)
    np.random.seed(0)
    b, _ = trimesh.sample.sample_surface(mesh, 200)
    assert np.allclose(a, b), "global np.random.seed must make sample_surface reproducible"
    np.random.seed(1)
    c, _ = trimesh.sample.sample_surface(mesh, 200)
    assert not np.allclose(a, c), "a different seed must change the surface samples"
