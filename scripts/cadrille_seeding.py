"""Deterministic seeding for cadrille point-cloud sampling.

Kept standalone (numpy/random only; torch optional) so it imports without the heavy
cadrille_infer_wrapper dependencies and can be unit-tested in CI. Upstream
repos/cadrille/dataset.py is left pristine: we seed the global RNG *before* each
__getitem__ so the unseeded trimesh.sample.sample_surface call inside it becomes
reproducible.
"""
from __future__ import annotations

import random

import numpy as np

# np.random.seed accepts a uint32; keep base_seed+index in range.
_SEED_MODULUS = 2**31 - 1


def seed_all(seed: int) -> None:
    """Seed numpy, the stdlib random module, and (if importable) torch."""
    s = int(seed) % _SEED_MODULUS
    np.random.seed(s)
    random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except Exception:
        # torch is absent in lightweight/CI environments; numpy is what trimesh uses.
        pass


class SeededDataset:
    """Wrap a map-style dataset so item ``index`` always samples from ``base_seed + index``.

    Duck-typed (no torch.utils.data.Dataset base) so it imports without torch. Because the
    seed is keyed to the index, each replica in ConcatDataset([dataset] * n_samples) gets a
    distinct, fixed seed: n_samples distinct, reproducible point clouds, independent of
    num_workers, batch size, or worker scheduling.
    """

    def __init__(self, base, seed: int):
        self.base = base
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        seed_all(self.seed + index)
        return self.base[index]
