"""Pure-geometry clustering logic for CAD body cleanup.

No CadQuery dependency, so it unit-tests anywhere. The CadQuery/OCP I/O lives in
cadrille_body_cleanup.py, which feeds plain numbers into these functions.
"""
from __future__ import annotations

from typing import Any


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find connected components over n nodes. Returns components as sorted
    index lists, ordered by smallest member ascending."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for x in range(n):
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(v) for v in groups.values()), key=lambda c: c[0])


def cluster_bodies(
    volumes: list[float],
    gap_matrix: list[list[float]],
    bbox_diagonal: float,
    epsilon_rel: float,
) -> dict[str, Any]:
    """Cluster solids by proximity and choose the main cluster.

    volumes[i]       : volume of solid i (>= 0)
    gap_matrix[i][j] : exact min surface gap between solids i, j (absolute units),
                       symmetric, diagonal ignored
    bbox_diagonal    : overall bounding-box diagonal D (> 0)
    epsilon_rel      : connect i, j when gap <= epsilon_rel * D
    """
    n = len(volumes)
    total_vol = float(sum(volumes)) if volumes else 0.0

    min_gap_rel: list[float] = []
    for i in range(n):
        others = [gap_matrix[i][j] for j in range(n) if j != i]
        g = min(others) if others else float("inf")
        min_gap_rel.append((g / bbox_diagonal) if bbox_diagonal > 0 else float("inf"))

    threshold = epsilon_rel * bbox_diagonal
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if gap_matrix[i][j] <= threshold
    ]
    comps = connected_components(n, edges)

    clusters: list[dict[str, Any]] = []
    for idx, bodies in enumerate(comps):
        vol = float(sum(volumes[b] for b in bodies))
        clusters.append({
            "index": idx,
            "bodies": bodies,
            "volume": vol,
            "volume_fraction": (vol / total_vol) if total_vol > 0 else 0.0,
            "kept": False,
        })

    kept_cluster_index = 0
    if clusters:
        # largest total volume; ties broken by smallest index (deterministic)
        kept_cluster_index = max(
            range(len(clusters)),
            key=lambda c: (clusters[c]["volume"], -c),
        )
        clusters[kept_cluster_index]["kept"] = True

    kept_body_indices = sorted(clusters[kept_cluster_index]["bodies"]) if clusters else []
    kept_set = set(kept_body_indices)

    body_cluster: dict[int, int] = {}
    for cl in clusters:
        for b in cl["bodies"]:
            body_cluster[b] = cl["index"]

    per_body: list[dict[str, Any]] = []
    for i in range(n):
        per_body.append({
            "index": i,
            "volume": float(volumes[i]),
            "volume_fraction": (float(volumes[i]) / total_vol) if total_vol > 0 else 0.0,
            "cluster": body_cluster.get(i, -1),
            "kept": i in kept_set,
            "min_gap_rel": min_gap_rel[i],
        })

    return {
        "clusters": clusters,
        "kept_cluster_index": kept_cluster_index,
        "kept_body_indices": kept_body_indices,
        "per_body": per_body,
        "n_clusters": len(clusters),
        "n_bodies_after": len(kept_body_indices),
        "noop": len(clusters) <= 1,
    }


def assess_confidence(
    clusters: list[dict[str, Any]],
    kept_cluster_index: int,
    removed_volume_fraction: float,
    *,
    removed_vol_frac_thresh: float = 0.05,
    runnerup_ratio_thresh: float = 0.30,
) -> tuple[bool, list[str]]:
    """Flag low-confidence deletions for GUI review. Does not change the decision."""
    reasons: list[str] = []
    if removed_volume_fraction > removed_vol_frac_thresh:
        reasons.append(
            f"removed volume {removed_volume_fraction:.3f} > {removed_vol_frac_thresh:.3f}"
        )
    if clusters:
        kept_vol = clusters[kept_cluster_index]["volume"]
        others = [c["volume"] for i, c in enumerate(clusters) if i != kept_cluster_index]
        runnerup = max(others) if others else 0.0
        if kept_vol > 0 and runnerup / kept_vol >= runnerup_ratio_thresh:
            reasons.append(
                f"runner-up cluster {runnerup / kept_vol:.2f}x kept (>= {runnerup_ratio_thresh:.2f})"
            )
    return (len(reasons) > 0, reasons)
