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


def merge_heights(gap_matrix: list[list[float]]) -> list[float]:
    """Single-linkage merge heights = MST edge weights over the complete gap
    graph (Prim, O(n^2)), sorted ascending. The components of the
    "gap <= t" graph change only at these heights, so cutting at each height
    (plus below the smallest) enumerates EVERY partition any threshold yields."""
    n = len(gap_matrix)
    if n <= 1:
        return []
    in_tree = [False] * n
    cost = [float("inf")] * n
    cost[0] = 0.0
    heights: list[float] = []
    for step in range(n):
        u = -1
        for i in range(n):
            if not in_tree[i] and (u == -1 or cost[i] < cost[u]):
                u = i
        in_tree[u] = True
        if step > 0:
            heights.append(cost[u])
        for v in range(n):
            if not in_tree[v] and gap_matrix[u][v] < cost[v]:
                cost[v] = gap_matrix[u][v]
    return sorted(heights)


def keep_set_at_threshold(volumes: list[float], gap_matrix: list[list[float]], threshold: float) -> list[int]:
    """Max-total-volume connected component of the "gap <= threshold" graph
    (ties -> smallest member index, matching cluster_bodies)."""
    n = len(volumes)
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if gap_matrix[i][j] <= threshold
    ]
    comps = connected_components(n, edges)
    if not comps:
        return []
    best = max(range(len(comps)), key=lambda c: (sum(volumes[b] for b in comps[c]), -c))
    return sorted(comps[best])


def cleanup_hypotheses(
    volumes: list[float],
    gap_matrix: list[list[float]],
    bbox_diagonal: float,
    *,
    epsilon_rel: float = 0.07,
) -> list[dict[str, Any]]:
    """Enumerate every distinct cleanup outcome any gap threshold could produce,
    plus keep-all / largest-single-solid baselines and the production
    epsilon_rel cut. Deduplicated by kept-set; ordered most-conservative first
    (fewest removed), so a downstream max() over equal scores keeps more bodies.

    Each hypothesis: {"kept_body_indices", "n_removed", "sources"}."""
    n = len(volumes)
    if n == 0:
        return []
    candidates: list[tuple[list[int], str]] = [(list(range(n)), "keep_all")]
    largest = max(range(n), key=lambda i: (volumes[i], -i))
    candidates.append(([largest], "largest_solid"))
    candidates.append((keep_set_at_threshold(volumes, gap_matrix, -1.0), "cut@singletons"))
    for h in merge_heights(gap_matrix):
        candidates.append((keep_set_at_threshold(volumes, gap_matrix, h), f"cut@{h:.6g}"))
    prod = cluster_bodies(volumes, gap_matrix, bbox_diagonal, epsilon_rel)
    candidates.append((prod["kept_body_indices"], f"production_eps{epsilon_rel:g}"))

    by_sig: dict[tuple[int, ...], dict[str, Any]] = {}
    for kept, src in candidates:
        sig = tuple(sorted(kept))
        if sig not in by_sig:
            by_sig[sig] = {"kept_body_indices": list(sig), "n_removed": n - len(sig), "sources": []}
        by_sig[sig]["sources"].append(src)
    hyps = list(by_sig.values())
    hyps.sort(key=lambda h: (h["n_removed"], h["kept_body_indices"]))
    return hyps


def apply_runnerup_guard(clustering: dict[str, Any], *, guard_ratio: float = 0.30) -> list[int]:
    """Reference-free guard: extend the kept set with every cluster whose volume
    is >= guard_ratio * the kept cluster's volume. Motivated by the 2026-06-10
    epsilon study: the dominant cleanup failure is deleting one of two
    comparable-volume true parts; never delete a comparable cluster blind."""
    clusters = clustering.get("clusters") or []
    kept = set(clustering.get("kept_body_indices") or [])
    if not clusters:
        return sorted(kept)
    kept_idx = clustering["kept_cluster_index"]
    kept_vol = clusters[kept_idx]["volume"]
    for c in clusters:
        if c["index"] == kept_idx:
            continue
        if kept_vol > 0 and c["volume"] >= guard_ratio * kept_vol:
            kept.update(c["bodies"])
    return sorted(kept)
