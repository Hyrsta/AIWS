"""Pure unit tests for body_cleanup_core (no CAD deps; runs on the Mac)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from body_cleanup_core import connected_components, cluster_bodies, assess_confidence

D = 1.0  # bbox diagonal in all cases below
EPS = 0.07


def _gap(n, pairs):
    """Build a symmetric n x n gap matrix; pairs is {(i,j): gap}, default far."""
    m = [[0.0 if i == j else 0.42 for j in range(n)] for i in range(n)]
    for (i, j), g in pairs.items():
        m[i][j] = m[j][i] = g
    return m


def test_connected_components_basic():
    assert connected_components(3, [(0, 1)]) == [[0, 1], [2]]
    assert connected_components(3, []) == [[0], [1], [2]]
    assert connected_components(3, [(0, 1), (1, 2)]) == [[0, 1, 2]]


def test_single_body_is_noop():
    r = cluster_bodies([1.0], [[0.0]], D, EPS)
    assert r["n_clusters"] == 1 and r["noop"] is True
    assert r["kept_body_indices"] == [0]


def test_two_touching_one_cluster_both_kept():
    r = cluster_bodies([0.6, 0.4], _gap(2, {(0, 1): 0.0}), D, EPS)
    assert r["n_clusters"] == 1 and r["noop"] is True
    assert r["kept_body_indices"] == [0, 1]


def test_far_small_speck_removed():
    r = cluster_bodies([0.999, 0.001], _gap(2, {(0, 1): 0.42}), D, EPS)
    assert r["n_clusters"] == 2 and r["noop"] is False
    assert r["kept_body_indices"] == [0]  # larger-volume cluster kept


def test_far_large_body_removed_regardless_of_size():
    # far ⇒ junk even when sizeable; the larger-volume body is kept.
    r = cluster_bodies([0.55, 0.45], _gap(2, {(0, 1): 0.42}), D, EPS)
    assert r["n_clusters"] == 2
    assert r["kept_body_indices"] == [0]
    assert r["per_body"][1]["kept"] is False


def test_three_touching_plus_floating_speck():
    vols = [0.574, 0.243, 0.183, 0.001]
    pairs = {(0, 1): 0.005, (1, 2): 0.005, (0, 2): 0.006}  # 0,1,2 touch; 3 far
    r = cluster_bodies(vols, _gap(4, pairs), D, EPS)
    assert r["kept_body_indices"] == [0, 1, 2]
    assert r["per_body"][3]["kept"] is False
    assert r["n_clusters"] == 2 and r["noop"] is False


def test_epsilon_boundary_is_inclusive():
    # gap exactly at eps*D connects (<=).
    r = cluster_bodies([0.5, 0.5], _gap(2, {(0, 1): EPS * D}), D, EPS)
    assert r["n_clusters"] == 1


def test_assess_confidence_flags_large_runnerup():
    clusters = [{"index": 0, "volume": 0.55, "bodies": [0]},
                {"index": 1, "volume": 0.45, "bodies": [1]}]
    flag, reasons = assess_confidence(clusters, 0, removed_volume_fraction=0.45)
    assert flag is True and len(reasons) >= 1


def test_assess_confidence_quiet_for_tiny_speck():
    clusters = [{"index": 0, "volume": 0.999, "bodies": [0]},
                {"index": 1, "volume": 0.001, "bodies": [1]}]
    flag, reasons = assess_confidence(clusters, 0, removed_volume_fraction=0.001)
    assert flag is False and reasons == []


# --- edge-case regression locks (behaviors that already work; guard against drift) ---

def test_empty_input_is_safe_noop():
    r = cluster_bodies([], [], D, EPS)
    assert r["n_clusters"] == 0 and r["noop"] is True
    assert r["kept_body_indices"] == [] and r["per_body"] == []


def test_zero_bbox_diagonal_does_not_crash():
    # Degenerate diagonal: threshold collapses to 0 (touching still connects);
    # min_gap_rel must fall back to inf, not divide by zero.
    r = cluster_bodies([1.0, 1.0], _gap(2, {(0, 1): 0.0}), 0.0, EPS)
    assert r["n_clusters"] == 1 and r["noop"] is True
    assert r["per_body"][0]["min_gap_rel"] == float("inf")


def test_volume_tie_breaks_to_smallest_index():
    # Two far, equal-volume bodies → deterministic: keep cluster/body index 0.
    r = cluster_bodies([0.5, 0.5], _gap(2, {(0, 1): 0.42}), D, EPS)
    assert r["n_clusters"] == 2
    assert r["kept_body_indices"] == [0]


def test_all_zero_volume_no_division_error():
    r = cluster_bodies([0.0, 0.0], _gap(2, {(0, 1): 0.0}), D, EPS)
    assert r["n_clusters"] == 1 and r["noop"] is True
    assert r["per_body"][0]["volume_fraction"] == 0.0


# ── multi-hypothesis cleanup + guard (2026-06 reference-rerank design) ──
from body_cleanup_core import (  # noqa: E402
    apply_runnerup_guard,
    cleanup_hypotheses,
    keep_set_at_threshold,
    merge_heights,
)


def test_merge_heights_triangle():
    # MST of gaps {01: 0.1, 02: 0.5, 12: 0.3} is edges 0-1 (0.1) and 1-2 (0.3).
    m = _gap(3, {(0, 1): 0.1, (0, 2): 0.5, (1, 2): 0.3})
    assert merge_heights(m) == [0.1, 0.3]
    assert merge_heights([[0.0]]) == []
    assert merge_heights([]) == []


def test_keep_set_at_threshold():
    vols = [5.0, 4.0, 1.0]
    m = _gap(3, {(0, 1): 0.1, (0, 2): 0.5, (1, 2): 0.3})
    assert keep_set_at_threshold(vols, m, -1.0) == [0]      # singletons -> largest solid
    assert keep_set_at_threshold(vols, m, 0.1) == [0, 1]    # 0-1 merged
    assert keep_set_at_threshold(vols, m, 0.3) == [0, 1, 2]  # all merged


def test_cleanup_hypotheses_distinct_and_conservative_first():
    vols = [5.0, 4.0, 1.0]
    m = _gap(3, {(0, 1): 0.1, (0, 2): 0.5, (1, 2): 0.3})
    hyps = cleanup_hypotheses(vols, m, D, epsilon_rel=EPS)
    sigs = [tuple(h["kept_body_indices"]) for h in hyps]
    assert sigs == sorted(sigs, key=lambda s: (3 - len(s), list(s)))  # fewest removed first
    assert set(sigs) == {(0, 1, 2), (0, 1), (0,)}
    assert len(sigs) == len(set(sigs))  # deduplicated
    # production eps=0.07 cut: gap 0.1 > 0.07*D, so nothing merges -> keeps [0]
    prod = [h for h in hyps if any(s.startswith("production_eps") for s in h["sources"])]
    assert prod and prod[0]["kept_body_indices"] == [0]


def test_cleanup_hypotheses_cover_every_epsilon():
    # Completeness: for ANY eps, cluster_bodies' keep-set is already a hypothesis.
    vols = [5.0, 1.0, 3.0, 0.5, 2.0]
    m = _gap(5, {(0, 1): 0.02, (1, 2): 0.09, (2, 3): 0.15, (3, 4): 0.31, (0, 4): 0.27})
    sigs = {tuple(h["kept_body_indices"]) for h in cleanup_hypotheses(vols, m, D)}
    for k in range(0, 101):
        eps = k / 100.0
        kept = tuple(cluster_bodies(vols, m, D, eps)["kept_body_indices"])
        assert kept in sigs, f"eps={eps} keep-set {kept} not enumerated"


def test_cleanup_hypotheses_empty_and_single():
    assert cleanup_hypotheses([], [], D) == []
    one = cleanup_hypotheses([2.0], [[0.0]], D)
    assert len(one) == 1 and one[0]["kept_body_indices"] == [0]


def test_runnerup_guard_keeps_comparable_cluster():
    # Two comparable far-apart bodies: the dominant amputation failure mode.
    cl = cluster_bodies([0.6, 0.4], _gap(2, {(0, 1): 0.2}), D, EPS)
    assert cl["kept_body_indices"] == [0]
    assert apply_runnerup_guard(cl, guard_ratio=0.30) == [0, 1]
    # A tiny speck stays removed.
    cl2 = cluster_bodies([0.95, 0.05], _gap(2, {(0, 1): 0.2}), D, EPS)
    assert apply_runnerup_guard(cl2, guard_ratio=0.30) == [0]
    # Guard is a no-op when everything is already kept.
    cl3 = cluster_bodies([0.6, 0.4], _gap(2, {(0, 1): 0.0}), D, EPS)
    assert apply_runnerup_guard(cl3, guard_ratio=0.30) == [0, 1]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} core tests passed.")


if __name__ == "__main__":
    _run_all()
