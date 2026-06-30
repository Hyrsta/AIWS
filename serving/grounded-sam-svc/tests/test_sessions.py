# serving/grounded-sam-svc/tests/test_sessions.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from app.sessions import SessionStore, SessionExpired


class FakeSeg:
    """Records calls; returns canned masks. No SAM."""
    def __init__(self):
        self.encoded = 0
        self.refines = []
    def encode_image(self, image_bytes):
        self.encoded += 1
        return {"img": image_bytes, "id": self.encoded}
    def auto_mask(self, state, prompt, box_threshold, text_threshold):
        return {"mask_png_base64": f"auto{state['id']}", "score": 0.5,
                "box": [0, 0, 1, 1], "width": 4, "height": 3, "detected": True}
    def refine_mask(self, state, prompt, points, box, reset, box_threshold, text_threshold):
        self.refines.append({"state": state["id"], "prompt": prompt,
                             "points": points, "box": box, "reset": reset})
        return {"mask_png_base64": f"ref{len(self.refines)}", "score": 0.6,
                "width": 4, "height": 3}


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def tick(self, dt): self.t += dt


def ids():
    seq = iter([f"s{i}" for i in range(100)])
    return lambda: next(seq)


def store(seg=None, max_sessions=8, ttl=900, clock=None, id_factory=None):
    return SessionStore(seg or FakeSeg(), max_sessions=max_sessions,
                        ttl_seconds=ttl, clock=clock or Clock(),
                        id_factory=id_factory or ids())


def test_create_returns_id_and_auto_mask():
    s = store()
    sid, result = s.create(b"img", "workpiece.", 0.3, 0.25)
    assert sid == "s0"
    assert result["mask_png_base64"] == "auto1"
    assert s.count() == 1


def test_refine_points_passes_through_to_segmentor():
    seg = FakeSeg()
    s = store(seg)
    sid, _ = s.create(b"img", "p", 0.3, 0.25)
    out = s.refine(sid, prompt=None, points=[(2.0, 3.0, 1)], box=None,
                   reset=False, box_threshold=0.3, text_threshold=0.25)
    assert out["mask_png_base64"] == "ref1"
    assert seg.refines[0]["points"] == [(2.0, 3.0, 1)]


def test_refine_unknown_session_raises_expired():
    s = store()
    with pytest.raises(SessionExpired):
        s.refine("nope", prompt=None, points=None, box=None, reset=False,
                 box_threshold=0.3, text_threshold=0.25)


def test_release_is_idempotent():
    s = store()
    sid, _ = s.create(b"img", "p", 0.3, 0.25)
    s.release(sid)
    s.release(sid)  # no error
    assert s.count() == 0
    with pytest.raises(SessionExpired):
        s.refine(sid, prompt=None, points=None, box=None, reset=False,
                 box_threshold=0.3, text_threshold=0.25)


def test_ttl_eviction_on_access():
    clk = Clock()
    s = store(ttl=100, clock=clk)
    sid, _ = s.create(b"img", "p", 0.3, 0.25)
    clk.tick(101)
    # A second create triggers the sweep; the stale session is gone.
    s.create(b"img2", "p", 0.3, 0.25)
    assert s.count() == 1
    with pytest.raises(SessionExpired):
        s.refine(sid, prompt=None, points=None, box=None, reset=False,
                 box_threshold=0.3, text_threshold=0.25)


def test_lru_eviction_when_over_max():
    s = store(max_sessions=2)
    a, _ = s.create(b"a", "p", 0.3, 0.25)
    b, _ = s.create(b"b", "p", 0.3, 0.25)
    # touch a so b is least-recently-used
    s.refine(a, prompt=None, points=None, box=None, reset=False,
             box_threshold=0.3, text_threshold=0.25)
    s.create(b"c", "p", 0.3, 0.25)  # over cap -> evict LRU (b)
    assert s.count() == 2
    with pytest.raises(SessionExpired):
        s.refine(b, prompt=None, points=None, box=None, reset=False,
                 box_threshold=0.3, text_threshold=0.25)
    # a still alive
    s.refine(a, prompt=None, points=None, box=None, reset=False,
             box_threshold=0.3, text_threshold=0.25)
