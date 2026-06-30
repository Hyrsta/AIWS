# Interactive Mask Refine for RGB-only Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a GUI user iteratively refine the auto-segmentation mask (re-prompt text, click +/- points, draw a box) on an uploaded RGB photo before launching the slow SAM3D+Cadrille reconstruction.

**Architecture:** Add a stateful segmentation session to `grounded-sam-svc` (encode the image into SAM once, then refine in milliseconds), a thin proxy in the GUI backend, and an interactive canvas in the GUI frontend. The refined mask feeds the existing `image_mask` reconstruction job unchanged. The existing stateless `/segment` and the headless REST gateway path are untouched.

**Tech Stack:** Python 3.10 / FastAPI / segment-anything (SAM `vit_h`) / transformers GroundingDINO (backend); React + TypeScript + Vite + Vitest (frontend); pytest (backend tests).

## Global Constraints

- No em-dashes or en-dashes in any file or prose (use comma, period, colon, hyphen). Grep `—|–` after edits.
- All work lands on branch `claude/mystifying-moore-6f426e` (PR #36), additive only (0 deletions vs origin/main).
- Backend unit tests must run CPU-only: SAM and GroundingDINO are never loaded in tests; fakes implement the segmentor/session interface.
- `Settings` is a frozen dataclass; every new field MUST have a default so existing `Settings(...)` construction in tests keeps working.
- The live deployment runs as host processes on RXL (grounded-sam-svc on `:18091`, GUI backend on `:18000`); the GUI backend reaches grounded-sam-svc via `GROUNDED_SAM_URL`.
- Coordinates crossing the service boundary are always original-image pixels (x, y), resolution-independent.
- Commit after each task with a `feat:`/`test:`/`docs:` message ending with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

## File Structure

Backend (`serving/grounded-sam-svc/`):
- `app/config.py` (modify) - add `session_ttl_seconds`, `max_sessions` settings.
- `app/segmentor.py` (modify) - factor a `_detect` helper out of `segment`; add session methods `encode_image`, `auto_mask`, `refine_mask`.
- `app/sessions.py` (create) - `SessionStore`: id generation, lookup, TTL + LRU eviction, `SessionExpired`. Pure lifecycle logic; segmentor injected.
- `app/main.py` (modify) - add `POST /segment/session`, `POST /segment/refine`, `DELETE /segment/session/{id}` and their request models.
- `tests/test_sessions.py` (create) - SessionStore unit tests with a fake segmentor + injected clock.
- `tests/test_session_api.py` (create) - endpoint contract tests with a fake segmentor.

GUI backend (`gui/backend/`):
- `app.py` (modify) - add proxy endpoints `POST /segment/session`, `POST /segment/refine`, `DELETE /segment/session/{id}` that forward to `GROUNDED_SAM_URL`.

GUI frontend (`gui/frontend/src/`):
- `api/types.ts` (modify) - `SegmentSession`, `RefineResult`, `RefinePoint` types.
- `api/client.ts` (modify) - `segmentSession`, `segmentRefine`, `segmentRelease`.
- `api/client.test.ts` (modify) - tests for the three client methods.
- `components/SegmentRefineCanvas.tsx` (create) - interactive canvas component.
- `components/SegmentRefineCanvas.test.tsx` (create) - canvas state-machine tests.
- `views/ConfigureView.tsx` (modify) - mount the canvas in RGB-only mode; "Use this mask" routes to `image_mask` reconstruct.
- `styles.css` (modify) - canvas styles.
- `i18n/en.json`, `i18n/zh.json` (modify) - labels.

---

### Task 1: SessionStore lifecycle (backend core logic)

The testable heart of the backend. `SessionStore` owns session ids, lookup, TTL and LRU eviction, and delegates all SAM work to an injected segmentor. Tests use a fake segmentor and a fake clock, so they are CPU-only and deterministic.

**Files:**
- Create: `serving/grounded-sam-svc/app/sessions.py`
- Test: `serving/grounded-sam-svc/tests/test_sessions.py`

**Interfaces:**
- Consumes: a segmentor object exposing:
  - `encode_image(image_bytes: bytes) -> object` (returns an opaque per-image `state`)
  - `auto_mask(state, prompt: str, box_threshold: float, text_threshold: float) -> dict` with keys `mask_png_base64, score, box, width, height, detected`
  - `refine_mask(state, prompt: str | None, points: list[tuple[float,float,int]] | None, box: list[float] | None, reset: bool, box_threshold: float, text_threshold: float) -> dict` with keys `mask_png_base64, score, width, height`
- Produces (used by Task 3):
  - `SessionStore(segmentor, max_sessions: int, ttl_seconds: float, clock=time.monotonic, id_factory=<uuid4 hex>)`
  - `create(image_bytes, prompt, box_threshold, text_threshold) -> tuple[str, dict]` returns `(session_id, auto_mask_dict)`
  - `refine(session_id, *, prompt, points, box, reset, box_threshold, text_threshold) -> dict`
  - `release(session_id) -> None` (idempotent)
  - `SessionExpired(Exception)` raised by `refine` on unknown/evicted id
  - `count() -> int`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd serving/grounded-sam-svc && python -m pytest tests/test_sessions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.sessions'`

- [ ] **Step 3: Write the implementation**

```python
# serving/grounded-sam-svc/app/sessions.py
import time
import uuid
from dataclasses import dataclass, field


class SessionExpired(Exception):
    """Raised when a session id is unknown or has been evicted."""


@dataclass
class _Entry:
    state: object
    prompt: str
    last_used: float
    box_threshold: float
    text_threshold: float


def _default_id():
    return uuid.uuid4().hex


class SessionStore:
    """In-memory SAM session lifecycle. All SAM work is delegated to `segmentor`.

    Not thread-safe on its own; the caller serializes access (the service runs a
    single GPU worker). TTL and LRU eviction bound GPU memory.
    """

    def __init__(self, segmentor, max_sessions, ttl_seconds,
                 clock=time.monotonic, id_factory=_default_id):
        self._seg = segmentor
        self._max = int(max_sessions)
        self._ttl = float(ttl_seconds)
        self._clock = clock
        self._id = id_factory
        self._sessions: dict[str, _Entry] = {}

    def count(self):
        return len(self._sessions)

    def _sweep(self):
        now = self._clock()
        stale = [sid for sid, e in self._sessions.items()
                 if now - e.last_used > self._ttl]
        for sid in stale:
            self._sessions.pop(sid, None)
        while len(self._sessions) > self._max:
            lru = min(self._sessions.items(), key=lambda kv: kv[1].last_used)[0]
            self._sessions.pop(lru, None)

    def create(self, image_bytes, prompt, box_threshold, text_threshold):
        state = self._seg.encode_image(image_bytes)
        result = self._seg.auto_mask(state, prompt, box_threshold, text_threshold)
        sid = self._id()
        self._sessions[sid] = _Entry(state=state, prompt=prompt,
                                     last_used=self._clock(),
                                     box_threshold=box_threshold,
                                     text_threshold=text_threshold)
        self._sweep()
        return sid, result

    def _get(self, session_id):
        entry = self._sessions.get(session_id)
        if entry is None:
            raise SessionExpired(session_id)
        if self._clock() - entry.last_used > self._ttl:
            self._sessions.pop(session_id, None)
            raise SessionExpired(session_id)
        return entry

    def refine(self, session_id, *, prompt, points, box, reset,
               box_threshold, text_threshold):
        entry = self._get(session_id)
        result = self._seg.refine_mask(
            entry.state, prompt=prompt, points=points, box=box, reset=reset,
            box_threshold=box_threshold, text_threshold=text_threshold)
        entry.last_used = self._clock()
        return result

    def release(self, session_id):
        self._sessions.pop(session_id, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd serving/grounded-sam-svc && python -m pytest tests/test_sessions.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add serving/grounded-sam-svc/app/sessions.py serving/grounded-sam-svc/tests/test_sessions.py
git commit -m "feat(grounded-sam-svc): SessionStore lifecycle (TTL + LRU)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Segmentor session methods + config

Refactor `segment` to share a `_detect` helper, then add the three SAM session methods the SessionStore calls. The new methods are thin wrappers over `SamPredictor` (warm one image, predict from box/points). They are validated on RXL in Task 8, not unit-tested with real SAM; the existing `segment` contract tests guard the refactor, and Task 1's fake pins the interface.

**Files:**
- Modify: `serving/grounded-sam-svc/app/config.py`
- Modify: `serving/grounded-sam-svc/app/segmentor.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Tasks 1, 3):
  - `Settings.session_ttl_seconds: int = 900`, `Settings.max_sessions: int = 8`
  - `Segmentor.encode_image(image_bytes) -> _ImageState`
  - `Segmentor.auto_mask(state, prompt, box_threshold, text_threshold) -> dict` keys `mask_png_base64, score, box, width, height, detected`
  - `Segmentor.refine_mask(state, prompt, points, box, reset, box_threshold, text_threshold) -> dict` keys `mask_png_base64, score, width, height`

- [ ] **Step 1: Add the config fields**

In `serving/grounded-sam-svc/app/config.py`, add two defaulted fields to the `Settings` dataclass (after `hf_home`):

```python
    session_ttl_seconds: int = 900
    max_sessions: int = 8
```

And in `load_settings()` (inside the `Settings(...)` call, after `hf_home=...`):

```python
        session_ttl_seconds=int(os.environ.get("SESSION_TTL_SECONDS", "900")),
        max_sessions=int(os.environ.get("MAX_SESSIONS", "8")),
```

- [ ] **Step 2: Verify existing tests still pass after the config change**

Run: `cd serving/grounded-sam-svc && python -m pytest tests/test_segment_contract.py tests/test_health.py -v`
Expected: PASS (existing tests unaffected; defaults keep `Settings(...)` calls valid)

- [ ] **Step 3: Refactor `segment` to share `_detect`, add session methods**

In `serving/grounded-sam-svc/app/segmentor.py`:

(a) Add a small image-state holder near the top (after `NoDetection`):

```python
from dataclasses import dataclass, field


@dataclass
class _ImageState:
    image_np: object        # HxWx3 uint8 ndarray
    width: int
    height: int
    points: list = field(default_factory=list)        # list[(x, y, label)]
    box: object = None                                  # [x0,y0,x1,y1] or None
    low_res_mask: object = None                         # SAM low-res mask logits or None
    sam_features: object = None                         # cached predictor.features
    sam_input_size: object = None
    sam_original_size: object = None
```

(b) Factor detection out of `segment`. Replace the body of `segment` so the GroundingDINO block becomes a helper `_detect(self, image_np, prompt, box_threshold, text_threshold) -> dict` returning `{"box": ndarray[4], "score": float, "label": str, "num_detections": int}` and raising `NoDetection` on zero boxes. `segment` calls `_detect`, then `set_image` + `predict(box=...)`, and builds the PNG via a new `_mask_to_b64(mask_uint8) -> str` helper. Keep `segment`'s return dict identical (keys `mask_png_base64, score, box, label, num_detections`) so the contract tests pass.

```python
    @staticmethod
    def _mask_to_b64(mask_uint8) -> str:
        buf = io.BytesIO()
        Image.fromarray(mask_uint8, mode="L").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _detect(self, image_np, image_size, prompt, box_threshold, text_threshold) -> dict:
        import torch
        image = Image.fromarray(image_np)
        text = self._normalize_prompt(prompt)
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._detector(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=[image_size[::-1]])[0]
        boxes = results["boxes"]; scores = results["scores"]
        labels = results.get("labels") or results.get("text_labels") or []
        if int(len(boxes)) == 0:
            raise NoDetection(prompt)
        best = int(torch.argmax(scores).item())
        return {"box": boxes[best].detach().cpu().numpy(),
                "score": float(scores[best].item()),
                "label": str(labels[best]) if best < len(labels) else prompt,
                "num_detections": int(len(boxes))}

    def segment(self, image_bytes, prompt, box_threshold, text_threshold) -> dict:
        import numpy as np
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        det = self._detect(image_np, image.size, prompt, box_threshold, text_threshold)
        self._predictor.set_image(image_np)
        masks, _, _ = self._predictor.predict(box=det["box"][None, :], multimask_output=False)
        mask = masks[0].astype(np.uint8) * 255
        return {"mask_png_base64": self._mask_to_b64(mask),
                "score": round(det["score"], 4),
                "box": [round(float(v), 2) for v in det["box"].tolist()],
                "label": det["label"], "num_detections": det["num_detections"]}
```

(c) Add the session methods. `encode_image` runs the slow `set_image` once and caches the predictor state on the `_ImageState`. `auto_mask` / `refine_mask` restore that state into the shared predictor before predicting, so multiple sessions can coexist.

```python
    def _restore(self, state):
        # Re-seat the cached image embedding into the shared predictor.
        p = self._predictor
        p.features = state.sam_features
        p.input_size = state.sam_input_size
        p.original_size = state.sam_original_size
        p.is_image_set = True

    def encode_image(self, image_bytes):
        import numpy as np
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        self._predictor.set_image(image_np)
        st = _ImageState(image_np=image_np, width=image.size[0], height=image.size[1])
        st.sam_features = self._predictor.features
        st.sam_input_size = self._predictor.input_size
        st.sam_original_size = self._predictor.original_size
        return st

    def auto_mask(self, state, prompt, box_threshold, text_threshold):
        import numpy as np
        self._restore(state)
        try:
            det = self._detect(state.image_np, (state.width, state.height),
                               prompt, box_threshold, text_threshold)
        except NoDetection:
            empty = np.zeros((state.height, state.width), dtype=np.uint8)
            state.box = None; state.low_res_mask = None; state.points = []
            return {"mask_png_base64": self._mask_to_b64(empty), "score": 0.0,
                    "box": None, "width": state.width, "height": state.height,
                    "detected": False}
        masks, _, low = self._predictor.predict(box=det["box"][None, :], multimask_output=False)
        state.box = det["box"].tolist(); state.points = []
        state.low_res_mask = low
        mask = masks[0].astype(np.uint8) * 255
        return {"mask_png_base64": self._mask_to_b64(mask),
                "score": round(det["score"], 4),
                "box": [round(float(v), 2) for v in det["box"].tolist()],
                "width": state.width, "height": state.height, "detected": True}

    def refine_mask(self, state, prompt, points, box, reset, box_threshold, text_threshold):
        import numpy as np
        self._restore(state)
        if reset:
            state.points = []; state.box = None; state.low_res_mask = None
            return self.auto_mask(state, prompt or "", box_threshold, text_threshold) \
                if False else {"mask_png_base64": self._mask_to_b64(
                    np.zeros((state.height, state.width), np.uint8)),
                    "score": 0.0, "width": state.width, "height": state.height}
        if prompt:
            # Fresh text detection replaces accumulated points/box.
            det = self._detect(state.image_np, (state.width, state.height),
                               prompt, box_threshold, text_threshold)
            state.points = []; state.box = det["box"].tolist()
            masks, _, low = self._predictor.predict(box=det["box"][None, :], multimask_output=False)
        else:
            if points:
                for (x, y, lab) in points:
                    state.points.append((float(x), float(y), int(lab)))
            if box is not None:
                state.box = [float(v) for v in box]
            pc = np.array([[p[0], p[1]] for p in state.points], dtype=np.float32) if state.points else None
            pl = np.array([p[2] for p in state.points], dtype=np.int32) if state.points else None
            bx = np.array(state.box, dtype=np.float32)[None, :] if state.box is not None else None
            mi = state.low_res_mask if state.low_res_mask is not None else None
            masks, _, low = self._predictor.predict(
                point_coords=pc, point_labels=pl, box=bx,
                mask_input=mi, multimask_output=False)
        state.low_res_mask = low
        mask = masks[0].astype(np.uint8) * 255
        return {"mask_png_base64": self._mask_to_b64(mask), "score": 0.0,
                "width": state.width, "height": state.height}
```

Note: the `reset` branch returns an empty mask and clears accumulation; the GUI follows a `reset` immediately with a re-prompt to repopulate the auto mask (see Task 6). Keep `import numpy as np` at module top if not already present (the original imported it at top; preserve that).

- [ ] **Step 4: Run the existing contract tests (guards the refactor)**

Run: `cd serving/grounded-sam-svc && python -m pytest tests/test_segment_contract.py -v`
Expected: PASS (4 passed) - `segment`'s public contract is unchanged.

- [ ] **Step 5: Em-dash check + commit**

```bash
grep -nE "—|–" serving/grounded-sam-svc/app/segmentor.py serving/grounded-sam-svc/app/config.py && echo FOUND || echo clean
git add serving/grounded-sam-svc/app/segmentor.py serving/grounded-sam-svc/app/config.py
git commit -m "feat(grounded-sam-svc): segmentor session methods + session config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Session endpoints (grounded-sam-svc)

Wire the SessionStore into `build_app` and expose the three endpoints. Contract tests use a fake segmentor (implementing the Task 2 interface) and the real SessionStore.

**Files:**
- Modify: `serving/grounded-sam-svc/app/main.py`
- Test: `serving/grounded-sam-svc/tests/test_session_api.py`

**Interfaces:**
- Consumes: `SessionStore`, `SessionExpired` (Task 1); `Settings.session_ttl_seconds`, `Settings.max_sessions` (Task 2).
- Produces: HTTP endpoints
  - `POST /segment/session` (multipart `image`, optional form `prompt`) -> `{session_id, mask_png_base64, box, score, width, height, detected}`
  - `POST /segment/refine` (JSON `{session_id, prompt?, points?:[{x,y,label}], box?:[x0,y0,x1,y1], reset?}`) -> `{mask_png_base64, score, width, height}`; `409` on `SessionExpired`
  - `DELETE /segment/session/{session_id}` -> `204`

- [ ] **Step 1: Write the failing tests**

```python
# serving/grounded-sam-svc/tests/test_session_api.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
from fastapi.testclient import TestClient
from app.main import build_app
from app.config import Settings


def settings():
    return Settings(default_prompt="workpiece. metal part.", grounding_dino_ckpt="x",
                    sam_ckpt="x", sam_model_type="vit_h", device="cuda:0",
                    box_threshold=0.3, text_threshold=0.25, max_sessions=4)


class FakeSeg:
    ready = True
    def segment(self, *a, **k):
        return {"mask_png_base64": "S", "score": 0.5, "box": [0, 0, 1, 1],
                "label": "x", "num_detections": 1}
    def encode_image(self, image_bytes):
        return {"img": image_bytes}
    def auto_mask(self, state, prompt, bt, tt):
        return {"mask_png_base64": "AUTO", "score": 0.5, "box": [0, 0, 1, 1],
                "width": 4, "height": 3, "detected": True}
    def refine_mask(self, state, prompt, points, box, reset, bt, tt):
        return {"mask_png_base64": "REF", "score": 0.6, "width": 4, "height": 3}


def png():
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def client():
    return TestClient(build_app(settings(), FakeSeg()))


def test_session_create_returns_id_and_mask():
    r = client().post("/segment/session", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["mask_png_base64"] == "AUTO"
    assert body["session_id"]


def test_refine_with_points_returns_mask():
    c = client()
    sid = c.post("/segment/session", files={"image": ("a.png", png(), "image/png")}).json()["session_id"]
    r = c.post("/segment/refine", json={"session_id": sid,
              "points": [{"x": 2, "y": 3, "label": 1}]})
    assert r.status_code == 200
    assert r.json()["mask_png_base64"] == "REF"


def test_refine_unknown_session_returns_409():
    r = client().post("/segment/refine", json={"session_id": "nope", "points": []})
    assert r.status_code == 409


def test_delete_session_returns_204():
    c = client()
    sid = c.post("/segment/session", files={"image": ("a.png", png(), "image/png")}).json()["session_id"]
    r = c.delete(f"/segment/session/{sid}")
    assert r.status_code == 204
    # refine after delete -> 409
    r2 = c.post("/segment/refine", json={"session_id": sid, "points": []})
    assert r2.status_code == 409
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd serving/grounded-sam-svc && python -m pytest tests/test_session_api.py -v`
Expected: FAIL with `404` (endpoints not defined yet)

- [ ] **Step 3: Add the endpoints to `build_app`**

In `serving/grounded-sam-svc/app/main.py`, add imports and build the store inside `build_app`, then register the routes (keep the existing `/healthz` and `/segment` untouched):

```python
from fastapi import Body
from pydantic import BaseModel
from .sessions import SessionStore, SessionExpired


class RefinePoint(BaseModel):
    x: float
    y: float
    label: int = 1


class RefineBody(BaseModel):
    session_id: str
    prompt: str | None = None
    points: list[RefinePoint] | None = None
    box: list[float] | None = None
    reset: bool = False
```

Inside `build_app(settings, segmentor)`, after `app = FastAPI(...)`:

```python
    store = SessionStore(segmentor,
                         max_sessions=getattr(settings, "max_sessions", 8),
                         ttl_seconds=getattr(settings, "session_ttl_seconds", 900))

    @app.post("/segment/session")
    def segment_session(image: UploadFile = File(...), prompt: str = Form(None)):
        data = image.file.read()
        used_prompt = prompt or settings.default_prompt
        try:
            sid, result = store.create(data, used_prompt,
                                       settings.box_threshold, settings.text_threshold)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))
        return {"session_id": sid, **result}

    @app.post("/segment/refine")
    def segment_refine(body: RefineBody = Body(...)):
        pts = [(p.x, p.y, p.label) for p in body.points] if body.points else None
        try:
            return store.refine(body.session_id, prompt=body.prompt, points=pts,
                                box=body.box, reset=body.reset,
                                box_threshold=settings.box_threshold,
                                text_threshold=settings.text_threshold)
        except SessionExpired:
            raise HTTPException(status_code=409, detail="session_expired")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/segment/session/{session_id}", status_code=204)
    def segment_release(session_id: str):
        store.release(session_id)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd serving/grounded-sam-svc && python -m pytest tests/ -v`
Expected: PASS (all session + contract + health tests green)

- [ ] **Step 5: Commit**

```bash
git add serving/grounded-sam-svc/app/main.py serving/grounded-sam-svc/tests/test_session_api.py
git commit -m "feat(grounded-sam-svc): session create/refine/delete endpoints

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: GUI backend proxy endpoints

Thin pass-through so the browser only talks to the GUI origin. Reuse the existing urllib multipart pattern already in `app.py` (used by the RGB-only auto-segment path around lines 460-509).

**Files:**
- Modify: `gui/backend/app.py`

**Interfaces:**
- Consumes: `GROUNDED_SAM_URL` (module constant already present).
- Produces: GUI endpoints
  - `POST /segment/session` (multipart `image`, optional `prompt`) -> grounded-sam-svc JSON
  - `POST /segment/refine` (JSON body forwarded verbatim) -> JSON; propagates `409`
  - `DELETE /segment/session/{session_id}` -> `204`

- [ ] **Step 1: Add the proxy endpoints**

Add near the existing job routes in `gui/backend/app.py` (after the `/jobs/simple-reconstruct` handler). Use `urllib.request` (already imported in this file for the grounded-sam call) and `json`:

```python
@app.post("/segment/session")
async def segment_session(image: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    data = await image.read()
    boundary = b"AIWS_GSAM_REFINE_BOUNDARY"
    parts = [b"--" + boundary,
             b'Content-Disposition: form-data; name="image"; filename="upload.png"',
             b"Content-Type: application/octet-stream", b"", data]
    if prompt:
        parts += [b"--" + boundary,
                  b'Content-Disposition: form-data; name="prompt"', b"",
                  prompt.encode()]
    parts += [b"--" + boundary + b"--", b""]
    payload = b"\r\n".join(parts)
    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/segment/session", data=payload,
        headers={"Content-Type": b"multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return JSONResponse(status_code=resp.status, content=json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=exc.read().decode("utf-8", "replace"))
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"grounded-sam-svc unreachable: {exc}")


@app.post("/segment/refine")
async def segment_refine(body: dict = Body(...)):
    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/segment/refine", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return JSONResponse(status_code=resp.status, content=json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        # Propagate 409 so the frontend can re-create the session.
        raise HTTPException(status_code=exc.code, detail=exc.read().decode("utf-8", "replace"))
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"grounded-sam-svc unreachable: {exc}")


@app.delete("/segment/session/{session_id}", status_code=204)
async def segment_release(session_id: str):
    req = urllib.request.Request(
        f"{GROUNDED_SAM_URL}/segment/session/{session_id}", method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception:  # noqa: BLE001
        pass  # best-effort release; TTL will sweep otherwise
    return None
```

Ensure the imports at the top of `app.py` include `Body` from fastapi and `json`, `urllib.request`, `urllib.error` (the grounded-sam call already uses urllib; add `from fastapi import Body` to the existing fastapi import line and `import json` if not present).

- [ ] **Step 2: Smoke-check the module imports**

Run: `cd gui && python -c "import ast; ast.parse(open('backend/app.py').read()); print('parse ok')"`
Expected: `parse ok` (syntax valid; the route wiring is exercised live in Task 8)

- [ ] **Step 3: Em-dash check + commit**

```bash
grep -nE "—|–" gui/backend/app.py && echo FOUND || echo clean
git add gui/backend/app.py
git commit -m "feat(gui): proxy endpoints for grounded-sam session refine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Frontend API client + types

**Files:**
- Modify: `gui/frontend/src/api/types.ts`
- Modify: `gui/frontend/src/api/client.ts`
- Test: `gui/frontend/src/api/client.test.ts`

**Interfaces:**
- Produces (used by Tasks 6, 7):
  - `RefinePoint = { x: number; y: number; label: 1 | 0 }`
  - `SegmentSession = { session_id: string; mask_png_base64: string; box: number[] | null; score: number; width: number; height: number; detected: boolean }`
  - `RefineResult = { mask_png_base64: string; score: number; width: number; height: number }`
  - `api.segmentSession(image: File, prompt?: string) => Promise<SegmentSession>`
  - `api.segmentRefine(sessionId: string, edit: { prompt?: string; points?: RefinePoint[]; box?: number[]; reset?: boolean }) => Promise<RefineResult>`
  - `api.segmentRelease(sessionId: string) => Promise<void>`

- [ ] **Step 1: Add the types**

Append to `gui/frontend/src/api/types.ts`:

```typescript
export interface RefinePoint { x: number; y: number; label: 1 | 0; }
export interface SegmentSession {
  session_id: string; mask_png_base64: string; box: number[] | null;
  score: number; width: number; height: number; detected: boolean;
}
export interface RefineResult { mask_png_base64: string; score: number; width: number; height: number; }
export interface RefineEdit { prompt?: string; points?: RefinePoint[]; box?: number[]; reset?: boolean; }
```

- [ ] **Step 2: Write the failing client tests**

Append to `gui/frontend/src/api/client.test.ts` (follow the existing fetch-mock style in that file; if the file stubs `global.fetch`, reuse that helper):

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { api } from "./client";

function mockFetchOnce(json: unknown, ok = true, status = 200) {
  (global.fetch as unknown) = vi.fn().mockResolvedValue({
    ok, status, json: async () => json, text: async () => JSON.stringify(json),
  });
}

describe("segment refine client", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("segmentSession posts the image and returns the session", async () => {
    mockFetchOnce({ session_id: "s1", mask_png_base64: "AUTO", box: null,
      score: 0.5, width: 4, height: 3, detected: true });
    const file = new File([new Uint8Array([1, 2, 3])], "a.png", { type: "image/png" });
    const out = await api.segmentSession(file, "bracket");
    expect(out.session_id).toBe("s1");
    const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(String(url)).toContain("/segment/session");
    expect((init as RequestInit).method).toBe("POST");
  });

  it("segmentRefine posts JSON body with points", async () => {
    mockFetchOnce({ mask_png_base64: "REF", score: 0.6, width: 4, height: 3 });
    const out = await api.segmentRefine("s1", { points: [{ x: 2, y: 3, label: 1 }] });
    expect(out.mask_png_base64).toBe("REF");
    const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(String(url)).toContain("/segment/refine");
    expect(JSON.parse((init as RequestInit).body as string).points[0].x).toBe(2);
  });

  it("segmentRelease issues DELETE", async () => {
    mockFetchOnce({}, true, 204);
    await api.segmentRelease("s1");
    const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(String(url)).toContain("/segment/session/s1");
    expect((init as RequestInit).method).toBe("DELETE");
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd gui/frontend && npx vitest run src/api/client.test.ts`
Expected: FAIL (`api.segmentSession is not a function`)

- [ ] **Step 4: Implement the client methods**

In `gui/frontend/src/api/client.ts`, add the import and three methods inside the `api` object (before the closing `}`):

```typescript
// add to the type import at the top:
//   import type { ..., SegmentSession, RefineResult, RefineEdit } from "./types";

  async segmentSession(image: File, prompt?: string): Promise<SegmentSession> {
    const fd = new FormData();
    fd.append("image", image);
    if (prompt) fd.append("prompt", prompt);
    const r = await fetch(`${BASE}/segment/session`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(`POST /segment/session → ${r.status}`);
    return (await r.json()) as SegmentSession;
  },
  async segmentRefine(sessionId: string, edit: RefineEdit): Promise<RefineResult> {
    const r = await fetch(`${BASE}/segment/refine`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, ...edit }),
    });
    if (!r.ok) { const e = new Error(`POST /segment/refine → ${r.status}`);
      (e as Error & { status?: number }).status = r.status; throw e; }
    return (await r.json()) as RefineResult;
  },
  async segmentRelease(sessionId: string): Promise<void> {
    await fetch(`${BASE}/segment/session/${sessionId}`, { method: "DELETE" });
  },
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd gui/frontend && npx vitest run src/api/client.test.ts`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gui/frontend/src/api/types.ts gui/frontend/src/api/client.ts gui/frontend/src/api/client.test.ts
git commit -m "feat(gui): segment session/refine/release API client

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: SegmentRefineCanvas component

A self-contained interactive canvas. It owns the refine state (points, box, current mask) and the coordinate mapping; it calls the API methods from Task 5 and reports the final mask up via `onMaskChange`. The pure state helpers are exported and unit-tested; the rendering/DOM is exercised live in Task 8.

**Files:**
- Create: `gui/frontend/src/components/SegmentRefineCanvas.tsx`
- Test: `gui/frontend/src/components/SegmentRefineCanvas.test.tsx`

**Interfaces:**
- Consumes: `api.segmentSession/segmentRefine/segmentRelease`, `RefinePoint` (Task 5).
- Produces:
  - `export function canvasToImage(cx, cy, scale) => { x, y }` (pure; `scale = displayWidth / imageWidth`)
  - `export function pngDataUrlToFile(dataUrl: string, name: string) => File`
  - `export function SegmentRefineCanvas(props: { image: File; defaultPrompt: string; onMaskChange: (maskFile: File | null) => void })`

- [ ] **Step 1: Write the failing tests for the pure helpers**

```typescript
// gui/frontend/src/components/SegmentRefineCanvas.test.tsx
import { describe, it, expect } from "vitest";
import { canvasToImage } from "./SegmentRefineCanvas";

describe("canvasToImage", () => {
  it("maps canvas coords to image pixels at unit scale", () => {
    expect(canvasToImage(10, 20, 1)).toEqual({ x: 10, y: 20 });
  });
  it("scales when the image is displayed smaller", () => {
    // image displayed at half size: scale = 0.5 -> divide by scale
    expect(canvasToImage(10, 20, 0.5)).toEqual({ x: 20, y: 40 });
  });
  it("rounds to whole pixels", () => {
    expect(canvasToImage(11, 21, 0.5)).toEqual({ x: 22, y: 42 });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd gui/frontend && npx vitest run src/components/SegmentRefineCanvas.test.tsx`
Expected: FAIL (module/exports not found)

- [ ] **Step 3: Implement the component**

```tsx
// gui/frontend/src/components/SegmentRefineCanvas.tsx
import { useEffect, useRef, useState, useCallback } from "react";
import { api } from "../api/client";
import type { RefinePoint } from "../api/types";

export function canvasToImage(cx: number, cy: number, scale: number) {
  return { x: Math.round(cx / scale), y: Math.round(cy / scale) };
}

export function pngDataUrlToFile(dataUrl: string, name: string): File {
  const b64 = dataUrl.split(",")[1] ?? dataUrl;
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new File([arr], name, { type: "image/png" });
}

type Props = { image: File; defaultPrompt: string; onMaskChange: (m: File | null) => void };

export function SegmentRefineCanvas({ image, defaultPrompt, onMaskChange }: Props) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [maskB64, setMaskB64] = useState<string | null>(null);
  const [points, setPoints] = useState<RefinePoint[]>([]);
  const [busy, setBusy] = useState(false);
  const [hint, setHint] = useState<string>("");
  const imgUrlRef = useRef<string>("");
  const imgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const wrapRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x0: number; y0: number } | null>(null);

  // Object URL for the uploaded image preview.
  useEffect(() => {
    const url = URL.createObjectURL(image);
    imgUrlRef.current = url;
    const im = new Image();
    im.onload = () => { imgSizeRef.current = { w: im.width, h: im.height }; };
    im.src = url;
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const publishMask = useCallback((b64: string | null) => {
    setMaskB64(b64);
    onMaskChange(b64 ? pngDataUrlToFile(b64, "refined_mask.png") : null);
  }, [onMaskChange]);

  // Create the session + initial auto-mask on mount / image change.
  useEffect(() => {
    let alive = true;
    setBusy(true); setHint("");
    api.segmentSession(image, prompt).then((s) => {
      if (!alive) return;
      setSessionId(s.session_id);
      setPoints([]);
      publishMask(s.mask_png_base64);
      if (!s.detected) setHint("No object detected. Try a different prompt or click a point.");
    }).catch(() => { if (alive) setHint("Segmentation service unavailable."); })
      .finally(() => { if (alive) setBusy(false); });
    return () => {
      alive = false;
      if (sessionId) api.segmentRelease(sessionId).catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image]);

  async function callRefine(edit: { prompt?: string; points?: RefinePoint[]; box?: number[]; reset?: boolean }) {
    if (!sessionId) return;
    setBusy(true);
    try {
      const r = await api.segmentRefine(sessionId, edit);
      publishMask(r.mask_png_base64);
      setHint("");
    } catch (e) {
      const status = (e as Error & { status?: number }).status;
      if (status === 409) {
        // session expired: re-create from the image we still hold, then retry once
        const s = await api.segmentSession(image, prompt);
        setSessionId(s.session_id);
        const r = await api.segmentRefine(s.session_id, edit);
        publishMask(r.mask_png_base64);
      } else if (status === 422) {
        setHint("No match for that prompt. Keeping the current mask.");
      } else {
        setHint("Refine failed. Try again.");
      }
    } finally { setBusy(false); }
  }

  const scale = () => {
    const w = wrapRef.current?.clientWidth ?? imgSizeRef.current.w;
    return imgSizeRef.current.w ? w / imgSizeRef.current.w : 1;
  };

  function onClickPoint(ev: React.MouseEvent) {
    if (busy) return;
    const rect = wrapRef.current!.getBoundingClientRect();
    const { x, y } = canvasToImage(ev.clientX - rect.left, ev.clientY - rect.top, scale());
    const label: 1 | 0 = ev.altKey ? 0 : 1;
    const next = [...points, { x, y, label }];
    setPoints(next);
    callRefine({ points: [{ x, y, label }] });
  }

  function onMouseDown(ev: React.MouseEvent) {
    const rect = wrapRef.current!.getBoundingClientRect();
    dragRef.current = { x0: ev.clientX - rect.left, y0: ev.clientY - rect.top };
  }
  function onMouseUp(ev: React.MouseEvent) {
    const d = dragRef.current; dragRef.current = null;
    if (!d) return;
    const rect = wrapRef.current!.getBoundingClientRect();
    const x1 = ev.clientX - rect.left, y1 = ev.clientY - rect.top;
    if (Math.abs(x1 - d.x0) < 6 && Math.abs(y1 - d.y0) < 6) { onClickPoint(ev); return; }
    const s = scale();
    const a = canvasToImage(Math.min(d.x0, x1), Math.min(d.y0, y1), s);
    const b = canvasToImage(Math.max(d.x0, x1), Math.max(d.y0, y1), s);
    callRefine({ box: [a.x, a.y, b.x, b.y] });
  }

  return (
    <div className="seg-refine">
      <div className="seg-canvas-wrap" ref={wrapRef}
           onMouseDown={onMouseDown} onMouseUp={onMouseUp}
           onContextMenu={(e) => e.preventDefault()}>
        {imgUrlRef.current && <img className="seg-base" src={imgUrlRef.current} alt="input" />}
        {maskB64 && <img className="seg-mask" src={`data:image/png;base64,${maskB64}`} alt="mask" />}
        {busy && <div className="seg-busy">…</div>}
      </div>
      <div className="seg-controls">
        <input className="seg-prompt" value={prompt}
               onChange={(e) => setPrompt(e.target.value)}
               placeholder="workpiece. metal part." />
        <button type="button" disabled={busy} onClick={() => callRefine({ prompt })}>Re-detect</button>
        <button type="button" disabled={busy || !points.length}
                onClick={() => { const n = points.slice(0, -1); setPoints(n);
                  callRefine({ reset: true }); if (n.length) callRefine({ points: n }); }}>Undo point</button>
        <button type="button" disabled={busy}
                onClick={() => { setPoints([]); callRefine({ reset: true }); callRefine({ prompt }); }}>Reset to auto</button>
      </div>
      {hint && <div className="seg-hint">{hint}</div>}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd gui/frontend && npx vitest run src/components/SegmentRefineCanvas.test.tsx`
Expected: PASS (canvasToImage tests green)

- [ ] **Step 5: Em-dash check + typecheck + commit**

```bash
grep -nE "—|–" gui/frontend/src/components/SegmentRefineCanvas.tsx && echo FOUND || echo clean
cd gui/frontend && npx tsc --noEmit
git add gui/frontend/src/components/SegmentRefineCanvas.tsx gui/frontend/src/components/SegmentRefineCanvas.test.tsx
git commit -m "feat(gui): interactive SegmentRefineCanvas component

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Wire the canvas into ConfigureView RGB-only mode

When `inputMode === "image"`, render `SegmentRefineCanvas` once a photo is chosen. Track the refined mask in state; enable "Use this mask and reconstruct" when a mask exists; on submit, call `createSimpleReconstruct` with `input_mode: "image_mask"`, the chosen image, and the refined mask file.

**Files:**
- Modify: `gui/frontend/src/views/ConfigureView.tsx`
- Modify: `gui/frontend/src/styles.css`
- Modify: `gui/frontend/src/i18n/en.json`, `gui/frontend/src/i18n/zh.json`
- Test: `gui/frontend/src/views/ConfigureView.test.tsx`

**Interfaces:**
- Consumes: `SegmentRefineCanvas` (Task 6); `api.createSimpleReconstruct` (existing) with the `image_mask` variant.

- [ ] **Step 1: Add i18n keys**

In `gui/frontend/src/i18n/en.json` add:

```json
  "refine.title": "Refine the mask",
  "refine.useMask": "Use this mask and reconstruct",
  "refine.hint": "Re-detect with a new prompt, click to add points (Alt-click to exclude), or drag a box.",
  "refine.needMask": "Adjust the mask, then reconstruct",
```

In `gui/frontend/src/i18n/zh.json` add (no em-dashes):

```json
  "refine.title": "优化掩膜",
  "refine.useMask": "使用此掩膜并重建",
  "refine.hint": "用新的提示词重新检测，点击添加点（Alt 点击表示排除），或拖拽框选。",
  "refine.needMask": "请先调整掩膜，然后重建",
```

- [ ] **Step 2: Add canvas styles**

Append to `gui/frontend/src/styles.css`:

```css
.seg-refine { display: flex; flex-direction: column; gap: 10px; }
.seg-canvas-wrap { position: relative; width: 100%; cursor: crosshair; user-select: none; line-height: 0; }
.seg-base { width: 100%; height: auto; display: block; border-radius: var(--r-sm); }
.seg-mask { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0.45;
  mix-blend-mode: screen; pointer-events: none; }
.seg-busy { position: absolute; top: 8px; right: 10px; font-family: var(--mono); color: var(--tx-hi); }
.seg-controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.seg-prompt { flex: 1 1 220px; min-width: 0; }
.seg-hint { font-size: 12px; color: var(--tx-lo); }
```

- [ ] **Step 3: Write the failing ConfigureView test**

Add to `gui/frontend/src/views/ConfigureView.test.tsx` a test that, in RGB-only mode with a selected image, the "Use this mask" button appears and routes a reconstruct call. Mock `../api/client` so `segmentSession` resolves an auto-mask and `createSimpleReconstruct` records its input. Follow the existing render/setup helpers in that test file. Assert that after the mask is published, clicking "Use this mask and reconstruct" calls `createSimpleReconstruct` with `input_mode === "image_mask"` and a non-null `mask`.

```typescript
// sketch - adapt to the file's existing render harness and mocking style
it("RGB-only: use-mask routes an image_mask reconstruct", async () => {
  // mock api.segmentSession -> { session_id:"s", mask_png_base64:"AAAA", detected:true, ... }
  // mock api.createSimpleReconstruct -> capture input
  // render ConfigureView, select RGB-only mode, attach an image File,
  // wait for the canvas to publish a mask, click the "Use this mask and reconstruct" button
  // expect captured.input_mode === "image_mask" and captured.mask instanceof File
});
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd gui/frontend && npx vitest run src/views/ConfigureView.test.tsx`
Expected: FAIL (no "Use this mask" control / mask not routed yet)

- [ ] **Step 5: Implement the wiring**

In `gui/frontend/src/views/ConfigureView.tsx`:
- import `SegmentRefineCanvas`.
- add state `const [refinedMask, setRefinedMask] = useState<File | null>(null);`
- in the RGB-only (`inputMode === "image"`) branch of the input panel, when an image File is selected, render:

```tsx
<SegmentRefineCanvas image={selectedImage}
  defaultPrompt={detectPrompt || t("refine.hint")}
  onMaskChange={setRefinedMask} />
```

(use the existing selected-image state variable name from the file; `detectPrompt` is the existing RGB-only prompt state.)
- gate the run action in RGB-only mode on `refinedMask != null` (show `t("refine.needMask")` otherwise), and change the RGB-only submit to:

```tsx
await api.createSimpleReconstruct({
  cadrille_checkpoint_preset, cadrille_mode, workpiece_class, model_code, gpu_index,
  input_mode: "image_mask", image: selectedImage, mask: refinedMask!,
});
```

Keep the mesh and image+mask branches unchanged. The run button label/flow for the other modes stays as-is.

- [ ] **Step 6: Run tests + typecheck + build**

Run: `cd gui/frontend && npx vitest run src/views/ConfigureView.test.tsx && npx tsc --noEmit && npm run build`
Expected: PASS and a clean production build.

- [ ] **Step 7: Em-dash check + commit**

```bash
grep -nE "—|–" gui/frontend/src/views/ConfigureView.tsx gui/frontend/src/styles.css gui/frontend/src/i18n/en.json gui/frontend/src/i18n/zh.json && echo FOUND || echo clean
git add gui/frontend/src/views/ConfigureView.tsx gui/frontend/src/styles.css gui/frontend/src/i18n/en.json gui/frontend/src/i18n/zh.json gui/frontend/src/views/ConfigureView.test.tsx
git commit -m "feat(gui): refine canvas in RGB-only flow, route refined mask to reconstruct

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: On-RXL integration validation

Deploy to the live host processes and confirm the loop end-to-end. This task has no new unit tests; it produces a written validation note and a fix-forward if anything is off.

**Files:**
- None (deployment + validation). Any fixes commit against the relevant task's files.

- [ ] **Step 1: Deploy backend + frontend on RXL**

```bash
# on RXL (relay-pushed branch already at HEAD)
cd /ssd1/rxl/zhankaiming/AIWS-gsam && git fetch origin && git reset --hard origin/claude/mystifying-moore-6f426e
# re-link repos symlinks if reset cleared them (see fix_lineage pattern)
# restart grounded-sam-svc host process (port 18091) so the new endpoints load
# rebuild the GUI frontend
cd gui/frontend && npm run build
# the GUI backend (uvicorn) picks up app.py on restart; restart the tmux backend process
```

- [ ] **Step 2: Backend endpoint smoke (curl through the GUI origin)**

```bash
# session create with a real workpiece photo:
curl -s -F image=@/path/to/workpiece.png http://127.0.0.1:18000/segment/session | python -m json.tool | head
# expect session_id + mask_png_base64 + detected:true
# refine by an exclude point (use the returned session_id):
curl -s -X POST http://127.0.0.1:18000/segment/refine -H 'Content-Type: application/json' \
  -d '{"session_id":"<id>","points":[{"x":100,"y":100,"label":0}]}' | python -c "import sys,json;print(list(json.load(sys.stdin).keys()))"
# expect ['mask_png_base64','score','width','height']
```

- [ ] **Step 3: Latency check**

Time a refine call; it must be well under one second (the image is already encoded).

```bash
time curl -s -X POST http://127.0.0.1:18000/segment/refine -H 'Content-Type: application/json' \
  -d '{"session_id":"<id>","points":[{"x":120,"y":120,"label":1}]}' >/dev/null
```

Expected: real time < 1s.

- [ ] **Step 4: Live GUI validation**

Through the tunnel (`localhost:18000`): pick RGB-only, upload the torch+workpiece photo, confirm the auto-mask renders, add an exclude point on the torch and confirm the torch leaves the mask within a second, then click "Use this mask and reconstruct" and confirm a normal `image_mask` job starts and completes with the refined mask.

- [ ] **Step 5: Record the result + commit any fixes**

Append a short dated validation note to `serving/README.md` (or the spec's status), commit. If a defect surfaced, fix it in the owning task's file with a `fix:` commit and re-run that task's tests.

```bash
git add -A && git commit -m "docs: record RXL validation of interactive mask refine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Stateful session (encode once) -> Tasks 1, 2. Endpoints session/refine/delete -> Task 3. Text re-prompt + points + box -> Task 2 (`refine_mask`), Task 6 (canvas). TTL + LRU + 409 re-create -> Task 1 (store), Task 3 (409), Task 6 (re-create). Stateless `/segment` + REST gateway untouched -> Tasks 2/3 keep `segment` and add only new routes. GUI proxy -> Task 4. Canvas + coordinate mapping + "Use this mask" -> Tasks 6, 7. Error handling (missed prompt keeps mask; reconstruct only on button) -> Task 6 (422 hint), Task 7 (gated submit). Testing (mocked SAM, canvas state, RXL validation) -> Tasks 1/3/5/6 unit tests, Task 8 RXL. All spec sections map to a task.

**Placeholder scan:** Task 7 Step 3 gives a test sketch rather than full code because it must adapt to ConfigureView.test.tsx's existing harness (unknown helper names); the assertions to make are stated explicitly. All other code steps are complete.

**Type consistency:** `refine_mask` / `auto_mask` / `encode_image` signatures match between Task 1's fake, Task 2's implementation, and Task 3's endpoints. `segmentSession/segmentRefine/segmentRelease` names match across Tasks 5, 6, 7. `SegmentSession`/`RefineResult`/`RefinePoint`/`RefineEdit` are defined in Task 5 and consumed in Tasks 6, 7. The reconstruct call uses the existing `image_mask` `ReconstructInput` variant from `types.ts`.
