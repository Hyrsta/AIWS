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
    access_seq: int = 0  # monotonically increasing; ties in last_used broken by this


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
        self._seq = 0

    def _next_seq(self):
        self._seq += 1
        return self._seq

    def count(self):
        return len(self._sessions)

    def _sweep(self):
        now = self._clock()
        stale = [sid for sid, e in self._sessions.items()
                 if now - e.last_used > self._ttl]
        for sid in stale:
            self._sessions.pop(sid, None)
        while len(self._sessions) > self._max:
            lru = min(self._sessions.items(),
                      key=lambda kv: (kv[1].last_used, kv[1].access_seq))[0]
            self._sessions.pop(lru, None)

    def create(self, image_bytes, prompt, box_threshold, text_threshold):
        state = self._seg.encode_image(image_bytes)
        result = self._seg.auto_mask(state, prompt, box_threshold, text_threshold)
        sid = self._id()
        self._sessions[sid] = _Entry(state=state, prompt=prompt,
                                     last_used=self._clock(),
                                     box_threshold=box_threshold,
                                     text_threshold=text_threshold,
                                     access_seq=self._next_seq())
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
        entry.access_seq = self._next_seq()
        return result

    def release(self, session_id):
        self._sessions.pop(session_id, None)
