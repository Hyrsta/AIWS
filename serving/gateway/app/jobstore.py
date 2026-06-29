import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from .models import JobState, Stage, ReconstructOptions, JobView

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT,
    error TEXT,
    options TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, options: ReconstructOptions) -> str:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (job_id, status, stage, error, options, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (job_id, JobState.queued.value, None, None,
                 options.model_dump_json(), now, now),
            )
        return job_id

    def get(self, job_id: str) -> Optional[JobView]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobView(
            job_id=row["job_id"],
            status=JobState(row["status"]),
            stage=Stage(row["stage"]) if row["stage"] else None,
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def set_status(self, job_id, status: JobState, stage: Optional[Stage] = None,
                   error: Optional[str] = None):
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status=?, stage=?, error=?, updated_at=? WHERE job_id=?",
                (status.value, stage.value if stage else None, error, _now(), job_id),
            )
            if cur.rowcount == 0:
                raise KeyError(job_id)

    def next_queued(self) -> Optional[str]:
        with self._conn() as c:
            row = c.execute(
                "SELECT job_id FROM jobs WHERE status=? ORDER BY created_at ASC, rowid ASC LIMIT 1",
                (JobState.queued.value,),
            ).fetchone()
        return row["job_id"] if row else None

    def options(self, job_id) -> ReconstructOptions:
        with self._conn() as c:
            row = c.execute("SELECT options FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return ReconstructOptions.model_validate_json(row["options"])
