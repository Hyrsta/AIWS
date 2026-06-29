import os
import time
from app.jobstore import JobStore
from app.worker import Worker
from app.models import ReconstructOptions, JobState


class SucceedOrchestrator:
    def __init__(self, store):
        self.store = store

    def run(self, job_id):
        self.store.set_status(job_id, JobState.succeeded, stage=None)


class CrashThenSucceedOrchestrator:
    def __init__(self, store):
        self.store = store
        self.calls = 0

    def run(self, job_id):
        self.calls += 1
        # mark running first, mimicking the real orchestrator contract that a
        # job leaves the queued state before any failure
        self.store.set_status(job_id, JobState.running)
        if self.calls == 1:
            raise RuntimeError("boom")
        self.store.set_status(job_id, JobState.succeeded, stage=None)


def wait_until(fn, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.01)
    return False


def test_worker_processes_queued_job(tmp_path):
    store = JobStore(os.path.join(tmp_path, "jobs.db"))
    jid = store.create(ReconstructOptions())
    w = Worker(store, SucceedOrchestrator(store), poll_interval_s=0.01)
    w.start()
    try:
        assert wait_until(lambda: store.get(jid).status == JobState.succeeded)
    finally:
        w.stop()


def test_worker_survives_orchestrator_exception(tmp_path):
    store = JobStore(os.path.join(tmp_path, "jobs.db"))
    bad = store.create(ReconstructOptions())
    w = Worker(store, CrashThenSucceedOrchestrator(store), poll_interval_s=0.01)
    w.start()
    try:
        assert wait_until(lambda: store.get(bad).status == JobState.running)
        good = store.create(ReconstructOptions())
        assert wait_until(lambda: store.get(good).status == JobState.succeeded)
    finally:
        w.stop()
