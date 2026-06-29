import os
from app.jobstore import JobStore
from app.models import ReconstructOptions, JobState, Stage, Mode


def make_store(tmp_path):
    return JobStore(os.path.join(tmp_path, "jobs.db"))


def test_create_and_get_roundtrip(tmp_path):
    s = make_store(tmp_path)
    jid = s.create(ReconstructOptions(mode=Mode.img, n_candidates=8))
    job = s.get(jid)
    assert job.job_id == jid
    assert job.status == JobState.queued
    assert job.stage is None


def test_options_persisted(tmp_path):
    s = make_store(tmp_path)
    jid = s.create(ReconstructOptions(mode=Mode.img, n_candidates=8, seed=7))
    opts = s.options(jid)
    assert opts.mode == Mode.img
    assert opts.n_candidates == 8
    assert opts.seed == 7


def test_status_transition_records_stage_and_error(tmp_path):
    s = make_store(tmp_path)
    jid = s.create(ReconstructOptions())
    s.set_status(jid, JobState.failed, stage=Stage.cadrille, error="boom")
    job = s.get(jid)
    assert job.status == JobState.failed
    assert job.stage == Stage.cadrille
    assert job.error == "boom"


def test_next_queued_is_fifo(tmp_path):
    s = make_store(tmp_path)
    a = s.create(ReconstructOptions())
    b = s.create(ReconstructOptions())
    assert s.next_queued() == a
    s.set_status(a, JobState.running, stage=Stage.sam3d)
    assert s.next_queued() == b


def test_get_missing_returns_none(tmp_path):
    s = make_store(tmp_path)
    assert s.get("nope") is None


def test_persists_across_reopen(tmp_path):
    path = os.path.join(tmp_path, "jobs.db")
    jid = JobStore(path).create(ReconstructOptions())
    assert JobStore(path).get(jid).job_id == jid
