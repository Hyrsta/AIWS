import pytest
from pydantic import ValidationError
from app.models import ReconstructOptions, Mode, JobState, SegmentMode, Stage

def test_defaults_match_canonical():
    o = ReconstructOptions()
    assert o.mode == Mode.pc
    assert o.n_candidates == 20
    assert o.seed == 42
    assert o.cleanup is True

def test_n_candidates_lower_bound_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(n_candidates=0)

def test_n_candidates_upper_bound_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(n_candidates=65)

def test_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(mode="solid")

def test_job_state_values():
    assert JobState.queued.value == "queued"
    assert JobState.succeeded.value == "succeeded"


def test_segment_defaults_to_auto():
    o = ReconstructOptions()
    assert o.segment == SegmentMode.auto
    assert o.detect_prompt is None


def test_segment_provided_accepted():
    o = ReconstructOptions(segment="provided")
    assert o.segment == SegmentMode.provided


def test_invalid_segment_rejected():
    with pytest.raises(ValidationError):
        ReconstructOptions(segment="magic")


def test_stage_has_segment():
    assert Stage.segment.value == "segment"
