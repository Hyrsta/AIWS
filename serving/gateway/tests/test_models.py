import pytest
from pydantic import ValidationError
from app.models import ReconstructOptions, Mode, JobState

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
