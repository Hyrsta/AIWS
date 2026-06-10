"""Unit tests for the backend's host/path/job-id safety helpers.

Pure-python: imports gui.backend.app (FastAPI + numpy + trimesh deps only),
no network, docker, or filesystem side effects, so it runs in CI.
"""
from pathlib import Path

import pytest
from fastapi import HTTPException

from gui.backend import app


class TestSshArgv:
    def test_inserts_separator_before_host(self):
        assert app.ssh_argv("RXL", "bash", "-s") == ["ssh", "--", "RXL", "bash", "-s"]

    def test_option_like_host_is_neutralized_into_a_positional(self):
        # Without the "--" separator ssh would parse this as an option
        # (argument injection -> RCE). With it, the value lands after "--"
        # so ssh treats it as a hostname (and rejects it) instead.
        argv = app.ssh_argv("-oProxyCommand=touch /tmp/pwned", "bash", "-s")
        assert argv[:2] == ["ssh", "--"]
        assert argv[2] == "-oProxyCommand=touch /tmp/pwned"

    def test_scp_argv_separator(self):
        assert app.scp_argv("/src", "RXL:/dst") == ["scp", "--", "/src", "RXL:/dst"]


class TestValidateJobId:
    @pytest.mark.parametrize("job_id", ["abc123", "gui-simple-20260101_120000", "job.1_v2"])
    def test_accepts_plain_identifiers(self, job_id):
        assert app.validate_job_id(job_id) == job_id

    @pytest.mark.parametrize("job_id", ["../etc/passwd", "a/b", "-rf", "", "with space", "x;rm"])
    def test_rejects_path_and_option_payloads(self, job_id):
        with pytest.raises(HTTPException) as exc:
            app.validate_job_id(job_id)
        assert exc.value.status_code == 400

    def test_job_path_stays_under_jobs_root(self):
        p = app.job_path("abc123")
        assert p.parent == app.JOBS_ROOT
        assert p.name == "abc123.json"


class TestIsUnderAllowedRoot:
    ALLOWED = ("/data/outputs/gui-simple", "/data/outputs/gui-jobs")

    def test_accepts_the_root_itself_and_children(self):
        assert app.is_under_allowed_root(Path("/data/outputs/gui-simple"), self.ALLOWED)
        assert app.is_under_allowed_root(Path("/data/outputs/gui-simple/job1"), self.ALLOWED)

    def test_rejects_sibling_sharing_a_string_prefix(self):
        # The bug this guards against: a bare startswith() would accept this.
        assert not app.is_under_allowed_root(Path("/data/outputs/gui-simple-backup"), self.ALLOWED)

    def test_rejects_unrelated_paths(self):
        assert not app.is_under_allowed_root(Path("/etc"), self.ALLOWED)
        assert not app.is_under_allowed_root(Path("/data/outputs"), self.ALLOWED)
