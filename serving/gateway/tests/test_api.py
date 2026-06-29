import io
import os
from fastapi.testclient import TestClient
from app.main import build_app
from app.jobstore import JobStore
from app.config import Settings
from app.models import JobState, Stage


class StubWorker:
    def start(self): pass
    def stop(self): pass
    def is_alive(self): return True


class DeadWorker:
    def start(self): pass
    def stop(self): pass
    def is_alive(self): return False


class StubHealth:
    def __init__(self, ok): self.ok = ok
    def healthz(self): return self.ok


def make_client(tmp_path, gsam_ok=True, sam_ok=True, cad_ok=True):
    settings = Settings(
        sam3d_url="http://sam3d", cadrille_url="http://cadrille",
        grounded_sam_url="http://gsam",
        artifacts_dir=str(tmp_path), db_path=os.path.join(tmp_path, "jobs.db"),
        workers=1, stage_timeout_s=10, max_images=4,
    )
    store = JobStore(settings.db_path)
    app = build_app(settings, store, StubHealth(gsam_ok), StubHealth(sam_ok),
                    StubHealth(cad_ok), StubWorker())
    return TestClient(app), store


def png_bytes():
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 16)


def test_livez_ok(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/livez").status_code == 200


def test_reconstruct_enqueues_and_returns_job_id(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert store.get(jid).status == JobState.queued
    assert os.path.exists(os.path.join(str(tmp_path), "jobs", jid, "input"))


def test_reconstruct_rejects_no_images(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/v1/reconstruct", data={"mode": "pc"})
    assert r.status_code == 422  # FastAPI required-field validation


def test_reconstruct_rejects_bad_mode(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "solid"})
    assert r.status_code == 400


def test_reconstruct_rejects_too_many_images(tmp_path):
    client, _ = make_client(tmp_path)
    files = [("images", (f"{i}.png", png_bytes(), "image/png")) for i in range(5)]
    r = client.post("/v1/reconstruct", files=files, data={"mode": "pc"})
    assert r.status_code == 400


def test_get_job_404_for_unknown(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/v1/jobs/nope").status_code == 404


def test_result_409_when_not_done(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    jid = r.json()["job_id"]
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 409


def test_result_returns_zip_when_succeeded(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    jid = r.json()["job_id"]
    zpath = os.path.join(str(tmp_path), "jobs", jid, f"{jid}.zip")
    open(zpath, "wb").write(b"PK\x03\x04zip")
    store.set_status(jid, JobState.succeeded, stage=None)
    rr = client.get(f"/v1/jobs/{jid}/result")
    assert rr.status_code == 200
    assert rr.headers["content-type"] == "application/zip"


def test_healthz_503_when_downstream_down(tmp_path):
    client, _ = make_client(tmp_path, cad_ok=False)
    assert client.get("/healthz").status_code == 503


def test_reconstruct_rejects_non_image_content_type(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.txt", io.BytesIO(b"hello"), "text/plain")},
                    data={"mode": "pc"})
    assert r.status_code == 400


def test_healthz_503_when_worker_dead(tmp_path):
    settings = Settings(
        sam3d_url="http://sam3d", cadrille_url="http://cadrille",
        grounded_sam_url="http://gsam",
        artifacts_dir=str(tmp_path), db_path=os.path.join(tmp_path, "jobs.db"),
        workers=1, stage_timeout_s=10, max_images=4,
    )
    store = JobStore(settings.db_path)
    app = build_app(settings, store, StubHealth(True), StubHealth(True), StubHealth(True),
                    DeadWorker())
    client = TestClient(app)
    assert client.get("/healthz").status_code == 503


def test_reconstruct_defaults_segment_auto(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png")},
                    data={"mode": "pc"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert store.options(jid).segment.value == "auto"


def test_reconstruct_provided_saves_mask(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/v1/reconstruct",
                    files={"images": ("a.png", png_bytes(), "image/png"),
                           "mask": ("m.png", png_bytes(), "image/png")},
                    data={"mode": "pc", "segment": "provided"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    import os
    assert os.path.exists(os.path.join(str(tmp_path), "jobs", jid, "input", "mask.png"))


def test_healthz_503_when_grounded_sam_down(tmp_path):
    client, _ = make_client(tmp_path, gsam_ok=False)
    assert client.get("/healthz").status_code == 503
