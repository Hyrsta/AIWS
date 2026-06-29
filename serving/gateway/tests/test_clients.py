import httpx
import pytest
from app.clients import Sam3dClient, CadrilleClient, ServiceError, GroundedSamClient
from app.models import ReconstructOptions


def client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://svc")


def test_sam3d_infer_returns_mesh_path():
    def handler(request):
        assert request.url.path == "/infer"
        return httpx.Response(200, json={"mesh_path": "/artifacts/jobs/x/mesh/sam3d_mesh.ply"})
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    assert c.infer("x", "/artifacts/jobs/x/input", "/artifacts/jobs/x/input/mask.png") == "/artifacts/jobs/x/mesh/sam3d_mesh.ply"


def test_sam3d_infer_raises_service_error_on_500():
    def handler(request):
        return httpx.Response(500, text="model crashed")
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    with pytest.raises(ServiceError) as ei:
        c.infer("x", "/in", "/in/mask.png")
    assert ei.value.stage == "sam3d"


def test_cadrille_infer_passes_options_and_parses_result():
    def handler(request):
        body = request.read().decode()
        assert '"mode": "pc"' in body or '"mode":"pc"' in body
        return httpx.Response(200, json={
            "cad_code_path": "/a/cad/model.py",
            "step_path": "/a/cad/model.step",
            "preview_path": "/a/preview/model.stl",
            "metrics": {"iou": 0.22},
        })
    c = CadrilleClient("http://svc", 5, client=client_with(handler))
    out = c.infer("x", "/a/mesh/sam3d_mesh.ply", ReconstructOptions())
    assert out["metrics"]["iou"] == 0.22


def test_sam3d_infer_raises_service_error_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("down")
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    with pytest.raises(ServiceError) as ei:
        c.infer("x", "/in", "/in/mask.png")
    assert ei.value.stage == "sam3d"


def test_cadrille_infer_raises_service_error_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("down")
    c = CadrilleClient("http://svc", 5, client=client_with(handler))
    with pytest.raises(ServiceError) as ei:
        c.infer("x", "/mesh.ply", ReconstructOptions())
    assert ei.value.stage == "cadrille"


def test_healthz_false_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("down")
    c = CadrilleClient("http://svc", 5, client=client_with(handler))
    assert c.healthz() is False


def test_grounded_sam_segment_parses_mask(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    def handler(request):
        assert request.url.path == "/segment"
        return httpx.Response(200, json={
            "mask_png_base64": "AAAA", "score": 0.5,
            "box": [1, 2, 3, 4], "label": "workpiece", "num_detections": 1})
    c = GroundedSamClient("http://svc", 5, client=client_with(handler))
    out = c.segment(str(img), "workpiece")
    assert out["mask_png_base64"] == "AAAA"
    assert out["score"] == 0.5


def test_grounded_sam_segment_422_raises_segment_stage(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    def handler(request):
        return httpx.Response(422, text="no object matched")
    c = GroundedSamClient("http://svc", 5, client=client_with(handler))
    with pytest.raises(ServiceError) as ei:
        c.segment(str(img), "workpiece")
    assert ei.value.stage == "segment"


def test_sam3d_infer_sends_mask_path():
    seen = {}
    def handler(request):
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"mesh_path": "/a/mesh/sam3d_mesh.ply"})
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    c.infer("x", "/a/input", "/a/input/mask.png")
    assert "mask.png" in seen["body"]
