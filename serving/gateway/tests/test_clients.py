import httpx
import pytest
from app.clients import Sam3dClient, CadrilleClient, ServiceError
from app.models import ReconstructOptions


def client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://svc")


def test_sam3d_infer_returns_mesh_path():
    def handler(request):
        assert request.url.path == "/infer"
        return httpx.Response(200, json={"mesh_path": "/artifacts/jobs/x/mesh/sam3d_mesh.ply"})
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    assert c.infer("x", "/artifacts/jobs/x/input") == "/artifacts/jobs/x/mesh/sam3d_mesh.ply"


def test_sam3d_infer_raises_service_error_on_500():
    def handler(request):
        return httpx.Response(500, text="model crashed")
    c = Sam3dClient("http://svc", 5, client=client_with(handler))
    with pytest.raises(ServiceError) as ei:
        c.infer("x", "/in")
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


def test_healthz_false_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("down")
    c = CadrilleClient("http://svc", 5, client=client_with(handler))
    assert c.healthz() is False
