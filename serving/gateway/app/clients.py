import httpx

from .models import ReconstructOptions


class ServiceError(Exception):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


class _Base:
    def __init__(self, base_url: str, timeout_s: float, client=None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    def _healthz(self) -> bool:
        try:
            r = self._client.get(f"{self.base_url}/healthz", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False


class Sam3dClient(_Base):
    def healthz(self) -> bool:
        return self._healthz()

    def infer(self, job_id: str, input_dir: str) -> str:
        try:
            r = self._client.post(
                f"{self.base_url}/infer",
                json={"job_id": job_id, "input_dir": input_dir},
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as e:
            raise ServiceError("sam3d", f"sam3d-svc unreachable: {e}")
        if r.status_code != 200:
            raise ServiceError("sam3d", f"sam3d-svc returned {r.status_code}: {r.text}")
        return r.json()["mesh_path"]


class CadrilleClient(_Base):
    def healthz(self) -> bool:
        return self._healthz()

    def infer(self, job_id: str, mesh_path: str, options: ReconstructOptions) -> dict:
        try:
            r = self._client.post(
                f"{self.base_url}/infer",
                json={
                    "job_id": job_id,
                    "mesh_path": mesh_path,
                    "mode": options.mode.value,
                    "n_candidates": options.n_candidates,
                    "seed": options.seed,
                    "cleanup": options.cleanup,
                },
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as e:
            raise ServiceError("cadrille", f"cadrille-svc unreachable: {e}")
        if r.status_code != 200:
            raise ServiceError("cadrille", f"cadrille-svc returned {r.status_code}: {r.text}")
        return r.json()
