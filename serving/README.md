# AIWS CAD Reconstruction REST Service

Submit RGB images, get a CAD reconstruction bundle back. The SAM3D then Cadrille
pipeline runs server-side as one async job. Only the gateway is exposed; the two
model services are internal.

## Host prerequisites

- Docker and docker-compose.
- NVIDIA Container Toolkit, so containers can reach the GPU.
- The cadrille and sam3d base images built on the host (see each service Dockerfile).
- Model checkpoints on the host, referenced by SAM3D_CKPT_HOST and CADRILLE_CKPT_HOST.

## Configure and run

```bash
cd serving
cp .env.example .env
# edit .env: set GATEWAY_PORT, device per GPU layout, checkpoint paths
docker compose up --build -d
curl -fsS http://localhost:8080/healthz
```

On a single-GPU host, set both SAM3D_DEVICE and CADRILLE_DEVICE to cuda:0. On a
multi-GPU host, set them to distinct devices.

## API

### Submit a job

```bash
curl -s -X POST http://localhost:8080/v1/reconstruct \
  -F "images=@view0.png" \
  -F "images=@view1.png" \
  -F "mode=pc" \
  -F "n_candidates=20" \
  -F "seed=42" \
  -F "cleanup=true"
```

Response:

```json
{ "job_id": "ab12cd...", "status": "queued" }
```

Options: mode is pc (default, best) or img. n_candidates default 20. seed default
42. cleanup default true.

### Poll status

```bash
curl -s http://localhost:8080/v1/jobs/ab12cd...
```

```json
{ "job_id": "ab12cd...", "status": "running", "stage": "cadrille", "error": null }
```

status is queued, running, succeeded, or failed. stage is sam3d or cadrille while
running, and identifies the failing stage when status is failed.

### Fetch the result

```bash
curl -s -o result.zip http://localhost:8080/v1/jobs/ab12cd.../result
```

Returns 409 until the job has succeeded. The zip contains:

- cad/model.py: generated CAD code
- cad/model.step: materialized STEP
- preview/model.stl and preview/render.png: preview of the reconstruction
- mesh/sam3d_mesh.ply: intermediate SAM3D mesh
- metrics.json: candidate metrics and chosen-candidate info
- manifest.json: job parameters and artifact index

## Notes

- Jobs run one at a time by default to serialize GPU use. Raise WORKERS only if
  the host has the VRAM to run both stages concurrently.
- Authentication and TLS are out of scope here. Put the gateway behind a reverse
  proxy for any non-trusted network.
