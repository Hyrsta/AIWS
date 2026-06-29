# Dockerized REST Service for the Offline CAD Reconstruction Pipeline

Date: 2026-06-29
Status: Approved design, pending implementation plan

## Goal

Package the offline CAD reconstruction pipeline (SAM3D then Cadrille) as a
Dockerized RESTful server. A remote device acts as the REST client: it submits
RGB image input over HTTP and receives the reconstructed CAD result back. The
two model stages run internally and are hidden behind a single public endpoint.

## Decisions captured from brainstorming

1. Public API surface: one full-pipeline call. The client POSTs image input and
   gets the final CAD result. It never calls SAM3D and Cadrille separately.
2. Host hardware: unknown, the client chooses. The design must be portable and
   must not assume a GPU count.
3. Job model: asynchronous. POST returns a job id immediately, the client polls
   for status, then fetches the result when done.
4. Cadrille mode: default PC mode with body-cleanup rerank (the canonical best,
   RL-PC). IMG mode is an optional per-request override.
5. Result bundle: generated CAD code, materialized STEP file, preview mesh and
   render, plus metrics and the intermediate SAM3D mesh.
6. Relationship to the existing GUI: this is a new standalone service. It reuses
   the existing orchestration logic but leaves the React GUI backend untouched.

## Architecture: split, not unified

Three containers wired by docker-compose, sharing one artifact volume on an
internal network. Only the gateway publishes a host port.

```
   Remote client (other device)
        |  REST + JSON over HTTP
        v
+---------------------------------------------+
|  aiws-gateway   (lightweight FastAPI, no ML)|   only public container
|   POST /v1/reconstruct      -> 202 {job_id} |
|   GET  /v1/jobs/{id}        -> status/stage |
|   GET  /v1/jobs/{id}/result -> CAD bundle   |
|   owns: job queue, orchestration, artifacts |
+------+----------------------------+---------+
       |  internal HTTP             |  internal HTTP
       v                            v
+------------------+        +--------------------------+
|  sam3d-svc       |        |  cadrille-svc            |
|  SAM3D conda env |        |  FROM cadrille:latest    |
|  MKL fix pinned  |        |  + thin FastAPI          |
|  image -> mesh   |        |  mesh -> CAD code / STEP |
+------------------+        |  + body-cleanup rerank   |
                            +--------------------------+
       (shared volume /artifacts for the mesh handoff)
```

### Why two services instead of one image

1. Dependency isolation is the decisive factor. Project history shows a real
   cross-contamination failure: the MKL and MoGe FFT regression, where a leaked
   LD_PRELOAD from one environment broke another. Two images mean SAM3D's MKL,
   BLAS, and MoGe stack and Cadrille's torch, open3d 0.18, and pythonocc stack
   can never collide. Unifying them forces re-solving both dependency trees in
   one image and stays fragile.
2. Cadrille is already a Docker image. The cadrille-svc container is
   FROM cadrille:latest plus a thin FastAPI wrapper around the existing
   cadrille_infer_wrapper.py. Unifying discards that.
3. Portability to unknown hardware. Each service reads its own device setting,
   so the same compose file runs on a single-GPU box (both map to cuda:0) or a
   multi-GPU box (distinct GPUs) with no code change. A per-service load mode
   (resident or on_demand) covers small versus large VRAM.
4. Independent rebuild, upgrade, and failure isolation. One model updates
   without rebuilding the other. A crash in one stage is reported by stage, not
   a total outage.

### Honest cost of splitting

Three containers instead of one, plus a mesh handoff between stages. The handoff
uses a shared Docker volume rather than base64 over HTTP, so multi-megabyte
meshes stay cheap. These costs are modest and justified by the dependency
fragility above. A single image would only win if the two stacks were compatible
and the GPU were guaranteed large, and neither holds here.

## Public API (gateway)

| Method | Path | Purpose |
|--------|------|---------|
| POST | /v1/reconstruct | multipart: 1..N images plus JSON options {mode, n_candidates, seed, cleanup}. Returns 202 {job_id, status: "queued"} |
| GET | /v1/jobs/{id} | {status: queued, running, succeeded, failed; stage: sam3d, cadrille, null; error?; created_at; updated_at} |
| GET | /v1/jobs/{id}/result | zip bundle, or per-artifact at /v1/jobs/{id}/result/{name} |
| GET | /healthz | 200 only when both downstream services report ready |

Request options and defaults:

- mode: "pc" (default) or "img". Default PC mode runs body-cleanup rerank.
- n_candidates: number of Cadrille candidates. Default follows the canonical
  decode configuration.
- seed: integer for reproducible sampling. Default 42.
- cleanup: boolean, default true.

### Result bundle layout

```
cad/model.py          generated CADQuery / Python source
cad/model.step        materialized B-rep exported to STEP
preview/model.stl     mesh of the reconstructed CAD
preview/render.png    rendered image of the reconstructed CAD
mesh/sam3d_mesh.ply   intermediate SAM3D mesh
metrics.json          candidate metrics and chosen-candidate info
manifest.json         job parameters, versions, artifact index
```

## Internal service contracts (not public)

These run on the internal compose network and exchange file paths on the shared
/artifacts volume.

sam3d-svc:
- POST /infer {job_id, input_dir} -> writes mesh into /artifacts/jobs/{id}/,
  returns {mesh_path, meta}.
- GET /healthz.

cadrille-svc:
- POST /infer {job_id, mesh_path, mode, n_candidates, seed, cleanup} -> writes
  CAD code, STEP, and preview into /artifacts/jobs/{id}/, returns
  {cad_code_path, step_path, preview_path, metrics}.
- GET /healthz.

## Orchestration and job lifecycle (gateway only)

- Job store: SQLite file on a persistent volume, so job metadata survives a
  gateway restart. Artifacts live under /artifacts/jobs/{id}/.
- Queue: FIFO, in-memory, with a single GPU-bound worker by default. Serial
  processing avoids GPU OOM. Worker count is configurable.
- Worker steps: validate input, write images to the job directory, call
  sam3d-svc, then call cadrille-svc, assemble the bundle, mark succeeded.
- The reference for stage parameters and ordering is the existing
  scripts/e2e_sam3d_to_cadrille.py. That orchestration logic moves into the
  gateway worker as two internal HTTP calls.

## Mesh handoff

A shared named volume mounted at /artifacts in all three containers, with a
per-job subdirectory /artifacts/jobs/{id}/. The gateway writes input images
there, sam3d-svc writes the mesh there, cadrille-svc reads the mesh and writes
its outputs there. No large payloads travel over HTTP.

## GPU and configuration (portable for unknown hardware)

Environment variables:

- SAM3D_DEVICE and CADRILLE_DEVICE, for example cuda:0. On a single-GPU box both
  map to cuda:0. On a multi-GPU box they can be distinct.
- MODEL_LOAD_MODE per service: resident keeps the model hot for speed,
  on_demand loads and unloads per request for small VRAM.
- Checkpoint paths for each model, mounted into the containers.
- WORKERS for the gateway, default 1.

Compose requests GPUs through the NVIDIA Container Toolkit. The toolkit is a
documented host prerequisite. The compose file does not hardcode a GPU count.

## Error handling

- Input validation at the gateway: file type, count, and size. Invalid input
  returns 400 before any job is created.
- Stage failure sets status to failed, records the stage that failed, and
  returns the error message. The gateway maps an internal 5xx into a failed job.
- Cadrille OCC materialization runs under a hard subprocess timeout, reusing the
  known OCC-deadlock-safe pattern, so a hang becomes a clean failure rather than
  a stuck job.
- Readiness: the gateway /healthz returns 503 while a downstream model is still
  loading, and 200 only when both services are ready.

## Packaging

- One Dockerfile per service.
- cadrille-svc: FROM cadrille:latest plus a thin FastAPI server reusing
  cadrille_infer_wrapper.py and the body-cleanup rerank.
- sam3d-svc: from the SAM3D conda base, with the MKL and LD_PRELOAD fix baked in
  and verified at build time so it cannot leak.
- gateway: from a slim python base, with no heavy ML dependencies.
- docker-compose.yml defines the three services, the shared artifact volume, the
  internal network, healthchecks, and the single published gateway port.
- A .env file holds configuration. A client README documents usage with curl
  examples.

## Testing

- Unit tests, no GPU: gateway job state machine, request validation, bundle
  assembly, manifest construction.
- Contract tests: mock sam3d-svc and cadrille-svc responses, verify orchestration
  ordering and error propagation.
- Integration smoke on a GPU box (RXL or GXD): one real image through the full
  pipeline, asserting the bundle contains every artifact and a valid STEP file.
- Health and readiness checks for all three containers.

## Out of scope (YAGNI)

- Authentication and TLS. Left to a deployment-time reverse proxy, noted as a
  deployment concern.
- Multi-GPU worker pools and horizontal scaling.
- Webhook callback delivery. Polling only for now.
- The online pipeline (YOLOv11-seg, GenPose++, FoundationPose).
- Rewiring the React GUI. The gateway is a new standalone service and the GUI is
  untouched. It can point at this API later.

## Where it lives

A new serving/ tree in the AIWS integration repo:

```
serving/
  gateway/
  sam3d-svc/
  cadrille-svc/
  docker-compose.yml
```

Service code lands via a PR to origin/main and builds on RXL, since the live
service code is not tracked in the local worktree. This design document lives in
the tracked docs/ tree.
