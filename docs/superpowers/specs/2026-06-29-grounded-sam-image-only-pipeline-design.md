# Grounded-SAM Auto-Segmentation and Image-Only Pipeline

Date: 2026-06-29
Status: Approved design, pending implementation plan

## Goal

Let a user submit an RGB image alone (no mask) and still get a CAD
reconstruction. SAM3D needs a per-object mask, so the service auto-generates one
with Grounded-SAM (GroundingDINO for text-prompted detection, SAM for the mask).
This image-only path is exposed in both the new REST service and the existing
React GUI. The same work also lands the SAM3D and Cadrille adapters that the REST
pipeline needs to run end to end.

## Background

- The REST service from `docs/superpowers/specs/2026-06-29-dockerized-rest-pipeline-design.md`
  (PR #36) wraps SAM3D then Cadrille behind one async endpoint. It was built and
  unit-tested on CPU; the two model-service images and the SAM3D/Cadrille adapter
  CLIs were deferred to the RXL GPU host.
- Reading the real `scripts/` and `gui/backend/` on origin/main showed that
  SAM3D is mask-driven: `sam3d_batch.py` builds a mask from a per-object polygon
  and calls the SAM3D repo `inference(image, mask, seed)`. A bare RGB image is
  not enough.
- The GUI already supports `image_mask` and `mesh` input modes via
  `gui/backend/app.py` `POST /jobs/simple-reconstruct`, run by
  `gui/backend/simple_reconstruct_job.py`. That job is a proven single-sample
  pipeline: image plus mask to SAM3D `inference`, normalize the STL to the unit
  cube, run Cadrille inside `cadrille:latest`, postscale, then emit the selected
  CadQuery code, STEP, STL, preview, and metrics.

## Decisions captured from brainstorming

1. Segmentation runs in a dedicated `grounded-sam-svc` container, reused by both
   the REST gateway and the GUI backend over HTTP.
2. GroundingDINO uses a fixed default text prompt with optional per-call
   override. Default `WORKPIECE_PROMPT` is `workpiece. metal part.`.
3. Image-only is the default REST path (`segment=auto`); a client may still
   provide a mask (`segment=provided`).
4. The GUI gains an RGB-only input mode that auto-segments, then runs the
   existing `image_mask` job unchanged.
5. The SAM3D and Cadrille adapters (`run_sam3d.py`, `run_cadrille.py`) are
   bundled into this work and reuse the proven functions in
   `simple_reconstruct_job.py` rather than reimplementing the pipeline.
6. Everything is validated end to end on the RXL GPU host.

## Architecture

Four containers wired by docker-compose. The gateway and `grounded-sam-svc` are
published on host ports; `sam3d-svc` and `cadrille-svc` stay internal.

```
   Remote client (RGB image, no mask)
        |  REST + JSON
        v
+--------------------------------------------------+
|  aiws-gateway (FastAPI)                           |
|   POST /v1/reconstruct (segment=auto default)     |
|   orchestrator stages: segment, sam3d, cadrille   |
+---+--------------+-------------------+------------+
    | seg (HTTP)   | sam3d (HTTP)      | cadrille (HTTP)
    v              v                   v
+-----------+  +-----------+   +---------------------+
| grounded- |  | sam3d-svc |   | cadrille-svc        |
| sam-svc   |  | image+    |   | mesh -> CAD/STEP    |
| img->mask |  | mask->mesh|   | + postscale         |
+-----------+  +-----------+   +---------------------+
     ^   shared /artifacts volume across the model services
     |
+-----------------------------+
|  GUI backend (on RXL, tmux) | --calls /segment for RGB-only mode-->
+-----------------------------+
```

The GUI backend runs on the same host but outside the compose stack, so it
reaches `grounded-sam-svc` through its published host port.

## Component: grounded-sam-svc

A FastAPI service wrapping GroundingDINO plus SAM.

- `POST /segment`: multipart `image` plus optional form fields `prompt`,
  `box_threshold` (default 0.3), `text_threshold` (default 0.25). Runs
  GroundingDINO to detect boxes for the prompt, picks the highest-confidence box
  above `box_threshold`, runs SAM on that box, returns JSON:
  `{ mask_png_base64, score, box: [x0,y0,x1,y1], label, num_detections }`.
  If no detection clears the threshold, returns `422` with
  `{ detail: "no object matched prompt '<prompt>'" }`.
- `GET /healthz`: 200 when the models are loaded, 503 while loading.
- The mask is returned in the response body, not via the shared volume, so both
  the gateway and the GUI backend (which does not share the volume) can consume
  it. Masks are small.
- Config via env: `WORKPIECE_PROMPT` (default `workpiece. metal part.`),
  `GROUNDING_DINO_CKPT`, `SAM_CKPT`, `SAM_MODEL_TYPE`, `SEG_DEVICE`. Weights are
  mounted like the other checkpoints.
- Dockerfile builds from a torch base, installs GroundingDINO and segment-anything,
  copies the app, and starts uvicorn via the `create_app` factory.

Selection rule: a single highest-confidence detection becomes one mask. The
single-workpiece scene does not need multi-object merging (YAGNI).

## REST pipeline changes (serving/)

- `ReconstructOptions` gains `segment: "auto" | "provided"` (default `auto`) and
  `detect_prompt: Optional[str]` (overrides the service default when set).
- `Stage` enum gains `segment`, ordered before `sam3d`.
- A `GroundedSamClient` (mirrors the existing service clients) calls
  `grounded-sam-svc /segment` and returns the mask bytes plus detection metadata;
  transport and non-200 errors map to a `ServiceError(stage="segment")`.
- Orchestrator: when `segment=auto`, the worker runs the `segment` stage first,
  writes the returned mask to `<job_dir>/input/mask.png`, then runs `sam3d`.
  When `segment=provided`, the segment stage is skipped and the client-supplied
  mask is used.
- `POST /v1/reconstruct` accepts an optional `mask` upload (used only when
  `segment=provided`) and the new options.
- `sam3d-svc /infer` contract gains a `mask_path`; the service runs SAM3D
  `inference(image, mask, seed)` for the single sample.
- The result bundle and manifest record the mask used and, for the auto path,
  the detection score and prompt, for traceability. `bundle.ARTIFACT_NAMES`
  gains `mesh/auto_mask.png` (present only on the auto path).

## SAM3D and Cadrille adapters

The two model services shell out to adapter CLIs. Both adapters reuse functions
from `simple_reconstruct_job.py` rather than reimplementing the pipeline, so the
proven single-sample logic is shared.

- `run_sam3d.py` (used by `sam3d-svc`): inputs `--input-image`, `--input-mask`,
  `--seed`, `--out-mesh`. Runs SAM3D `inference(image, mask, seed)` and writes
  the reconstructed mesh (STL). Reuses the SAM3D-determinism setup and the
  inference call from `simple_reconstruct_job.py`.
- `run_cadrille.py` (used by `cadrille-svc`): inputs `--mesh`, `--mode`,
  `--n-candidates`, `--seed`, `--ckpt`, `--device`, `--out-dir`, `--cleanup`.
  Normalizes the STL to the unit cube, runs Cadrille (the existing
  `run_cadrille_on_split` path or its host equivalent), runs postscale, and
  emits the canonical bundle layout: `cad/model.py`, `cad/model.step`,
  `preview/model.stl`, `preview/render.png`, `metrics.json`. Reuses the
  normalize, Cadrille invocation, and postscale logic from
  `simple_reconstruct_job.py`.

Both adapters live in the AIWS repo and are copied into their service images. The
factored shared logic lives in a small module both the GUI job and the adapters
import, so there is one implementation of each pipeline step.

## GUI changes (gui/)

- Backend `gui/backend/app.py`: `create_simple_reconstruct` gains
  `input_mode="image"`. On that mode the backend calls `grounded-sam-svc /segment`
  with the uploaded image, saves the returned mask next to the image, then runs
  the existing `image_mask` job path unchanged. Adds the seg service URL to
  config and includes it in the `/health` probe.
- Frontend `gui/frontend/src/components/InputImagesPanel.tsx`: add an
  "RGB only (auto-segment)" option to the existing input-mode selector. When
  selected, hide the mask uploader and show an optional advanced "detection
  prompt" field. Optionally preview the generated mask once segmentation
  returns.

The `image_mask` and `mesh` modes are unchanged.

## Data flow (image-only)

```
client -> gateway POST /v1/reconstruct (image, segment=auto)
       -> [segment] grounded-sam-svc -> mask
       -> [sam3d]   sam3d-svc(image, mask) -> mesh
       -> [cadrille] cadrille-svc(mesh) -> cad/step/preview/metrics
       -> bundle (zip) including the auto mask and detection score
```

GUI: user selects RGB only -> backend calls grounded-sam-svc -> mask ->
existing `simple_reconstruct_job` (image plus mask) -> results.

## Error handling

- No detection above threshold: `grounded-sam-svc` returns 422; the gateway
  marks the job `failed` at stage `segment` with the message
  "no object matched prompt '<prompt>'; try another prompt or provide a mask".
  The GUI surfaces the same message.
- `grounded-sam-svc` unreachable: gateway `/healthz` returns 503; an in-flight
  job fails at stage `segment`.
- Existing SAM3D and Cadrille failure handling (hard subprocess timeout,
  per-stage failure recording) is unchanged.

## Testing

- `grounded-sam-svc`: contract tests with a mocked segmentor (detect to mask),
  `/healthz`, and the no-detection to 422 path. The real model runs on RXL.
- Gateway: orchestrator test for the new `segment` stage with a mocked
  `GroundedSamClient`; an option test that `segment` defaults to `auto`; a
  `segment=provided` path that skips the seg call.
- `sam3d-svc`: contract test that `/infer` accepts and forwards `mask_path`.
- GUI backend: `input_mode="image"` calls the seg service then runs the job
  (mocked seg).
- RXL integration smoke: image-only request to a full bundle with a valid STEP
  and the auto mask present.

## Deployment and sequencing (RXL)

1. Provide GroundingDINO and SAM weights on the host and mount them.
2. Build the `grounded-sam-svc`, `sam3d-svc`, and `cadrille-svc` images.
3. Bring up the compose stack; run the image-only integration smoke.
4. Deploy the GUI: backend reads the seg service URL; rebuild the frontend
   bundle; the GUI image-only mode is live.

## Out of scope (YAGNI)

- Multi-object segmentation and mask merging.
- Interactive mask editing in the GUI.
- Replacing the GUI job runner with the REST gateway. The GUI keeps its own
  pipeline and only adds the seg call.
- Authentication and TLS (a deployment-time reverse proxy concern).
