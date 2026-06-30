# Interactive mask refine for RGB-only mode

- Date: 2026-06-30
- Status: Design approved, pending spec review
- Branch / PR: `claude/mystifying-moore-6f426e` / [#36](https://github.com/Hyrsta/AIWS/pull/36)
- Related: builds on the Grounded-SAM image-only pipeline (`docs/superpowers/specs/2026-06-29-grounded-sam-image-only-pipeline-design.md`)

## Motivation

In RGB-only mode the GUI auto-segments the uploaded photo with Grounded-SAM and then jumps straight into the slow SAM3D plus Cadrille reconstruction (about 3.5 minutes). If the auto-mask grabs the wrong object (for example the welding torch instead of the workpiece) the user only finds out after the whole pipeline runs. They need a fast, interactive loop to get the mask right before committing to reconstruction: change the detection text, click points to include or exclude regions, or draw a box.

## Scope

In scope:
- A stateful segmentation session in `grounded-sam-svc` so repeated refinements are fast (the image is encoded into SAM once).
- Three refinement methods: text re-prompt, positive/negative point clicks, and a drag box.
- A GUI refine canvas in the RGB-only flow: auto-mask shown immediately, refine as much as wanted, then "Use this mask and reconstruct".

Out of scope:
- The headless REST gateway path stays as-is. Interactive refinement is a GUI feature; a non-interactive client cannot click, so it keeps using the existing stateless `/segment` auto-path.
- No changes to SAM3D or Cadrille. The refined mask feeds the existing `image_mask` reconstruction job unchanged.
- Multi-object / multi-mask composition. One workpiece, one mask.

## Architecture

Two units with a clean interface between them.

### Unit A: stateful segmentation session (`grounded-sam-svc`)

The slow part of SAM is `set_image` (the image encoder, about 0.5 to 1 second). Predicting a mask from points or a box, once the image is encoded, is tens of milliseconds. So we encode once per session and keep the embedding warm.

New endpoints (the existing stateless `POST /segment` is untouched):

- `POST /segment/session` (multipart: `image`, optional form field `prompt`)
  - Encodes the image into SAM, stores the predictor state under a generated `session_id`.
  - Runs the initial auto-segment (GroundingDINO with `prompt` or the configured default, take the highest-confidence box, SAM mask).
  - Returns `{ session_id, mask_png_base64, box: [x0,y0,x1,y1] | null, score, width, height, expires_at }`.
  - On no detection: returns an empty mask, `box: null`, and `detected: false` (not an error).

- `POST /segment/refine` (JSON)
  - Body: `{ session_id, prompt?: string, points?: [{x, y, label}], box?: [x0,y0,x1,y1], reset?: bool }` with `label` being `1` (include) or `0` (exclude); coordinates in original-image pixels.
  - Behavior:
    - If `prompt` is present: re-run GroundingDINO on the cached image, take the best box, and SAM-predict. This replaces the point/box accumulation (a fresh text detection is a fresh start).
    - Else: SAM-predict from the supplied `points` and/or `box`, feeding the session's previous low-resolution mask back as `mask_input` for iterative stability.
    - `reset: true` clears accumulated points/box/mask back to the post-session-create auto state.
  - Returns `{ mask_png_base64, score, width, height }`.

- `DELETE /segment/session/{session_id}` releases the session immediately.

Session store: an in-memory dict `session_id -> {predictor_state, image_size, last_low_res_mask, created_at, last_used_at}`, guarded by a lock (the service already serializes GPU work). A background sweeper evicts sessions idle longer than `SESSION_TTL_SECONDS` (default 900) and caps the count at `MAX_SESSIONS` (default 8), evicting least-recently-used. Eviction frees the cached embedding. A `DELETE` on an unknown id is a no-op 204; a `refine`/`session` call against an expired id returns `409 session_expired` so the GUI can transparently re-create the session from the image it still holds.

The existing single-shot `/segment` is refactored to share the GroundingDINO-detect and SAM-predict helpers with the session code, so detection logic lives in one place.

### Unit B: GUI refine canvas

In `ConfigureView` RGB-only mode, after the user uploads a photo:

1. The frontend POSTs the image to `/segment/session` (through the GUI backend, which proxies to grounded-sam-svc). It receives `session_id` plus the initial mask and renders the photo with the mask overlaid (semi-transparent tint) on an HTML canvas.
2. Controls below the canvas:
   - A **prompt** text field plus a **Re-detect** button (calls `/segment/refine` with `prompt`).
   - **Click to add points**: left-click adds an include point, right-click or Alt-click adds an exclude point; markers render as green (+) / red (-) dots. Each click calls `/segment/refine` with the accumulated points.
   - **Draw a box**: click-drag draws a rectangle; on release, `/segment/refine` is called with the box.
   - **Undo last** (drops the last point), **Clear points**, **Reset to auto** (`reset: true`).
3. A **Use this mask and reconstruct** button. It is enabled as soon as a mask exists (so accepting the auto-mask is effectively one click). On click, the frontend hands the current mask PNG to the existing reconstruction flow as a provided mask, and the GUI runs its normal `image_mask` job (no re-segmentation in the pipeline).

Coordinate mapping: the canvas may display the image scaled. The frontend converts click/drag coordinates from canvas space to original-image pixels before sending, and scales the returned mask to the display. All server-side coordinates are original-image pixels, so the contract is resolution-independent.

GUI backend role: thin proxy endpoints (`POST /segment/session`, `POST /segment/refine`, `DELETE /segment/session/{id}`) that forward to grounded-sam-svc and stream back JSON. This keeps the browser talking only to the GUI origin and reuses the existing service-client pattern. The final reconstruct call reuses the current `image_mask` job path, with the mask sourced from the refine canvas instead of an upload.

## Data flow

```
upload image
  -> GUI backend -> grounded-sam-svc POST /segment/session
       -> {session_id, initial auto-mask}
  -> canvas shows photo + mask
  -> [loop] user edits (prompt | points | box)
       -> GUI backend -> grounded-sam-svc POST /segment/refine
            -> {updated mask}  (overlay updates live)
  -> "Use this mask and reconstruct"
       -> GUI image_mask reconstruction job (image + final mask)
       -> SAM3D -> normalize -> Cadrille -> postscale -> results
  -> session released (DELETE, or TTL sweep)
```

## Error handling

- No detection on a prompt: keep the previous mask, surface a non-blocking hint ("no match for that prompt"). Never clear a good mask because a new prompt missed.
- Expired or unknown session on refine: GUI catches `409 session_expired`, silently re-creates the session from the image it still holds (replaying the current prompt), then retries the refine once.
- grounded-sam-svc unreachable: the canvas shows an error state and disables refine; the user can still fall back to the existing image+mask upload mode.
- The slow reconstruction is triggered only by the explicit button, never by a refine call, so accidental clicks cost milliseconds, not minutes.
- GPU pressure: `MAX_SESSIONS` plus LRU eviction bounds memory; a refine against an evicted session takes the `409` re-create path.

## Testing

Backend (pytest, SAM and GroundingDINO mocked so tests stay CPU-only and fast):
- session create returns an id and an initial mask; the default prompt path and an explicit prompt path.
- refine by points (include only, include plus exclude) changes the returned mask; coordinates round-trip in original-image pixels.
- refine by box; refine by re-prompt replaces accumulation; `reset` returns to auto state.
- TTL and MAX_SESSIONS eviction; refine against an expired id returns `409 session_expired`; `DELETE` is idempotent.
- the shared detect/predict helpers are exercised by both `/segment` and the session path.

Frontend (component test):
- canvas state machine: point accumulation, include/exclude labelling, undo/clear/reset, box draw.
- canvas-to-image coordinate mapping at a non-unit display scale.
- "Use this mask" routes the current mask into the reconstruction job.

On-RXL manual validation (host-process deployment):
- a refine click updates the mask overlay in well under one second.
- the torch-vs-workpiece case: an exclude point or a tighter prompt removes the torch from the mask, and the subsequent reconstruction uses the corrected mask.

## Deployment notes

- Lands on PR #36, additive. New endpoints and a new frontend panel; no deletions.
- grounded-sam-svc runs as a host process on RXL today (port 18091). The session store is in-process, which matches the single-process deployment. If the service is ever scaled to multiple workers, sessions would need sticky routing or a shared store; called out as future work, not built now.
- No new heavy dependencies: SAM point/box prediction uses the already-loaded `SamPredictor`.

## Future work (not in this spec)

- "Pick among detections" (show all GroundingDINO boxes, click the right one).
- Brush/scribble refinement.
- Persisting the chosen mask alongside the job record for later re-runs.
- Multi-worker session affinity for grounded-sam-svc.
