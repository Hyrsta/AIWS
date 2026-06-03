export const BASE_STAGES = [
  "SAM3D: Loading checkpoints",
  "SAM3D: Generating mesh",
  "Cadrille: Preparing input",
  "Cadrille: Generating CAD result",
] as const;
export const POSTSCALE_STAGE = "Post-scaling: Aligning CAD to catalog (mm)";
export const CLEANUP_STAGE = "Body cleanup: Removing hallucinated bodies";
export const HIDDEN_CLEANUP_STAGE = CLEANUP_STAGE;

export function visibleStages(hasPostscale: boolean): string[] {
  return hasPostscale
    ? [...BASE_STAGES, CLEANUP_STAGE, POSTSCALE_STAGE]
    : [...BASE_STAGES, CLEANUP_STAGE];
}

/** Index of the currently-active visible step for a given backend stage_label. */
export function stageToStep(stageLabel: string, hasPostscale: boolean): number {
  const stages = visibleStages(hasPostscale);
  const i = stages.indexOf(stageLabel);
  if (i >= 0) return i;
  return stages.length - 1; // completed / terminal / unknown
}

// Coarse backend `stage` key -> first visible step of that group. Fallback only;
// stage_timings is the precise signal.
const STAGE_KEY_TO_STEP: Record<string, number> = {
  queued: 0,
  sam3d: 0,
  cadrille: 2,
  body_cleanup: 4,
  postscale: 5,
};

/**
 * Index of the visible step that FAILED. On failure the backend overwrites
 * `stage_label` to the literal "Failed", so stageToStep() would wrongly fall
 * through to the last step (Post-scale). Derive the real failing step from the
 * furthest stage that actually started (`stage_timings`), falling back to the
 * coarse `stage` key.
 */
export function failedStep(
  stageTimings: Record<string, unknown> | null | undefined,
  stageKey: string,
  hasPostscale: boolean,
): number {
  const stages = visibleStages(hasPostscale);
  let idx = -1;
  if (stageTimings) {
    for (let i = 0; i < stages.length; i++) {
      if (stageTimings[stages[i]] != null) idx = i;
    }
  }
  if (idx >= 0) return idx;
  const k = STAGE_KEY_TO_STEP[stageKey];
  return k != null ? Math.min(k, stages.length - 1) : 0;
}
