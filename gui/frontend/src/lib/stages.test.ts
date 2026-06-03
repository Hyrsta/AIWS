import { describe, it, expect } from "vitest";
import { visibleStages, stageToStep, failedStep } from "./stages";
describe("stages", () => {
  it("4 steps without postscale, 5 with", () => {
    expect(visibleStages(false)).toHaveLength(4);
    expect(visibleStages(true)).toHaveLength(5);
  });
  it("maps a visible stage label to its index", () => {
    expect(stageToStep("Cadrille: Generating CAD result", true)).toBe(3);
  });
  it("maps the hidden cleanup stage to just-after-Cadrille (3)", () => {
    expect(stageToStep("Body cleanup: Removing hallucinated bodies", true)).toBe(3);
  });
  it("completed/unknown → last index", () => {
    expect(stageToStep("completed", true)).toBe(4);
    expect(stageToStep("completed", false)).toBe(3);
  });
  // Regression: a SAM3D failure must NOT be reported as the last (Post-scale) step.
  it("failedStep uses stage_timings to find the real failing step", () => {
    expect(failedStep({ "SAM3D: Loading checkpoints": {} }, "sam3d", true)).toBe(0);
    expect(
      failedStep(
        { "SAM3D: Loading checkpoints": {}, "SAM3D: Generating mesh": {} },
        "sam3d",
        true,
      ),
    ).toBe(1);
  });
  it("failedStep falls back to the stage key when stage_timings is empty", () => {
    expect(failedStep({}, "sam3d", true)).toBe(0);
    expect(failedStep(null, "cadrille", true)).toBe(2);
    expect(failedStep(undefined, "postscale", true)).toBe(4);
    expect(failedStep({}, "postscale", false)).toBe(3);
  });
});
