import { describe, it, expect } from "vitest";
import { canvasToImage } from "./SegmentRefineCanvas";

describe("canvasToImage", () => {
  it("maps canvas coords to image pixels at unit scale", () => {
    expect(canvasToImage(10, 20, 1)).toEqual({ x: 10, y: 20 });
  });
  it("scales when the image is displayed smaller", () => {
    // image displayed at half size: scale = 0.5 -> divide by scale
    expect(canvasToImage(10, 20, 0.5)).toEqual({ x: 20, y: 40 });
  });
  it("rounds to whole pixels", () => {
    expect(canvasToImage(11, 21, 0.5)).toEqual({ x: 22, y: 42 });
  });
});
