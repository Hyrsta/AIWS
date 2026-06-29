import { describe, it, expect } from "vitest";
import { fmtIoU, fmtCd, fmtMm, relTime } from "./format";
describe("formatters", () => {
  it("IoU pct", () => expect(fmtIoU(0.5476700730233248)).toBe("54.77%"));
  it("Chamfer sci", () => expect(fmtCd(0.0020117831832847377)).toBe("2.012e-3"));
  it("mm 2dp", () => expect(fmtMm(150.0000002)).toBe("150.00"));
  it("relTime seconds", () => expect(relTime(10, 3)).toBe("7s ago"));
  it("relTime minutes/hours/days + negative clamp", () => {
    expect(relTime(200, 50)).toBe("2m ago");
    expect(relTime(3700, 100)).toBe("1h ago");
    expect(relTime(200000, 1000)).toBe("2d ago");
    expect(relTime(5, 10)).toBe("0s ago");
  });
  it("formatters guard non-finite/null", () => {
    expect(fmtCd(Infinity)).toBe("—");
    expect(fmtIoU(NaN)).toBe("—");
    expect(fmtMm(undefined)).toBe("—");
  });
});
