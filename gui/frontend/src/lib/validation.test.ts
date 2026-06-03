import { describe, it, expect } from "vitest";
import { sameDimensions } from "./validation";
describe("sameDimensions", () => {
  it("true when equal", () => expect(sameDimensions({ w: 1024, h: 1024 }, { w: 1024, h: 1024 })).toBe(true));
  it("false when different", () => expect(sameDimensions({ w: 1024, h: 1024 }, { w: 512, h: 1024 })).toBe(false));
});
