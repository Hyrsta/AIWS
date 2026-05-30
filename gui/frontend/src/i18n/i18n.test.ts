import { describe, it, expect } from "vitest";
import en from "./en.json";
import zh from "./zh.json";
describe("i18n catalogs", () => {
  it("zh has every key en has", () => {
    const ek = Object.keys(en).sort(), zk = Object.keys(zh).sort();
    expect(zk).toEqual(ek);
  });
  it("has core keys", () => { expect(en).toHaveProperty("app.title"); expect(zh).toHaveProperty("app.title"); });
});
