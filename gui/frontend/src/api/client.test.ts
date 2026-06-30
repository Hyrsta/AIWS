import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { api } from "./client";

// The client uses a relative BASE ("") so requests resolve against jsdom's
// origin (http://localhost:3000). MSW node matches by absolute URL, so handlers
// are registered absolute to that same origin.
const ORIGIN = "http://localhost:3000";

const server = setupServer(
  http.get(`${ORIGIN}/health`, () => HttpResponse.json({ ok: true, workspace_root: "/w" })),
  http.get(`${ORIGIN}/jobs`, () => HttpResponse.json([{ job_id: "j1", status: "completed" }])),
  http.get(`${ORIGIN}/jobs/:id/metrics`, () =>
    HttpResponse.json({ job_id: "j1", sam3d: { available: false }, cadrille: { available: false }, postscale: { available: false } })),
);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => {
  server.close();
});

function mockPostFetch() {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    json: async () => ({ job_id: "new", status: "running" }),
  } as Response);
}

describe("api client", () => {
  it("GET /health", async () => { expect((await api.health()).ok).toBe(true); });
  it("GET /jobs returns a list", async () => { expect((await api.listJobs())[0].job_id).toBe("j1"); });
  it("POST /jobs/simple-reconstruct transmits the multipart fields", async () => {
    const fetchMock = mockPostFetch();
    const f = new File(["x"], "rgb.png", { type: "image/png" });
    const job = await api.createSimpleReconstruct({ image: f, mask: f, cadrille_checkpoint_preset: "RL", cadrille_mode: "PC" });
    expect(job.job_id).toBe("new");
    // the real client must actually send the form fields, not just hit the URL
    expect(fetchMock).toHaveBeenCalledWith("/jobs/simple-reconstruct", expect.objectContaining({ method: "POST" }));
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("cadrille_mode")).toBe("PC");
    expect(body.get("cadrille_checkpoint_preset")).toBe("RL");
    expect(body.get("image")).toBe(f);
    expect(body.get("mask")).toBe(f);
  });
  it("omits optional workpiece fields when not provided", async () => {
    const fetchMock = mockPostFetch();
    const f = new File(["x"], "rgb.png", { type: "image/png" });
    await api.createSimpleReconstruct({ image: f, mask: f, cadrille_checkpoint_preset: "SFT", cadrille_mode: "IMG" });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.has("workpiece_class")).toBe(false);
    expect(body.has("model_code")).toBe(false);
  });
  it("POST /jobs/simple-reconstruct transmits mesh uploads without image or mask fields", async () => {
    const fetchMock = mockPostFetch();
    const mesh = new File(["solid"], "favorite.stl", { type: "model/stl" });
    await api.createSimpleReconstruct({ input_mode: "mesh", mesh, cadrille_checkpoint_preset: "RL", cadrille_mode: "PC" });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("input_mode")).toBe("mesh");
    expect(body.get("mesh")).toBe(mesh);
    expect(body.has("image")).toBe(false);
    expect(body.has("mask")).toBe(false);
  });
});

function mockFetchOnce(json: unknown, ok = true, status = 200) {
  vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
    ok, status, json: async () => json, text: async () => JSON.stringify(json),
  } as Response);
}

describe("segment refine client", () => {
  afterEach(() => vi.restoreAllMocks());

  it("segmentSession posts the image and returns the session", async () => {
    mockFetchOnce({ session_id: "s1", mask_png_base64: "AUTO", box: null,
      score: 0.5, width: 4, height: 3, detected: true });
    const file = new File([new Uint8Array([1, 2, 3])], "a.png", { type: "image/png" });
    const out = await api.segmentSession(file, "bracket");
    expect(out.session_id).toBe("s1");
    const fetchMock = vi.mocked(globalThis.fetch);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/segment/session");
    expect((init as RequestInit).method).toBe("POST");
  });

  it("segmentRefine posts JSON body with points", async () => {
    mockFetchOnce({ mask_png_base64: "REF", score: 0.6, width: 4, height: 3 });
    const out = await api.segmentRefine("s1", { points: [{ x: 2, y: 3, label: 1 }] });
    expect(out.mask_png_base64).toBe("REF");
    const fetchMock = vi.mocked(globalThis.fetch);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/segment/refine");
    expect(JSON.parse((init as RequestInit).body as string).points[0].x).toBe(2);
  });

  it("segmentRelease issues DELETE", async () => {
    mockFetchOnce({}, true, 204);
    await api.segmentRelease("s1");
    const fetchMock = vi.mocked(globalThis.fetch);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/segment/session/s1");
    expect((init as RequestInit).method).toBe("DELETE");
  });
});
