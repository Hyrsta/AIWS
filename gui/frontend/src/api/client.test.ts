import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { api } from "./client";

// The client uses a relative BASE ("") so requests resolve against jsdom's
// origin (http://localhost:3000). MSW node matches by absolute URL, so handlers
// are registered absolute to that same origin.
const ORIGIN = "http://localhost:3000";

let lastForm: FormData | null = null;

const server = setupServer(
  http.get(`${ORIGIN}/health`, () => HttpResponse.json({ ok: true, workspace_root: "/w" })),
  http.get(`${ORIGIN}/jobs`, () => HttpResponse.json([{ job_id: "j1", status: "completed" }])),
  http.get(`${ORIGIN}/jobs/:id/metrics`, () =>
    HttpResponse.json({ job_id: "j1", sam3d: { available: false }, cadrille: { available: false }, postscale: { available: false } })),
  http.post(`${ORIGIN}/jobs/simple-reconstruct`, async ({ request }) => {
    lastForm = await request.formData();
    return HttpResponse.json({ job_id: "new", status: "running" });
  }),
);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => { server.resetHandlers(); lastForm = null; });
afterAll(() => server.close());

describe("api client", () => {
  it("GET /health", async () => { expect((await api.health()).ok).toBe(true); });
  it("GET /jobs returns a list", async () => { expect((await api.listJobs())[0].job_id).toBe("j1"); });
  it("POST /jobs/simple-reconstruct transmits the multipart fields", async () => {
    const f = new File(["x"], "rgb.png", { type: "image/png" });
    const job = await api.createSimpleReconstruct({ image: f, mask: f, cadrille_checkpoint_preset: "RL", cadrille_mode: "PC" });
    expect(job.job_id).toBe("new");
    // the real client must actually send the form fields, not just hit the URL
    expect(lastForm?.get("cadrille_mode")).toBe("PC");
    expect(lastForm?.get("cadrille_checkpoint_preset")).toBe("RL");
    expect(lastForm?.get("image")).toBeTruthy();
    expect(lastForm?.get("mask")).toBeTruthy();
  });
  it("omits optional workpiece fields when not provided", async () => {
    const f = new File(["x"], "rgb.png", { type: "image/png" });
    await api.createSimpleReconstruct({ image: f, mask: f, cadrille_checkpoint_preset: "SFT", cadrille_mode: "IMG" });
    expect(lastForm?.has("workpiece_class")).toBe(false);
    expect(lastForm?.has("model_code")).toBe(false);
  });
});
