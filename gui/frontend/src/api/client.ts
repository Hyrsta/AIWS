import type { Health, Catalog, JobSummary, Metrics, CleanupMetadata, ReconstructInput, JobInputs } from "./types";

// Same-origin in prod; vite dev proxy forwards these paths to VITE_BACKEND_URL.
const BASE = import.meta.env.VITE_API_BASE ?? "";

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return (await r.json()) as T;
}
async function jpost<T>(path: string, body?: FormData): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { method: "POST", body });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  health: () => jget<Health>("/health"),
  catalog: () => jget<Catalog>("/catalog"),
  listJobs: () => jget<JobSummary[]>("/jobs"),
  getJob: (id: string) => jget<JobSummary>(`/jobs/${id}`),
  getLogs: (id: string, tail = 400) =>
    jget<{ job_id: string; log: string }>(`/jobs/${id}/logs?tail_lines=${tail}`),
  getMetrics: (id: string) => jget<Metrics>(`/jobs/${id}/metrics`),
  // metadata files referenced by result_paths
  getJsonFile: <T>(id: string, remotePath: string) =>
    jget<T>(`/jobs/${id}/file?path=${encodeURIComponent(remotePath)}`),
  getCleanupMeta: (id: string, remotePath: string) => api.getJsonFile<CleanupMetadata>(id, remotePath),
  // raw STL bytes for the 3D viewer
  async getMeshBytes(id: string, remotePath: string): Promise<ArrayBuffer> {
    const r = await fetch(`${BASE}/jobs/${id}/file?path=${encodeURIComponent(remotePath)}`);
    if (!r.ok) throw new Error(`mesh ${remotePath} → ${r.status}`);
    return r.arrayBuffer();
  },
  fileUrl: (id: string, remotePath: string) =>
    `${BASE}/jobs/${id}/file?path=${encodeURIComponent(remotePath)}`,
  createSimpleReconstruct(input: ReconstructInput): Promise<JobSummary> {
    const fd = new FormData();
    if (input.input_mode === "mesh") {
      fd.append("input_mode", "mesh");
      fd.append("mesh", input.mesh);
    } else if (input.input_mode === "image") {
      fd.append("input_mode", "image");
      fd.append("image", input.image);
      if (input.detect_prompt) fd.append("detect_prompt", input.detect_prompt);
    } else {
      fd.append("input_mode", "image_mask");
      fd.append("image", input.image);
      fd.append("mask", input.mask);
    }
    fd.append("cadrille_checkpoint_preset", input.cadrille_checkpoint_preset);
    fd.append("cadrille_mode", input.cadrille_mode);
    if (input.workpiece_class) fd.append("workpiece_class", input.workpiece_class);
    if (input.model_code) fd.append("model_code", input.model_code);
    if (input.gpu_index != null) fd.append("gpu_index", String(input.gpu_index));
    return jpost<JobSummary>("/jobs/simple-reconstruct", fd);
  },
  terminate: (id: string) => jpost<{ ok: boolean }>(`/jobs/${id}/terminate`),
  getJobInputs: (id: string) => jget<JobInputs>(`/jobs/${id}/inputs`),
  async deleteJob(id: string): Promise<{ ok: boolean }> {
    const r = await fetch(`${BASE}/jobs/${id}`, { method: "DELETE" });
    if (!r.ok) throw new Error(`DELETE /jobs/${id} → ${r.status}`);
    return (await r.json()) as { ok: boolean };
  },
};
