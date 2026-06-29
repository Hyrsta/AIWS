import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import type { JobSummary } from "@/api/types";
import "@/i18n";

const fixtures = vi.hoisted(() => {
  const job: JobSummary = {
    job_id: "simple_reconstruct-20260603-001345-d13d4a20",
    kind: "simple_reconstruct",
    status: "completed",
    ssh_host: "local",
    output_root: "/tmp/job",
    created_at: 1780416825.712223,
    updated_at: 1780416982.735195,
    remote_pid: null,
    exit_code: 0,
    command: [],
    log_path: "/tmp/job/job.log",
    stage: "completed",
    stage_label: "Done",
    started_at: 1780416825.7035446,
    ended_at: 1780416982.735195,
    stage_timings: {},
    result_paths: null,
    request: {
      image_filename: "input.png",
      mask_filename: "mask.png",
      cadrille_checkpoint_preset: "RL",
      cadrille_checkpoint: "ckpt/cadrille_rl",
      cadrille_mode: "pc",
      cadrille_mode_label: "PC",
      workpiece_class: "cover_plate",
      model_code: "G140",
      postscale_enabled: true,
      gpu_index: 2,
    },
    error: null,
  };

  return {
    job,
    health: {
      ok: true,
      workspace_root: "/w",
      jobs_root: "/jobs",
      defaults: {},
      simple_defaults: {},
      runtime: {},
      catalog_path: "/catalog.json",
    },
    metrics: {
      job_id: job.job_id,
      sam3d: { available: false },
      cadrille: { available: false },
      postscale: { available: false },
    },
  };
});

vi.mock("@/api/client", () => ({
  api: {
    health: vi.fn().mockResolvedValue(fixtures.health),
    listJobs: vi.fn().mockResolvedValue([fixtures.job]),
    getJob: vi.fn().mockResolvedValue(fixtures.job),
    getMetrics: vi.fn().mockResolvedValue(fixtures.metrics),
    getJobInputs: vi.fn().mockResolvedValue({
      job_id: fixtures.job.job_id,
      input_image: null,
      input_mask: null,
    }),
    getLogs: vi.fn().mockResolvedValue({ job_id: fixtures.job.job_id, log: "" }),
    fileUrl: (_id: string, path: string) => `/jobs/${_id}/file?path=${encodeURIComponent(path)}`,
  },
}));

function renderApp() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <App />
    </QueryClientProvider>,
  );
}

describe("App route persistence", () => {
  beforeEach(() => {
    localStorage.clear();
    window.history.replaceState(null, "", "/");
  });

  it("opens the active result from the URL hash after a browser reload", async () => {
    window.history.replaceState(
      null,
      "",
      `/#view=result&job=${fixtures.job.job_id}`,
    );

    renderApp();

    expect(await screen.findByRole("button", { name: "Results" })).toBeInTheDocument();
    expect(screen.queryByText("Configure reconstruction")).not.toBeInTheDocument();
    expect(screen.getByText("Output folder")).toBeInTheDocument();
  });
});
