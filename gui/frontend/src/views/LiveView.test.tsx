import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import i18n from "@/i18n";
import { api } from "@/api/client";
import { LiveView } from "./LiveView";
import type { JobSummary } from "@/api/types";

const fixtures = vi.hoisted(() => {
  const now = 1780416900;
  const job: JobSummary = {
    job_id: "simple_reconstruct-live-cleanup",
    kind: "simple_reconstruct",
    status: "running",
    ssh_host: "local",
    output_root: "/tmp/job",
    created_at: now,
    updated_at: now + 92.9,
    remote_pid: 1254250,
    exit_code: null,
    command: [],
    log_path: "/tmp/job/job.log",
    stage: "sam3d",
    stage_label: "SAM3D: Generating mesh",
    started_at: now,
    ended_at: null,
    stage_timings: {
      "SAM3D: Loading checkpoints": { started_at: now, ended_at: now + 64.2 },
      "SAM3D: Generating mesh": { started_at: now + 64.2, ended_at: null },
    },
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
  return { job };
});

vi.mock("@/api/client", () => ({
  api: {
    getJob: vi.fn().mockResolvedValue(fixtures.job),
    getLogs: vi.fn().mockResolvedValue({ job_id: fixtures.job.job_id, log: "" }),
    terminate: vi.fn().mockResolvedValue({ ok: true }),
  },
}));

function renderLiveView() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <LiveView jobId={fixtures.job.job_id} onCompleted={vi.fn()} />
    </QueryClientProvider>,
  );
}

describe("LiveView pipeline", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("shows Body cleanup between Generate CAD and Metric alignment while reconstructing", async () => {
    renderLiveView();

    expect(await screen.findByText("Body cleanup")).toBeInTheDocument();

    const stageNames = Array.from(document.querySelectorAll(".stage-name")).map((el) => el.textContent);
    expect(stageNames).toEqual([
      "Load checkpoints",
      "Generate mesh",
      "Prepare input",
      "Generate CAD",
      "Body cleanup",
      "Metric alignment",
    ]);
    expect(document.querySelectorAll(".stage")).toHaveLength(6);
    expect(api.getJob).toHaveBeenCalledWith(fixtures.job.job_id);
  });
});
