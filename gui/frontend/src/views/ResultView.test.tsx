import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import i18n from "@/i18n";
import { api } from "@/api/client";
import { ResultView } from "./ResultView";
import type { JobSummary } from "@/api/types";

const fixtures = vi.hoisted(() => {
  const job = {
    job_id: "simple_reconstruct-20260603-001345-d13d4a20",
    kind: "simple_reconstruct",
    status: "completed",
    ssh_host: "local",
    output_root: "/tmp/job",
    created_at: 1780416825.712223,
    updated_at: 1780463866.4184322,
    remote_pid: 1254250,
    exit_code: 0,
    command: [],
    log_path: "/tmp/job/job.log",
    stage: "completed",
    stage_label: "Done",
    started_at: 1780416825.7035446,
    ended_at: 1780416982.735195,
    stage_timings: {
      "SAM3D: Loading checkpoints": { started_at: 1780416825.7035446, ended_at: null },
      "SAM3D: Generating mesh": { started_at: 1780416883.0337243, ended_at: null },
      "Cadrille: Preparing input": { started_at: 1780416908.4224722, ended_at: null },
      "Cadrille: Generating CAD result": { started_at: 1780416909.0666857, ended_at: null },
      "Body cleanup: Removing hallucinated bodies": { started_at: 1780416963.821345, ended_at: null },
      "Post-scaling: Aligning CAD to catalog (mm)": { started_at: 1780416966.991576, ended_at: null },
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

  const metrics = {
    job_id: job.job_id,
    sam3d: {
      available: true,
      duration_sec: 25.357,
      model_init_sec: 39.552,
      peak_memory_reserved_mb: 22536,
      peak_memory_allocated_mb: 18055.26,
      cuda_visible_devices: "2",
    },
    cadrille: {
      available: true,
      duration_sec: 14.298,
      peak_memory_reserved_mb: 4802,
      peak_memory_allocated_mb: 4702.75,
      device_name: "NVIDIA RTX A6000",
      device_total_memory_mb: 48539.44,
    },
    postscale: {
      available: true,
      workpiece_class: "cover_plate",
      model_code: "G140",
      rewrite_mode: "axiswise",
      canonical_extents: [76.5, 200, 67.5],
      catalog_target_mm: [150, 404, 150],
      after_scale_mm: [150.0000002, 404.0000002, 150.0000002],
      max_rel_error: 1.333333254175765e-9,
      match_ok: true,
    },
  };

  const pointCloud = {
    n_points: 256,
    points: Array.from({ length: 256 }, (_, i) => [i / 256, (i % 16) / 16, (i % 8) / 8]),
    source_candidate: "GUI__user_upload__input__obj01+2",
  };
  const backfilledPointCloud = {
    ...pointCloud,
    backfilled: true,
    provenance: "regenerated_from_saved_bridge_stl",
  };
  const scaledMeta = {
    rewrite_mode: "axiswise",
    catalog: {
      workpiece_class: "cover_plate",
      model_code: "G140",
      bbox_m: [0.15, 0.404, 0.15],
      bbox_mm: [150, 404, 150],
    },
    canonical_bbox: {
      xlen: 90,
      ylen: 200,
      zlen: 103,
      xmin: -45,
      ymin: -100,
      zmin: -51.5,
      xmax: 45,
      ymax: 100,
      zmax: 51.5,
    },
    after_scale_bbox_mm: { xlen: 150, ylen: 404, zlen: 150 },
    scale: {
      mode: "axiswise",
      matrix_3x3: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    },
  };

  return { job, metrics, pointCloud, backfilledPointCloud, scaledMeta };
});

const job = fixtures.job as unknown as JobSummary;

vi.mock("@/components/PointCloudViewer", () => ({
  PointCloudViewer: ({ points }: { points: [number, number, number][] }) => (
    <div data-testid="point-cloud-viewer">{points.length} points</div>
  ),
}));

vi.mock("@/components/MeshViewer", () => ({
  MeshViewer: ({ label }: { label: string }) => (
    <div data-testid="mesh-viewer">{label}</div>
  ),
}));

vi.mock("@/api/client", () => ({
  api: {
    getJob: vi.fn().mockResolvedValue(fixtures.job),
    getMetrics: vi.fn().mockResolvedValue(fixtures.metrics),
    getJobInputs: vi.fn().mockResolvedValue({
      job_id: fixtures.job.job_id,
      input_image: null,
      input_mask: null,
    }),
    getJsonFile: vi.fn().mockResolvedValue(fixtures.pointCloud),
    getLogs: vi.fn().mockResolvedValue({ job_id: fixtures.job.job_id, log: "" }),
    fileUrl: vi.fn((_id: string, path: string) => `/file?path=${encodeURIComponent(path)}`),
  },
}));

function renderResultView() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <ResultView jobId={job.job_id} onNew={vi.fn()} />
    </QueryClientProvider>,
  );
}

describe("ResultView timeline", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("does not render a topbar New reconstruction action", async () => {
    renderResultView();

    expect(await screen.findByRole("button", { name: "Results" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "New reconstruction" })).not.toBeInTheDocument();
  });

  it("renders wall-clock stage durations that add up to elapsed time", async () => {
    renderResultView();

    await userEvent.click(await screen.findByRole("button", { name: "Reconstruction" }));

    expect(screen.getByText("2m 37s")).toBeInTheDocument();
    expect(screen.getByText("57.3s")).toBeInTheDocument();
    expect(screen.getByText("25.4s")).toBeInTheDocument();
    expect(screen.getByText("0.6s")).toBeInTheDocument();
    expect(screen.getByText("54.8s")).toBeInTheDocument();
    expect(screen.getByText("Body cleanup")).toBeInTheDocument();
    expect(screen.getByText("3.2s")).toBeInTheDocument();
    expect(screen.getByText("15.7s")).toBeInTheDocument();
    expect(screen.getByText("peak allocated")).toBeInTheDocument();
    expect(screen.getByText("peak reserved")).toBeInTheDocument();
    expect(screen.getByText(/17\.63 GB peak alloc/)).toBeInTheDocument();
    expect(screen.getByText(/4\.59 GB peak alloc/)).toBeInTheDocument();
    expect(screen.getByText(/memory bars show peak allocated\/reserved VRAM against 47\.40 GB capacity/)).toBeInTheDocument();
  });

  it("uses one stage-time bar color and one post-processing chip style", async () => {
    const { container } = renderResultView();

    await userEvent.click(await screen.findByRole("button", { name: "Reconstruction" }));

    const rows = Array.from(container.querySelectorAll<HTMLElement>(".ptl-row"));
    const stageTimeBars = rows
      .map((row) => row.querySelector<HTMLElement>(".ptl-metric .fill"))
      .filter((bar): bar is HTMLElement => Boolean(bar));

    expect(stageTimeBars).toHaveLength(6);
    expect(new Set(stageTimeBars.map((bar) => bar.style.background)).size).toBe(1);

    const cleanupRow = rows.find((row) => row.textContent?.includes("Body cleanup"));
    const alignRow = rows.find((row) => row.textContent?.includes("Metric alignment"));
    expect(cleanupRow).toBeTruthy();
    expect(alignRow).toBeTruthy();

    const cleanupChip = cleanupRow?.querySelector<HTMLElement>(".ptl-mod");
    const alignChip = alignRow?.querySelector<HTMLElement>(".ptl-mod");
    expect(cleanupChip).toHaveTextContent("Post-processing");
    expect(alignChip).toHaveTextContent("Post-processing");
    expect(cleanupChip?.getAttribute("style")).toBe(alignChip?.getAttribute("style"));
  });

  it("shows the actual 256 Cadrille-sampled points when the result exposes them", async () => {
    const jobWithPoints = {
      ...fixtures.job,
      result_paths: {
        results_root: "/tmp/job/results",
        cadrille_input_points: "/tmp/job/results/cadrille_input_points.json",
      },
    } as unknown as JobSummary;
    vi.mocked(api.getJob).mockResolvedValueOnce(jobWithPoints);
    vi.mocked(api.getJsonFile).mockResolvedValueOnce(fixtures.pointCloud);

    renderResultView();

    await userEvent.click(await screen.findByRole("button", { name: "Reconstruction" }));

    expect(await screen.findByText("Cadrille input preview")).toBeInTheDocument();
    expect(screen.getByText("Actual 256 Cadrille-sampled points")).toBeInTheDocument();
    expect(await screen.findByTestId("point-cloud-viewer")).toHaveTextContent("256 points");
  });

  it("labels regenerated historical point previews as backfilled", async () => {
    const jobWithPoints = {
      ...fixtures.job,
      result_paths: {
        results_root: "/tmp/job/results",
        cadrille_input_points: "/tmp/job/results/cadrille_input_points.json",
      },
    } as unknown as JobSummary;
    vi.mocked(api.getJob).mockResolvedValueOnce(jobWithPoints);
    vi.mocked(api.getJsonFile).mockResolvedValueOnce(fixtures.backfilledPointCloud);

    renderResultView();

    await userEvent.click(await screen.findByRole("button", { name: "Reconstruction" }));

    expect(await screen.findByText("Cadrille input preview")).toBeInTheDocument();
    expect(screen.getByText("Backfilled 256-point preview from the saved Cadrille bridge mesh")).toBeInTheDocument();
    expect(screen.getByText("Backfilled")).toBeInTheDocument();
  });

  it("shows body-count chips before range chips in generated CAD preview cards", async () => {
    const jobWithPreviewPaths = {
      ...fixtures.job,
      result_paths: {
        sam3d_mesh_stl: "/tmp/job/results/sam3d_mesh.stl",
        selected_mesh: "/tmp/job/results/cadrille_selected_mesh.stl",
        cleaned_mesh_stl: "/tmp/job/results/cadrille_cleaned.stl",
        scaled_mesh_stl: "/tmp/job/results/cadrille_scaled.stl",
        scaled_metadata: "/tmp/job/results/cadrille_scaled_metadata.json",
      },
    } as unknown as JobSummary;
    vi.mocked(api.getJob).mockResolvedValueOnce(jobWithPreviewPaths);
    vi.mocked(api.getJsonFile).mockResolvedValueOnce(fixtures.scaledMeta);

    const { container } = renderResultView();

    await screen.findByText("Mesh previews");
    await screen.findByText("Target 150 × 404 × 150 mm");

    const previewCells = Array.from(container.querySelectorAll<HTMLElement>(".preview-cell"));
    const canonical = previewCells.find((cell) => cell.textContent?.includes("Cadrille canonical"));
    const cleaned = previewCells.find((cell) => cell.textContent?.includes("Body cleanup"));
    const scaled = previewCells.find((cell) => cell.textContent?.includes("Metric alignment"));

    for (const cell of [canonical, cleaned, scaled]) {
      expect(cell).toBeTruthy();
      expect(cell).toHaveTextContent("1 body");
    }
    expect(canonical!.textContent!.indexOf("1 body")).toBeLessThan(canonical!.textContent!.indexOf("Canonical [-100, 100]"));
    expect(cleaned!.textContent!.indexOf("1 body")).toBeLessThan(cleaned!.textContent!.indexOf("Canonical [-100, 100]"));
    expect(scaled!.textContent!.indexOf("1 body")).toBeLessThan(scaled!.textContent!.indexOf("Target 150 × 404 × 150 mm"));
  });
});
