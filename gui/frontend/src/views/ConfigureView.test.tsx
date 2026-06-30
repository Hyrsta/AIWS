/* ============================================================
   ConfigureView — unit tests
   ============================================================ */
import { describe, it, expect, vi, beforeAll } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConfigureView } from "./ConfigureView";

// ---- i18n: initialise catalog so useTranslation() returns real strings ----
import "@/i18n";

// ---- Stub the catalog query so it returns data immediately ----
vi.mock("@/api/client", () => ({
  api: {
    catalog: vi.fn().mockResolvedValue({
      source: "test",
      classes: {
        WidgetA: {
          models: ["W-100", "W-200"],
          entries: {
            "W-100": { bbox_m: [0.1, 0.1, 0.05], bbox_mm: [100, 100, 50] },
            "W-200": { bbox_m: [0.2, 0.1, 0.05], bbox_mm: [200, 100, 50] },
          },
        },
      },
    }),
    health: vi.fn().mockResolvedValue({ ok: true, gpus: [] }),
    // segmentSession and segmentRelease are used by SegmentRefineCanvas; provide
    // defaults here and override per-test as needed.
    segmentSession: vi.fn().mockResolvedValue({
      session_id: "test-session",
      mask_png_base64: "AAAA",
      detected: true,
    }),
    segmentRelease: vi.fn().mockResolvedValue(undefined),
  },
}));

// ---- suppress act() warnings from URL.createObjectURL not being in jsdom ----
beforeAll(() => {
  if (!globalThis.URL.createObjectURL) {
    Object.defineProperty(globalThis.URL, "createObjectURL", {
      value: vi.fn(() => "blob:mock"),
      writable: true,
    });
    Object.defineProperty(globalThis.URL, "revokeObjectURL", {
      value: vi.fn(),
      writable: true,
    });
  }

  class MockImage {
    naturalWidth = 1024;
    naturalHeight = 768;
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;

    set src(_value: string) {
      queueMicrotask(() => {
        this.onload?.();
      });
    }
  }

  vi.stubGlobal("Image", MockImage);
});

function renderConfigureView() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const onStart = vi.fn();
  render(
    <QueryClientProvider client={qc}>
      <ConfigureView onStart={onStart} />
    </QueryClientProvider>,
  );
  return { onStart };
}

describe("ConfigureView", () => {
  it("Start button is disabled before any upload", () => {
    renderConfigureView();
    const btn = screen.getByRole("button", { name: /start/i });
    expect(btn).toBeDisabled();
  });

  it("marks the configuration panels with stable layout hooks", () => {
    renderConfigureView();

    expect(screen.getByRole("heading", { name: "Inputs" }).closest(".panel")).toHaveClass(
      "cfg-input-panel",
      "has-sample-action",
    );
    expect(screen.getByRole("heading", { name: "Cadrille settings" }).closest(".panel")).toHaveClass(
      "cfg-settings-panel",
    );
    expect(screen.getByRole("heading", { name: "Metric post-scaling" }).closest(".panel")).toHaveClass(
      "cfg-post-panel",
    );
    expect(document.querySelector(".cfg-settings-hint")).not.toBeNull();
  });

  it("shows only the filename (not parent folders) for uploaded inputs", async () => {
    renderConfigureView();

    const [photoInput, maskInput] = document.querySelectorAll<HTMLInputElement>("input[type=file]");
    const photoFile = new File(["photo"], "input.png", { type: "image/png" });
    const maskFile = new File(["mask"], "mask.png", { type: "image/png" });
    // Even if the browser exposes a directory-relative path, the caption must
    // show the bare filename only (parent folders are intentionally not shown).
    Object.defineProperty(photoFile, "webkitRelativePath", {
      value: "Best Results/RL_IMG/rank_01/Input Data/input.png",
    });

    fireEvent.change(photoInput, { target: { files: [photoFile] } });
    fireEvent.change(maskInput, { target: { files: [maskFile] } });

    expect(await screen.findByText("input.png")).toBeInTheDocument();
    expect(await screen.findByText("mask.png")).toBeInTheDocument();
    expect(
      screen.queryByText("Best Results/RL_IMG/rank_01/Input Data/input.png"),
    ).toBeNull();
  });

  it("opens the single-file picker from the visible image drop area", async () => {
    renderConfigureView();

    const photoFileInput = document.querySelector<HTMLInputElement>('input[data-upload-kind="photo-file"]');
    expect(photoFileInput).not.toBeNull();
    const fileClick = vi.fn();
    photoFileInput!.click = fileClick;

    fireEvent.click(screen.getByText("Drop a photo or click to browse"));

    expect(fileClick).toHaveBeenCalledOnce();
  });

  it("starts reconstruction from an uploaded mesh without requiring photo or mask", async () => {
    const { onStart } = renderConfigureView();

    fireEvent.click(await screen.findByRole("button", { name: /mesh input/i }));
    const meshInput = document.querySelector<HTMLInputElement>('input[type="file"][accept=".stl,.glb,.obj,.ply"]');
    expect(meshInput).not.toBeNull();
    const meshDropText = await screen.findByText("Drop a mesh or click to browse");
    const fillArea = meshDropText.closest(".mesh-fill-area");
    expect(fillArea).not.toBeNull();
    expect(fillArea?.querySelector(".drop")).not.toBeNull();

    const meshFile = new File(["solid"], "favorite.stl", { type: "model/stl" });
    fireEvent.change(meshInput!, { target: { files: [meshFile] } });
    expect(await screen.findByText("favorite.stl")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /start/i }));
    expect(onStart).toHaveBeenCalledWith(expect.objectContaining({ input_mode: "mesh", mesh: meshFile }));
  });

  it("RGB-only: use-mask routes an image_mask reconstruct", async () => {
    const { onStart } = renderConfigureView();

    // Switch to RGB-only (auto-segment) mode.
    fireEvent.click(await screen.findByRole("button", { name: /rgb only/i }));

    // Upload a photo so the SegmentRefineCanvas mounts and calls segmentSession.
    const photoInput = document.querySelector<HTMLInputElement>('input[data-upload-kind="photo-file"]');
    expect(photoInput).not.toBeNull();
    const photoFile = new File(["img"], "part.png", { type: "image/png" });
    fireEvent.change(photoInput!, { target: { files: [photoFile] } });

    // Wait for the mask to be published: the "Use this mask and reconstruct" button
    // becomes enabled once segmentSession resolves and onMaskChange fires.
    const useBtn = await screen.findByRole("button", { name: /use this mask/i });
    await waitFor(() => expect(useBtn).not.toBeDisabled(), { timeout: 3000 });

    fireEvent.click(useBtn);

    expect(onStart).toHaveBeenCalledOnce();
    const arg = onStart.mock.calls[0][0] as Record<string, unknown>;
    expect(arg.input_mode).toBe("image_mask");
    expect(arg.mask).toBeInstanceOf(File);
  });
});
