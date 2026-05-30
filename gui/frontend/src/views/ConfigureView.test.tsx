/* ============================================================
   ConfigureView — unit tests
   ============================================================ */
import { describe, it, expect, vi, beforeAll } from "vitest";
import { render, screen } from "@testing-library/react";
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
});
