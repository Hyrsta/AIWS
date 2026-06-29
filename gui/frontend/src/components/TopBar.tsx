/* ============================================================
   TopBar — instrument top bar. Ported from AIWS Design Reference aiws/chrome.jsx
   ============================================================ */
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { JobSummary, Health } from "@/api/types";
import { Icon } from "@/components/Icon";

type View = "configure" | "live" | "result";

function cfgChip(k: string, v: string, muted: boolean) {
  return (
    <div className={"cfg-chip" + (muted ? " muted" : "")}>
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  );
}

export function TopBar({
  view,
  job,
  lang,
  setLang,
}: {
  view: View;
  job: JobSummary | null;
  lang: string;
  setLang: (l: string) => void;
}) {
  const { t } = useTranslation();

  // Device name from metrics. Fetched for any active job (incl. failed /
  // terminated) so the device pill still shows when a run errors out — the
  // Cadrille metrics record the GPU even on failure.
  const metricsQ = useQuery({
    queryKey: ["metrics", job?.job_id],
    queryFn: () => api.getMetrics(job!.job_id),
    enabled: !!job?.job_id && view !== "configure",
  });

  // /health carries the GPU inventory; lets us name the device DURING generation
  // (before Cadrille metrics exist) via the job's selected gpu_index.
  const healthQ = useQuery<Health>({ queryKey: ["health"], queryFn: api.health });

  const r = job?.request ?? null;
  const active = job !== null && (view === "live" || view === "result");

  // Device label, only for an active run (never on the Configure page — no run
  // has happened yet). Fallback chain so it appears immediately during
  // generation, before Cadrille metrics exist:
  //   1. metrics device name (once Cadrille has run)
  //   2. the selected GPU's name from /health (by request.gpu_index)
  //   3. the common GPU model if every GPU is the same
  const gpus = healthQ.data?.gpus ?? [];
  const selectedGpu =
    r?.gpu_index != null ? gpus.find((g) => g.index === r.gpu_index) : undefined;
  const commonName =
    gpus.length > 0 && gpus.every((g) => g.name === gpus[0].name) ? gpus[0].name : null;
  const deviceName = active
    ? metricsQ.data?.cadrille?.device_name ?? selectedGpu?.name ?? commonName ?? null
    : null;
  // Pair the GPU model with its cuda index, shown together.
  const cudaTag = r?.gpu_index != null ? `cuda:${r.gpu_index}` : null;

  // Left side: job id OR screen title. No status pill — run state is conveyed
  // by the pipeline stepper / banner in the main content area.
  let left: React.ReactNode;
  if (active && job) {
    left = (
      <div className="bar-id">
        <span className="bar-name mono">{job.job_id}</span>
      </div>
    );
  } else {
    left = (
      <div className="bar-id">
        <span className="bar-name bar-title-h">{t("title.configure")}</span>
      </div>
    );
  }

  const chips = r ? (
    <div className="cfg-chips">
      {cfgChip(t("bar.ckpt"), r.cadrille_checkpoint_preset || "RL", false)}
      {cfgChip(t("bar.input"), r.cadrille_mode_label || (r.cadrille_mode || "PC").toUpperCase(), false)}
      {cfgChip(
        t("bar.scale"),
        r.postscale_enabled
          ? (r.workpiece_class || "") + " / " + (r.model_code || "")
          : t("bar.noScale"),
        !r.postscale_enabled,
      )}
    </div>
  ) : null;

  return (
    <header className="bar">
      {left}
      {chips}
      <div className="bar-spacer" />
      {deviceName || cudaTag ? (
        <div className="health device">
          <Icon n="gauge" size={14} style={{ color: "var(--tx-lo)" }} />
          {deviceName ? <b>{deviceName}</b> : null}
          {cudaTag ? <span className="health-gpu mono">{cudaTag}</span> : null}
        </div>
      ) : null}
      <div className="seg lang-seg">
        <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>
          EN
        </button>
        <button className={lang === "zh" ? "on" : ""} onClick={() => setLang("zh")}>
          中文
        </button>
      </div>
    </header>
  );
}
