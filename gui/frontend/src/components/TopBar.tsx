/* ============================================================
   TopBar — instrument top bar. Ported from AIWS Design Reference aiws/chrome.jsx
   ============================================================ */
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { JobSummary } from "@/api/types";
import { Icon } from "@/components/Icon";
import { Chip } from "@/components/Icon";

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

  // Device name from metrics when a completed job is active
  const metricsQ = useQuery({
    queryKey: ["metrics", job?.job_id],
    queryFn: () => api.getMetrics(job!.job_id),
    enabled: !!job?.job_id && job?.status === "completed",
  });
  const device = metricsQ.data?.cadrille?.device_name ?? null;

  const r = job?.request ?? null;
  const active = job !== null && (view === "live" || view === "result");

  // Left side: status chip + job id OR screen title
  let left: React.ReactNode;
  if (active && job) {
    type StInfo = { tone: "info" | "warn" | "bad"; icon?: import("@/components/Icon").IconName; dot?: boolean; label: string };
    const stMap: Record<string, StInfo> = {
      running: { tone: "info", dot: true, label: t("st.running") },
      terminated: { tone: "warn", icon: "ban", label: t("st.stopped") },
      failed: { tone: "bad", icon: "alert", label: t("st.failed") },
    };
    const s = stMap[job.status];
    left = (
      <div className="bar-id">
        {s ? (
          <Chip tone={s.tone} icon={s.icon} dot={s.dot}>
            {s.label}
          </Chip>
        ) : null}
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
      {device ? (
        <div className="health device">
          <Icon n="gauge" size={14} style={{ color: "var(--tx-lo)" }} />
          <b>{device}</b>
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
