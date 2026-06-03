/* ============================================================
   Rail — left sidebar. Ported from AIWS Design Reference aiws/chrome.jsx
   ============================================================ */
import { useTranslation } from "react-i18next";
import type { JobSummary } from "@/api/types";
import { Icon } from "@/components/Icon";
import { StatusDot } from "@/components/Icon";
import { relTime } from "@/lib/format";

function wpLabel(job: JobSummary): string {
  const r = job.request;
  const parts: string[] = [];
  parts.push(r.cadrille_checkpoint_preset || "RL");
  parts.push(r.cadrille_mode_label || (r.cadrille_mode || "PC").toUpperCase());
  if (r.workpiece_class) parts.push(r.model_code || "");
  return parts.filter(Boolean).join(" · ");
}

export function Rail({
  jobs,
  activeId,
  onNew,
  onSelect,
  onDelete,
}: {
  jobs: JobSummary[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const { t } = useTranslation();
  const nowSec = Date.now() / 1000;

  return (
    <aside className="rail">
      <div className="rail-head">
        <div className="mark">
          <Icon n="cube3d" size={19} />
        </div>
        <div className="wordmark">
          <b>{t("app.name")}</b>
          <span>{t("app.tagline")}</span>
        </div>
      </div>

      <div className="rail-new">
        <button className="btn btn-primary block lg" onClick={onNew}>
          <Icon n="plus" size={18} />
          {t("nav.new")}
        </button>
      </div>

      <div className="rail-scroll">
        <div className="rail-section">
          <span className="eyebrow">{t("nav.history")}</span>
          <span className="mono" style={{ fontSize: 12, color: "var(--tx-dim)" }}>
            {jobs.length}
          </span>
        </div>

        {jobs.length === 0 ? (
          <div className="empty-rail">{t("nav.noHistory")}</div>
        ) : (
          jobs.map((j) => (
            <div
              key={j.job_id}
              className={"history-row" + (j.job_id === activeId ? " active" : "")}
            >
              <button
                className={"history-item" + (j.job_id === activeId ? " active" : "")}
                onClick={() => onSelect(j.job_id)}
              >
                <StatusDot status={j.status as import("@/components/Icon").StatusKind} />
                <div className="hi-main">
                  <div className="hi-when">
                    {relTime(nowSec, j.created_at)}
                  </div>
                  <div className="hi-meta">{wpLabel(j)}</div>
                </div>
                <Icon n="chevR" size={14} className="hi-chev" style={{ color: "var(--tx-dim)" }} />
              </button>
              <button
                className="hi-del"
                title={t("nav.delete")}
                aria-label={t("nav.delete")}
                onClick={(e) => { e.stopPropagation(); onDelete(j.job_id); }}
              >
                <Icon n="trash" size={15} />
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
