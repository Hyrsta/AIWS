/* ============================================================
   LiveView — pixel-accurate port of AIWS Design Reference aiws/live.jsx.
   Uses real data layer: useJobPolling, visibleStages/stageToStep, fmtElapsed.
   ============================================================ */
import { useEffect, useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { useJobPolling } from "@/hooks/useJobPolling";
import { visibleStages, stageToStep, failedStep } from "@/lib/stages";
import { fmtElapsed } from "@/lib/format";
import { api } from "@/api/client";
import { Icon } from "@/components/Icon";
import { InputImagesPanel } from "@/components/InputImagesPanel";
import { LogConsole } from "@/components/LogConsole";
import type { JobSummary } from "@/api/types";

/* ============================================================
   STAGE_DEFS — maps visible-stage index → icon + group i18n key.
   Cleanup is a visible post-processing stage; metric alignment is optional.
   ============================================================ */
interface StageDef {
  key: "s1" | "s2" | "s3" | "s4" | "sClean" | "s5";
  icon: "download" | "box" | "layers" | "cube3d" | "sliders" | "ruler";
  group: "stage.s1g" | "stage.s2g" | "stage.s3g" | "stage.s4g" | "stage.sCleang" | "stage.s5g";
  postscaleOnly?: boolean;
}

const STAGE_DEFS: StageDef[] = [
  { key: "s1", icon: "download", group: "stage.s1g" },
  { key: "s2", icon: "box",      group: "stage.s2g" },
  { key: "s3", icon: "layers",   group: "stage.s3g" },
  { key: "s4", icon: "cube3d",   group: "stage.s4g" },
  { key: "sClean", icon: "sliders", group: "stage.sCleang" },
  { key: "s5", icon: "ruler",    group: "stage.s5g", postscaleOnly: true },
];

/* ============================================================
   PipelineStepper
   ============================================================ */
interface PipelineStepperProps {
  job: JobSummary;
  hasPostscale: boolean;
}

function PipelineStepper({ job, hasPostscale }: PipelineStepperProps) {
  const { t } = useTranslation();

  // Failure point + active step. On failure the backend overwrites stage_label
  // with the literal "Failed", so stageToStep() would wrongly fall through to the
  // last (Post-scale) step. Derive the real failing step from stage_timings/stage.
  const isFailed = job.status === "failed" || job.status === "terminated";
  const failedIdx = isFailed
    ? failedStep(job.stage_timings, job.stage, hasPostscale)
    : -1;
  const cur = job.status === "completed"
    ? visibleStages(hasPostscale).length  // all done
    : isFailed
      ? failedIdx
      : stageToStep(job.stage_label, hasPostscale);

  // Per-stage timing from backend
  const nowSec = Date.now() / 1000;
  const timings = job.stage_timings ?? {};

  // Backend stage labels aligned with STAGE_DEFS indices
  const BACKEND_LABELS = [
    "SAM3D: Loading checkpoints",
    "SAM3D: Generating mesh",
    "Cadrille: Preparing input",
    "Cadrille: Generating CAD result",
    "Body cleanup: Removing hallucinated bodies",
    "Post-scaling: Aligning CAD to catalog (mm)",
  ];

  // The backend records each stage's started_at but not ended_at, so a finished
  // stage's duration is derived from the NEXT stage's start. All start times
  // across the full pipeline, ascending.
  const allStarts = Object.values(timings)
    .map((tm) => tm?.started_at)
    .filter((s): s is number => s != null)
    .sort((a, b) => a - b);

  function stageDur(i: number, isLive: boolean): number | null {
    const tm = timings[BACKEND_LABELS[i]];
    if (!tm || tm.started_at == null) return null;
    const start = tm.started_at;
    // Prefer a recorded end; else the next stage's start; else job-end / now.
    let end: number | null = tm.ended_at ?? null;
    if (end == null) {
      const nextStart = allStarts.find((s) => s > start);
      if (nextStart != null) end = nextStart;
      else if (isLive) end = nowSec;
      else end = job.ended_at ?? null;
    }
    if (end == null) return null;
    return Math.max(0, end - start);
  }

  const defs = STAGE_DEFS.filter((d) => !d.postscaleOnly || hasPostscale);

  const cells: React.ReactNode[] = [];
  defs.forEach((d, i) => {
    let cls = "stage";
    let stateLabel = t("stage.waiting");

    if (isFailed && failedIdx === i) {
      cls += " failed";
      stateLabel = job.status === "terminated" ? t("stage.skipped") : t("stage.failed");
    } else if (i < cur) {
      cls += " done";
      stateLabel = t("stage.done");
    } else if (i === cur && job.status === "running") {
      cls += " live";
      stateLabel = t("stage.running");
    } else if (isFailed && failedIdx >= 0 && i < failedIdx) {
      cls += " done";
      stateLabel = t("stage.done");
    }

    const isLive = i === cur && job.status === "running";
    const dur = stageDur(i, isLive);
    const durStr = dur != null ? dur.toFixed(1) + "s" : null;

    cells.push(
      <div className={cls} key={d.key}>
        <div className="stage-num">
          <span>{"0" + (i + 1)}</span>
          <span className="mono">{t(d.group)}</span>
        </div>
        <div className="stage-ico">
          <Icon n={d.icon} size={16} />
        </div>
        <div className="stage-name">{t("stage." + d.key)}</div>
        <div className="stage-foot">
          {isLive ? (
            <div className="scan-bar">
              <span className="fill" />
            </div>
          ) : null}
          <div className="stage-time">
            <span>{stateLabel}</span>
            {durStr ? <span className="stage-dur">{durStr}</span> : null}
          </div>
        </div>
      </div>,
    );

    if (i < defs.length - 1) {
      cells.push(
        <div
          className={"stage-arrow" + (i < cur ? " lit" : "")}
          key={"a" + i}
        >
          <Icon n="arrowR" size={16} />
        </div>,
      );
    }
  });

  const colTemplate = defs.map(() => "1fr auto").join(" ").replace(/ auto$/, "");

  return (
    <div className="pipe">
      <div className="pipe-track" style={{ gridTemplateColumns: colTemplate }}>
        {cells}
      </div>
    </div>
  );
}

/* ============================================================
   Elapsed timer hook
   ============================================================ */
function useElapsed(job: JobSummary | undefined): string {
  const [elapsed, setElapsed] = useState("0.0s");

  useEffect(() => {
    if (!job) return;
    const isTerminal =
      job.status === "completed" ||
      job.status === "failed" ||
      job.status === "terminated";

    const startSec = job.started_at ?? job.created_at;

    function tick() {
      const nowSec = Date.now() / 1000;
      setElapsed(fmtElapsed(Math.max(0, nowSec - startSec)));
    }

    if (isTerminal) {
      // Show final elapsed without a running interval
      const endSec = job.ended_at ?? (Date.now() / 1000);
      setElapsed(fmtElapsed(Math.max(0, endSec - startSec)));
      return;
    }

    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [job, job?.status]);  // re-run when status becomes terminal

  return elapsed;
}

/* ============================================================
   LiveView (exported)
   ============================================================ */
export interface LiveViewProps {
  jobId: string;
  onCompleted: () => void;
}

export function LiveView({ jobId, onCompleted }: LiveViewProps) {
  const { t } = useTranslation();
  const { job, logText } = useJobPolling(jobId, onCompleted);
  const elapsed = useElapsed(job);
  const [cancelling, setCancelling] = useState(false);

  const onCancel = useCallback(() => {
    setCancelling(true);
    api.terminate(jobId).catch(() => {
      // Even on error, keep disabled — user clicked once
    });
  }, [jobId]);

  // Split log text into lines for rendering
  const lines = logText ? logText.split("\n").filter((l) => l.length > 0) : [];

  // Loading guard — first fetch not yet returned
  if (!job) {
    return <div className="canvas" />;
  }

  const hasPostscale = !!(job.request && job.request.postscale_enabled);
  // Cancel only makes sense while the job is still in flight. A failed /
  // Cancel only makes sense while the job is still in flight. A terminal job is
  // already stopped — no button, and no redundant status pill either (the
  // full-width banner below already conveys the failed / terminated state).
  const inFlight = job.status === "running" || job.status === "queued";

  return (
    <div className="canvas wide">
      <div className="run-head reveal">
        <span className="elapsed">
          <Icon
            n="activity"
            size={16}
            style={{ verticalAlign: "-3px", marginRight: 7, color: "var(--sig)" }}
          />
          {t("pipe.elapsed") + " "}
          <b>{elapsed}</b>
        </span>
        {inFlight ? (
          <button
            className="btn btn-danger"
            onClick={onCancel}
            disabled={cancelling}
          >
            <Icon n="stop" size={15} />
            {cancelling ? t("act.cancelling") : t("act.cancel")}
          </button>
        ) : null}
      </div>

      <div className="panel panel-pad reveal-2" style={{ marginBottom: 18 }}>
        <PipelineStepper job={job} hasPostscale={hasPostscale} />
      </div>

      {(job.status === "failed" || job.status === "terminated") && job.error ? (
        <div
          className={"alert " + (job.status === "failed" ? "bad" : "warn")}
          style={{ marginBottom: 18 }}
        >
          <Icon n={job.status === "failed" ? "alert" : "ban"} size={16} />
          <div style={{ minWidth: 0 }}>
            <b>{job.status === "failed" ? t("st.failed") : t("st.stopped")}</b>
            <div
              style={{
                fontFamily: "var(--mono)",
                fontSize: 12,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                maxHeight: 168,
                overflow: "auto",
                marginTop: 4,
              }}
            >
              {job.error}
            </div>
          </div>
        </div>
      ) : null}

      <LogConsole lines={lines} live={job.status === "running" || job.status === "queued"} />

      {/* Uploaded inputs — shown under the log during generation too, not just
          after finishing. Self-fetches the photo+mask. */}
      <div style={{ marginTop: 18 }}>
        <InputImagesPanel jobId={jobId} />
      </div>
    </div>
  );
}
