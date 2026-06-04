/* ============================================================
   ResultView — pixel-accurate port of AIWS Design Reference
   aiws/result.jsx.  Wired to real backend via @tanstack/react-query.
   ============================================================ */
import { useState, lazy, Suspense } from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { useNow } from "@/hooks/useNow";
import { api } from "@/api/client";
import { fmtIoU } from "@/lib/format";
import { Icon, Chip } from "@/components/Icon";
import { InputImagesPanel } from "@/components/InputImagesPanel";
import { LogConsole } from "@/components/LogConsole";
import { Alert, AlertTitle } from "@/components/ui/alert";
import type { CadrilleInputPoints, CleanupMetadata, ScaledMetadata, JobSummary, Metrics, StageMetrics, Vec3 } from "@/api/types";

/* ---- lazy-load the 3-D viewer to keep it in its own chunk ---- */
const MeshViewer = lazy(() =>
  import("@/components/MeshViewer").then((m) => ({ default: m.MeshViewer }))
);
const PointCloudViewer = lazy(() =>
  import("@/components/PointCloudViewer").then((m) => ({ default: m.PointCloudViewer }))
);

/* ============================================================
   Props
   ============================================================ */
interface ResultViewProps {
  jobId: string;
  onNew: () => void;
}

/* ============================================================
   Helpers
   ============================================================ */
function fmtSec(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s >= 60) {
    const m = Math.floor(s / 60);
    return `${m}m ${(s % 60).toFixed(0)}s`;
  }
  return s.toFixed(1) + "s";
}

function fmtGB(mb: number | null | undefined): string {
  if (mb == null) return "—";
  return (mb / 1024).toFixed(2) + " GB";
}

function fileExt(path: string): string {
  const m = path.match(/\.(\w+)$/);
  return m ? m[1].toLowerCase() : "file";
}

function baseName(path: string): string {
  return path.split("/").pop() ?? path;
}

function extClass(ext: string): string {
  if (ext === "stl") return "stl";
  if (ext === "step" || ext === "stp") return "step";
  if (ext === "glb" || ext === "gltf") return "glb";
  if (ext === "py") return "py";
  return "json";
}

/* Color helpers for StepBadge */
function ppRgba(hex: string, a: number): string {
  const n = hex.replace("#", "");
  const r = parseInt(n.slice(0, 2), 16);
  const g = parseInt(n.slice(2, 4), 16);
  const b = parseInt(n.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}
function ppDark(hex: string, f: number): string {
  const n = hex.replace("#", "");
  const r = Math.round(parseInt(n.slice(0, 2), 16) * f);
  const g = Math.round(parseInt(n.slice(2, 4), 16) * f);
  const b = Math.round(parseInt(n.slice(4, 6), 16) * f);
  return `rgb(${r},${g},${b})`;
}

/* ============================================================
   StepBadge
   ============================================================ */
interface StepBadgeProps {
  n: number;
  color?: string;
  lg?: boolean;
}
function StepBadge({ n, color = "#8aa0b8", lg }: StepBadgeProps) {
  return (
    <span
      className={"step-badge" + (lg ? " lg" : "")}
      style={{
        background: ppRgba(color, 0.14),
        color: ppDark(color, 0.58),
        borderColor: ppRgba(color, 0.42),
      }}
    >
      {n}
    </span>
  );
}

/* ============================================================
   Metric / MetricGroup
   ============================================================ */
interface MetricProps {
  label: string;
  value: string;
  sig?: boolean;
  sm?: boolean;
  hint?: string;
}
function Metric({ label, value, sig, sm, hint }: MetricProps) {
  return (
    <div className="metric">
      <div className="ml">{label}</div>
      <div className={"mv" + (sig ? " sig" : "") + (sm ? " sm" : "")}>{value}</div>
      {hint ? <div className="mh">{hint}</div> : null}
    </div>
  );
}

interface MetricGroupProps {
  title: string;
  icon?: "gauge" | "cube3d" | "crosshair" | "layers" | "box" | "sliders" | "ruler";
  tag?: string;
  children?: React.ReactNode;
}
function MetricGroup({ title, icon = "gauge", tag, children }: MetricGroupProps) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div className="metric-group-label">
        <Icon n={icon} size={14} style={{ color: "var(--sig)" }} />
        {title}
        {tag ? <span className="tag">{tag}</span> : null}
      </div>
      <div className="metric-grid">{children}</div>
    </div>
  );
}

/* ============================================================
   Process timeline
   ============================================================ */
const PTL_TONE = {
  sam:  { bar: "#5f80a8", soft: "rgba(95,128,168,0.22)", cT: "#3f5f88", cBg: "rgba(95,128,168,0.10)", cBd: "rgba(95,128,168,0.36)" },
  cad:  { bar: "#2f5fa8", soft: "rgba(47,95,168,0.20)",  cT: "#28518f", cBg: "rgba(47,95,168,0.10)",  cBd: "rgba(47,95,168,0.34)"  },
};
const STAGE_TIME_BAR = "#5f80a8";

interface PtlRow {
  key: string;
  backendLabel: string;
  icon: "download" | "box" | "layers" | "cube3d" | "sliders" | "ruler";
  tone: keyof typeof PTL_TONE;
  kind: "gpu" | "cpu";
  nameKey: string;
  groupKey: string;
  sec: number | null;
  mem: { r: number | null; a: number | null } | null;
}

interface ProcessTimelineProps {
  job: JobSummary;
  metrics: Metrics;
  postscale: boolean;
}
function ProcessTimeline({ job, metrics, postscale }: ProcessTimelineProps) {
  const { t } = useTranslation();
  const sam = metrics.sam3d?.available ? metrics.sam3d : null;
  const cad = metrics.cadrille?.available ? metrics.cadrille : null;
  const timings = job.stage_timings ?? {};
  // ticks only while running; only read for the live stage's in-progress duration
  const nowSec = useNow(job.status === "running");
  const allStarts = Object.values(timings)
    .map((tm) => tm?.started_at)
    .filter((s): s is number => s != null)
    .sort((a, b) => a - b);

  function stageSec(label: string, fallback: number | null = null): number | null {
    const tm = timings[label];
    if (!tm || tm.started_at == null) return fallback;
    const start = tm.started_at;
    let end = tm.ended_at ?? null;
    if (end == null) {
      const nextStart = allStarts.find((s) => s > start);
      if (nextStart != null) end = nextStart;
      else if (job.status === "running") end = nowSec;
      else end = job.ended_at ?? null;
    }
    if (end == null) return fallback;
    return Math.max(0, end - start);
  }

  const showCleanup = Boolean(
    timings["Body cleanup: Removing hallucinated bodies"]
      || job.result_paths?.cleanup_metadata
      || job.stage === "body_cleanup"
      || job.stage_label === "Body cleanup: Removing hallucinated bodies",
  );

  const rows: PtlRow[] = [
    {
      key: "s1",
      backendLabel: "SAM3D: Loading checkpoints",
      icon: "download",
      tone: "sam",
      kind: "gpu",
      nameKey: "stage.s1",
      groupKey: "stage.s1g",
      sec: stageSec("SAM3D: Loading checkpoints", sam?.model_init_sec ?? null),
      mem: null,
    },
    {
      key: "s2",
      backendLabel: "SAM3D: Generating mesh",
      icon: "box",
      tone: "sam",
      kind: "gpu",
      nameKey: "stage.s2",
      groupKey: "stage.s2g",
      sec: stageSec("SAM3D: Generating mesh", sam?.duration_sec ?? null),
      mem: sam ? { r: sam.peak_memory_reserved_mb ?? null, a: sam.peak_memory_allocated_mb ?? null } : null,
    },
    {
      key: "s3",
      backendLabel: "Cadrille: Preparing input",
      icon: "layers",
      tone: "cad",
      kind: "cpu",
      nameKey: "stage.s3",
      groupKey: "stage.s3g",
      sec: stageSec("Cadrille: Preparing input"),
      mem: null,
    },
    {
      key: "s4",
      backendLabel: "Cadrille: Generating CAD result",
      icon: "cube3d",
      tone: "cad",
      kind: "gpu",
      nameKey: "stage.s4",
      groupKey: "stage.s4g",
      sec: stageSec("Cadrille: Generating CAD result", cad?.duration_sec ?? null),
      mem: cad ? { r: cad.peak_memory_reserved_mb ?? null, a: cad.peak_memory_allocated_mb ?? null } : null,
    },
  ];
  if (showCleanup) {
    rows.push({
      key: "cleanup",
      backendLabel: "Body cleanup: Removing hallucinated bodies",
      icon: "sliders",
      tone: "cad",
      kind: "cpu",
      nameKey: "res.cleanup",
      groupKey: "res.postproc",
      sec: stageSec("Body cleanup: Removing hallucinated bodies"),
      mem: null,
    });
  }
  if (postscale) {
    rows.push({
      key: "s5",
      backendLabel: "Post-scaling: Aligning CAD to catalog (mm)",
      icon: "ruler",
      tone: "cad",
      kind: "cpu",
      nameKey: "stage.s5",
      groupKey: "res.postproc",
      sec: stageSec("Post-scaling: Aligning CAD to catalog (mm)"),
      mem: null,
    });
  }

  const totalSec = Math.max(1, rows.reduce((a, r) => a + (r.sec ?? 0), 0));
  const vramMb = cad?.device_total_memory_mb
    ?? (Math.max(1, ...rows.map((r) => r.mem?.r ?? 0)) * 1.18);
  const peakMem = Math.max(0, ...rows.map((r) => r.mem?.r ?? 0));

  const failed = job.status === "failed" || job.status === "terminated";
  const lastStartedIdx = rows.reduce((last, row, i) => {
    return timings[row.backendLabel]?.started_at != null ? i : last;
  }, -1);
  const failedIdx = failed ? Math.max(0, lastStartedIdx) : -1;
  let cur: number;
  if (job.status === "completed") cur = 99;
  else if (failed) cur = failedIdx;
  else cur = Math.max(0, rows.findIndex((row) => row.backendLabel === job.stage_label));

  return (
    <div className="panel reveal-2">
      <div className="panel-head">
        <div>
          <h3>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <Icon n="activity" size={15} style={{ color: "var(--sig)" }} />
              {t("pipe.title")}
            </span>
          </h3>
          <p>{t("rt.timelineSub")}</p>
        </div>
        <div className="ptl-cap">
          <Icon n="gauge" size={13} style={{ color: "var(--tx-lo)" }} />
          {fmtGB(vramMb)} VRAM
          {peakMem ? (
            <>
              <span className="sep">·</span>
              <span className="pk">{t("rt.peak", { v: fmtGB(peakMem) })}</span>
            </>
          ) : null}
        </div>
      </div>
      <div className="panel-pad">
        <div className="ptl-head">
          <span />
          <span className="r">{t("rt.colStage")}</span>
          <span className="r">{t("rt.colTime")}</span>
          <span className="r">{t("rt.colMem")}</span>
        </div>
        {rows.map((r, i) => {
          const tone = PTL_TONE[r.tone];
          let cls = "ptl-row";
          let st = t("stage.waiting");
          if (failed && i === failedIdx) {
            cls += " failed";
            st = job.status === "terminated" ? t("stage.skipped") : t("stage.failed");
          } else if (i < cur) {
            cls += " done";
            st = t("stage.done");
          } else if (i === cur && job.status === "running") {
            cls += " live";
            st = t("stage.running");
          }
          if (r.kind === "cpu") cls += " cpu";

          const timeCell = r.sec != null ? (
            <div className="ptl-metric">
              <div className="ptl-bar">
                <div className="fill" style={{ width: (r.sec / totalSec * 100) + "%", background: STAGE_TIME_BAR }} />
              </div>
              <div className="ptl-val">
                <b>{fmtSec(r.sec)}</b>{" "}
                <span className="dim">{Math.round(r.sec / totalSec * 100)}%</span>
              </div>
            </div>
          ) : (
            <div className="ptl-metric empty">
              <div className="ptl-bar" />
              <div className="ptl-val" />
            </div>
          );

          const memCell = r.mem ? (() => {
            const rv = r.mem.r ?? 0;
            const av = r.mem.a ?? 0;
            const rPct = Math.min(100, rv / vramMb * 100);
            const aPct = Math.min(100, av / vramMb * 100);
            return (
              <div className="ptl-metric">
                <div className="ptl-bar">
                  <div className="fill soft" style={{ width: rPct + "%", background: tone.soft }} />
                  <div className="fill" style={{ width: aPct + "%", background: tone.bar }} />
                </div>
                <div className="ptl-val">
                  <b>{fmtGB(rv)}</b>{" "}
                  <span className="dim">{Math.round(rPct)}% · {t("rt.allocShort", { v: fmtGB(av) })}</span>
                </div>
              </div>
            );
          })() : (
            <div className="ptl-metric empty">
              <div className="ptl-bar" />
              <div className="ptl-val">{r.kind === "cpu" ? t("rt.cpuStage") : ""}</div>
            </div>
          );

          return (
            <div className={cls} key={r.key}>
              <div className="ptl-node">
                <div className="ptl-dot">
                  <Icon n={r.icon} size={15} />
                </div>
              </div>
              <div className="ptl-name">
                <div className="nm">{t(r.nameKey)}</div>
                <div className="meta">
                  <span className="ptl-mod" style={{ color: tone.cT, background: tone.cBg, borderColor: tone.cBd }}>
                    {t(r.groupKey)}
                  </span>
                  <span className="ptl-st">
                    <span className="d" />
                    {st}
                  </span>
                </div>
              </div>
              {timeCell}
              {memCell}
            </div>
          );
        })}
        <div className="ptl-foot">
          <span className="lg">
            <b className="alloc" />
            {t("rt.allocated")}
          </span>
          <span className="lg">
            <b className="resv" />
            {t("rt.reserved")}
          </span>
          <span className="note">
            {t("rt.scaleNote", { t: fmtSec(totalSec), v: fmtGB(vramMb) })}
          </span>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   QualityMetrics
   ============================================================ */
interface QualityMetricsProps {
  metrics: Metrics;
  stageMetrics?: StageMetrics | null;
}
function QualityMetrics({ metrics, stageMetrics }: QualityMetricsProps) {
  const { t } = useTranslation();
  const cad = metrics.cadrille;
  if (!cad?.available) return null;

  const iouStr = cad.mean_iou != null ? fmtIoU(cad.mean_iou) : "—";
  const cdStr  = cad.median_cd != null ? (Number(cad.median_cd) * 1e3).toFixed(3) : "—";

  // Per-stage IoU/CD re-scored against the SAM3D GT under BOTH conventions:
  // centered (upstream Cadrille) and corner (bottom-aligned). canonical → cleanup.
  const stageLabel: Record<string, string> = {
    canonical: t("v.canonical"), cleaned: t("res.cleanup"), scaled: t("res.align"),
  };
  type StageRow = { key: string; label: string; cenIou: number | null; cenCd: number | null; corIou: number | null; corCd: number | null };
  const stageRows: StageRow[] = [];
  for (const k of ["canonical", "cleaned"]) {
    const s = stageMetrics?.stages?.[k];
    if (s) stageRows.push({
      key: k, label: stageLabel[k] ?? k,
      cenIou: s.iou ?? null, cenCd: s.cd ?? null,
      corIou: s.iou_corner ?? null, corCd: s.cd_corner ?? null,
    });
  }
  const hasCorner = stageRows.some((r) => r.corIou != null || r.corCd != null);
  const best = (vals: (number | null)[], kind: "max" | "min") => {
    const v = vals.filter((x): x is number => x != null);
    return v.length ? (kind === "max" ? Math.max(...v) : Math.min(...v)) : null;
  };
  const bCenIou = best(stageRows.map((r) => r.cenIou), "max");
  const bCenCd = best(stageRows.map((r) => r.cenCd), "min");
  const bCorIou = best(stageRows.map((r) => r.corIou), "max");
  const bCorCd = best(stageRows.map((r) => r.corCd), "min");
  const pctOrDash = (v: number | null) => (v != null ? (v * 100).toFixed(2) + "%" : "—");
  const cdOrDash = (v: number | null) => (v != null ? (Number(v) * 1e3).toFixed(3) : "—");

  return (
    <div className="panel reveal-2">
      <div className="panel-head">
        <div>
          <h3>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <Icon n="target" size={15} style={{ color: "var(--sig)" }} />
              {t("res.quality")}
            </span>
          </h3>
          <p>{t("res.qualitySub")}</p>
        </div>
      </div>
      <div className="panel-pad">

      {stageRows.length ? (
        <table className="tbl tbl-quality">
          <thead>
            {hasCorner ? (
              <tr>
                <th rowSpan={2}>{t("m.qMesh")}</th>
                <th className="r" colSpan={2} style={{ textAlign: "center" }}>{t("m.qCentered")}</th>
                <th className="r" colSpan={2} style={{ textAlign: "center", borderLeft: "1px solid var(--line)" }}>{t("m.qCorner")}</th>
              </tr>
            ) : null}
            <tr>
              {!hasCorner ? <th>{t("m.qMesh")}</th> : null}
              <th className="r">{t("m.qIou")}</th>
              <th className="r">{t("m.qCd")}</th>
              {hasCorner ? <th className="r" style={{ borderLeft: "1px solid var(--line)" }}>{t("m.qIou")}</th> : null}
              {hasCorner ? <th className="r">{t("m.qCd")}</th> : null}
            </tr>
          </thead>
          <tbody>
            {stageRows.map((r) => (
              <tr key={r.key}>
                <td className="label">{r.label}</td>
                <td className={"r" + (r.cenIou != null && r.cenIou === bCenIou ? " q-best" : "")}>{pctOrDash(r.cenIou)}</td>
                <td className={"r" + (r.cenCd != null && r.cenCd === bCenCd ? " q-best" : "")}>{cdOrDash(r.cenCd)}</td>
                {hasCorner ? (
                  <td className={"r" + (r.corIou != null && r.corIou === bCorIou ? " q-best" : "")} style={{ borderLeft: "1px solid var(--line)" }}>{pctOrDash(r.corIou)}</td>
                ) : null}
                {hasCorner ? (
                  <td className={"r" + (r.corCd != null && r.corCd === bCorCd ? " q-best" : "")}>{cdOrDash(r.corCd)}</td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <MetricGroup title={t("m.quality")} icon="cube3d">
          <Metric label={t("m.meanIou")} value={iouStr} sig hint={t("m.iouHint")} />
          <Metric label={t("m.medianCd")} value={cdStr} hint={t("m.cdHint")} />
        </MetricGroup>
      )}

      {/* Scoring space — how GT + reconstruction are normalized into one unit
          cube before IoU/CD. Mirrors the AIWS Design Reference q-method block. */}
      <div className="q-method">
        <div className="q-norm-viz">
          <svg viewBox="0 0 72 72" aria-hidden="true">
            <line className="dim" x1={15} y1={13} x2={57} y2={13} />
            <line className="dim" x1={15} y1={10} x2={15} y2={16} />
            <line className="dim" x1={57} y1={10} x2={57} y2={16} />
            <text className="lab" x={36} y={9} textAnchor="middle">1</text>
            <rect className="cube" x={15} y={22} width={42} height={42} rx={2} />
            <rect className="mesh" x={16.5} y={32} width={39} height={22} rx={2} />
            <circle className="dot" cx={36} cy={43} r={2} />
          </svg>
        </div>
        <div className="q-norm-body">
          <div className="q-norm-title">{t("m.qMethod")}</div>
          <div className="q-norm-chips">
            <span className="q-chip">{t("m.qChipScale")}</span>
            <span className="q-chip">{t("m.qChipCenter")}</span>
          </div>
          <p className="q-norm-cap">{t("m.qNormNote")}</p>
        </div>
      </div>
      </div>
    </div>
  );
}

/* ============================================================
   CleanupBody
   ============================================================ */
interface CleanupBodyProps {
  cleanup: CleanupMetadata;
}
function CleanupBody({ cleanup }: CleanupBodyProps) {
  const { t } = useTranslation();
  const reasons = cleanup.confidence_reasons ?? [];

  // Format a volume fraction without flattening tiny-but-nonzero values to
  // "0.0%". A body can be removed while contributing a minuscule volume.
  const frac = cleanup.removed_volume_fraction;
  let removedPct: string;
  if (frac == null) {
    removedPct = "—";
  } else {
    const pct = frac * 100;
    if (pct === 0) removedPct = "0%";
    else if (pct < 0.01) removedPct = "<0.01%";
    else if (pct < 1) removedPct = pct.toFixed(2) + "%";
    else removedPct = pct.toFixed(1) + "%";
  }

  const removed = cleanup.n_bodies_removed ?? 0;

  return (
    <>
      <div>
        <div className="metric-group-label">
          <Icon n="box" size={14} style={{ color: "var(--sig)" }} />
          {t("cl.removedVol")}
        </div>
        {/* Removed volume is the hero number (matches the reference). */}
        <div className="pp-bignum">{removedPct}</div>
        <div className="dim" style={{ fontSize: 13, marginTop: 2 }}>
          {t("cl.fromTo", { a: cleanup.n_bodies_before, b: cleanup.n_bodies_after })}
          {"  ·  "}
          {removed} {removed === 1 ? t("v.body") : t("v.bodies")} {t("cl.bodiesRemoved").toLowerCase()}
        </div>
      </div>
      {reasons.length > 0 ? (
        <div>
          <div className="metric-group-label">
            <Icon n="sliders" size={14} style={{ color: "var(--sig)" }} />
            {t("cl.reason")}
          </div>
          <ul className="reason-list">
            {reasons.map((r, i) => (
              <li key={i}>
                <span className="dot" />
                {r}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </>
  );
}

/* ============================================================
   AlignmentBody  (post-scale details inside PPSub)
   ============================================================ */
interface AlignmentBodyProps {
  meta: ScaledMetadata;
}
function AlignmentBody({ meta }: AlignmentBodyProps) {
  const { t } = useTranslation();
  const target = meta.catalog.bbox_mm as [number, number, number];
  const after = [
    meta.after_scale_bbox_mm.xlen,
    meta.after_scale_bbox_mm.ylen,
    meta.after_scale_bbox_mm.zlen,
  ];
  const axes = ["X", "Y", "Z"];
  const rels = after.map((a, i) =>
    target[i] !== 0 ? Math.abs(a - target[i]) / target[i] : 0
  );

  // axis_map may not be present on all metadata versions
  type AxisMapEntry = { rank: number; canonical_axis: string; catalog_axis: string };
  const axisMap: AxisMapEntry[] = (meta.scale as Record<string, unknown>).axis_map as AxisMapEntry[] ?? [];

  // det_note — rendered as mono caption under the 3×3 matrix (result.jsx line 304)
  const detNote = (meta.scale as Record<string, unknown>).det_note;

  return (
    <>
      {/* per-axis table */}
      <div>
        <div className="metric-group-label">
          <Icon n="crosshair" size={14} style={{ color: "var(--sig)" }} />
          {t("al.perAxis")}
        </div>
        <table className="tbl">
          <thead>
            <tr>
              <th>{t("al.axis")}</th>
              <th className="r">{t("al.target")}</th>
              <th className="r">{t("al.actual")}</th>
              <th className="r">{t("al.rel")}</th>
              <th className="r">{t("al.match")}</th>
            </tr>
          </thead>
          <tbody>
            {axes.map((ax, i) => (
              <tr key={ax}>
                <td className="label">{ax}</td>
                <td className="r">{target[i].toFixed(2)}</td>
                <td className="r">{after[i].toFixed(2)}</td>
                <td className="r">{rels[i].toExponential(1)}</td>
                <td className={"r " + (rels[i] < 1e-3 ? "match-ok" : "match-no")}>
                  {rels[i] < 1e-3 ? "✓" : "✗"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* axis pairing */}
      {axisMap.length > 0 ? (
        <div>
          <div className="metric-group-label">
            <Icon n="layers" size={14} style={{ color: "var(--sig)" }} />
            {t("al.pairing")}
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>{t("al.rank")}</th>
                <th>{t("al.canonAxis")}</th>
                <th>{t("al.catAxis")}</th>
              </tr>
            </thead>
            <tbody>
              {axisMap.map((m) => (
                <tr key={m.rank}>
                  <td className="label">#{m.rank}</td>
                  <td>{m.canonical_axis}</td>
                  <td>{m.catalog_axis}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {/* 3×3 matrix + det_note caption (result.jsx lines 299–304) */}
      <div>
        <div className="metric-group-label">
          <Icon n="box" size={14} style={{ color: "var(--sig)" }} />
          {t("al.matrix")}
        </div>
        <div className="matrix">
          {meta.scale.matrix_3x3.flatMap((row, ri) =>
            row.map((v, ci) => (
              <span key={`${ri}-${ci}`} className={ri === ci ? "diag" : ""}>
                {v.toFixed(4)}
              </span>
            ))
          )}
        </div>
        {detNote ? (
          <div
            className="mh"
            style={{ marginTop: 8, fontFamily: "var(--mono)", color: "var(--tx-dim)" }}
          >
            {String(detNote)}
          </div>
        ) : null}
      </div>
    </>
  );
}

/* ============================================================
   PPSub (collapsible post-process step)
   ============================================================ */
interface PPSubProps {
  step: number;
  color: string;
  title: string;
  cue: string;
  children?: React.ReactNode;
}
function PPSub({ step, color, title, cue, children }: PPSubProps) {
  const { t } = useTranslation();
  return (
    <details className="pp-sub disclosure">
      <summary className="pp-summary">
        <StepBadge n={step} color={color} lg />
        <div className="pp-subhead-main">
          <h4>{title}</h4>
          <span className="pp-cue">
            <span className="pp-cue-show">{t("pp.show")} {cue}</span>
            <span className="pp-cue-hide">{t("pp.hide")} {cue}</span>
          </span>
        </div>
        <Icon n="chevR" size={18} className="chev" />
      </summary>
      <div className="body">{children}</div>
    </details>
  );
}

/* ============================================================
   PostProcessSection
   ============================================================ */
interface PostProcessSectionProps {
  cleanup: CleanupMetadata | null;
  meta: ScaledMetadata | null;
}
function PostProcessSection({ cleanup, meta }: PostProcessSectionProps) {
  const { t } = useTranslation();
  const [allOpen, setAllOpen] = useState(false);

  if (!cleanup && !meta) return null;

  const cleanupStep = 3;
  const alignStep = cleanup ? 4 : 3;

  const toggleAll = (e: React.MouseEvent<HTMLButtonElement>) => {
    const panel = (e.currentTarget as HTMLElement).closest(".panel");
    const next = !allOpen;
    setAllOpen(next);
    if (panel) {
      panel.querySelectorAll<HTMLDetailsElement>("details.pp-sub").forEach((d) => {
        d.open = next;
      });
    }
  };

  return (
    <div className="panel reveal-3 pp-panel">
      <div className="panel-head">
        <div>
          <h3>
            <span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
              <Icon n="sliders" size={15} style={{ color: "var(--sig)" }} />
              {t("res.postproc")}
            </span>
          </h3>
          <p>{t("res.postprocSub")}</p>
        </div>
        <button className="pp-expand-all" onClick={toggleAll}>
          {t(allOpen ? "pp.collapseAll" : "pp.expandAll")}
          <span className="pp-ea-chev" style={{ transform: allOpen ? "rotate(90deg)" : "none" }}>
            <Icon n="chevR" size={15} />
          </span>
        </button>
      </div>
      <div className="panel-pad pp-pad">
        {cleanup?.confidence_flag ? (
          <Alert variant="destructive" className="mb-3">
            <AlertTitle>{t("cl.warn")}</AlertTitle>
          </Alert>
        ) : null}
        {cleanup ? (
          <PPSub
            step={cleanupStep}
            color="#c2814e"
            title={t("res.cleanup")}
            cue={t("pp.cleanupDetail")}
          >
            <CleanupBody cleanup={cleanup} />
          </PPSub>
        ) : null}
        {meta ? (
          <PPSub
            step={alignStep}
            color="#3f9f63"
            title={t("res.align")}
            cue={t("pp.alignDetail")}
          >
            <AlignmentBody meta={meta} />
          </PPSub>
        ) : null}
      </div>
    </div>
  );
}

/* ============================================================
   DownloadsSection
   ============================================================ */
interface DownloadsSectionProps {
  jobId: string;
  job: JobSummary;
}
function DownloadsSection({ jobId, job }: DownloadsSectionProps) {
  const { t } = useTranslation();
  const rp = job.result_paths ?? {};
  const ps = !!(job.request?.postscale_enabled);

  const groups: { title: string; items: [string, string | undefined][] }[] = [
    {
      title: t("dl.sam3d"),
      items: [
        [t("dl.sam3dGlb"), rp.sam3d_mesh_glb],
        [t("dl.sam3dStl"), rp.sam3d_mesh_stl],
      ],
    },
    {
      title: t("dl.cadrille"),
      items: [
        [t("dl.selBrep"), rp.selected_brep],
        [t("dl.selStl"),  rp.selected_mesh],
        [t("dl.selPy"),   rp.selected_py],
      ],
    },
    {
      title: t("dl.cleaned"),
      items: [
        [t("dl.cleanStep"), rp.cleaned_brep_step],
        [t("dl.cleanStl"),  rp.cleaned_mesh_stl],
        [t("dl.cleanMeta"), rp.cleanup_metadata],
      ],
    },
  ];

  if (ps) {
    groups.push({
      title: t("dl.scaled"),
      items: [
        [t("dl.scaledStep"), rp.scaled_brep_step],
        [t("dl.scaledStl"),  rp.scaled_mesh_stl],
        [t("dl.scaledPy"),   rp.scaled_py],
        [t("dl.scaledMeta"), rp.scaled_metadata],
      ],
    });
  }

  return (
    <div className="panel reveal-3">
      <div className="panel-head">
        <div>
          <h3>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <Icon n="download" size={15} style={{ color: "var(--sig)" }} />
              {t("res.downloads")}
            </span>
          </h3>
          <p>{t("dl.note")}</p>
        </div>
      </div>
      <div className="panel-pad">
      <div className="dl-groups">
        {groups.map((g, gi) => (
          <div className="dl-group" key={gi}>
            <h4>{g.title}</h4>
            <div className="dl-list">
              {g.items.map(([label, path], i) =>
                path ? (
                  <a
                    key={i}
                    className="dl-item"
                    href={api.fileUrl(jobId, path)}
                    download={baseName(path)}
                  >
                    <span className={"ft " + extClass(fileExt(path))}>
                      {fileExt(path).toUpperCase().slice(0, 4)}
                    </span>
                    <span className="dl-name">
                      <b>{label}</b>
                      <small>{baseName(path)}</small>
                    </span>
                    <Icon n="download" size={16} className="dl-ico" />
                  </a>
                ) : null
              )}
            </div>
          </div>
        ))}
      </div>
      </div>
    </div>
  );
}

/* ============================================================
   OutputFolder
   ============================================================ */
interface OutputFolderProps {
  job: JobSummary;
}
function OutputFolder({ job }: OutputFolderProps) {
  const { t } = useTranslation();
  const rp = job.result_paths ?? {};
  const root = (rp.results_root as string | undefined) ?? job.output_root;

  const copy = () => {
    try {
      void navigator.clipboard?.writeText(root);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="panel reveal-3">
      <div className="panel-head" style={{ alignItems: "center" }}>
        <h3>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <Icon n="folder" size={15} style={{ color: "var(--sig)" }} />
            {t("res.outFolder")}
          </span>
        </h3>
        <button className="btn btn-ghost btn-sm" onClick={copy}>
          <Icon n="copy" size={14} />
          {t("x.copy")}
        </button>
      </div>
      <div className="panel-pad">
        <div className="path-box">
          <Icon n="terminal" size={14} />
          {root}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   MeshPreviews — 2×2 pipeline-stage preview grid
   Matches result.jsx lines 55–102 exactly:
     cell 1 = SAM3D mesh           (sam3d_mesh_stl,    #8aa0b8)
     cell 2 = Cadrille canonical   (selected_mesh,     #9aa0a6)
     cell 3 = Cleaned              (cleaned_mesh_stl,  #43a047)  — when present
     cell 4 = Scaled (mm)          (scaled_mesh_stl,   #3f9f63)  — when present
   ============================================================ */
interface PreviewCell {
  id: string;
  path: string;
  label: string;
  units: string;
  sw: string;
  tone: "info" | "sig" | "ok" | "copper" | "warn";
  pill?: string;
  pillIcon?: "cube3d" | "crosshair";
  bodies?: string;
  decim?: { raw: number; kept: number; vRaw: number; vKept: number; pct: number } | null;
  badgeN: number;
}

interface MeshPreviewsProps {
  jobId: string;
  rp: NonNullable<import("@/api/types").ResultPaths>;
  cleanup: CleanupMetadata | null;
  scaledMeta: ScaledMetadata | null;
  ps: boolean;
}

function MeshPreviews({ jobId, rp, cleanup, scaledMeta, ps }: MeshPreviewsProps) {
  const { t } = useTranslation();

  const bodyStr = (n: number | string) =>
    `${n} ${n === 1 ? t("v.body") : t("v.bodies")}`;

  /* catalog target pill string for the scaled cell (prototype line 73) */
  const targetStr = scaledMeta
    ? `${t("v.target")} ${scaledMeta.catalog.bbox_mm.join(" × ")} mm`
    : undefined;

  /* assemble cells — only include those with a path */
  const cells: PreviewCell[] = [];

  if (rp.sam3d_mesh_stl) {
    // Prefer the cached decimated copy (≈60k faces) over the raw SAM3D mesh
    // (~432k faces / 21MB) for a fast, un-aliased preview; fall back to full.
    // When the backend recorded before/after counts, surface a Downsampled pill.
    const fr = rp.sam3d_faces_raw, fk = rp.sam3d_faces_kept;
    const decim =
      rp.sam3d_mesh_preview_stl && fr != null && fk != null
        ? {
            raw: fr,
            kept: fk,
            vRaw: rp.sam3d_verts_raw ?? 0,
            vKept: rp.sam3d_verts_kept ?? 0,
            pct: rp.sam3d_reduce_pct ?? Math.round((1 - fk / Math.max(1, fr)) * 100),
          }
        : null;
    cells.push({
      id: "sam3d",
      path: (rp.sam3d_mesh_preview_stl || rp.sam3d_mesh_stl) as string,
      label: t("v.sam3d"),
      units: t("v.units.mesh"),
      sw: "#8aa0b8",
      tone: "info",
      decim,
      badgeN: 1,
    });
  }

  if (rp.selected_mesh) {
    cells.push({
      id: "canonical",
      path: rp.selected_mesh as string,
      label: t("v.canonical"),
      units: t("v.units.canon"),
      sw: "#9aa0a6",
      tone: "sig",
      bodies: cleanup ? bodyStr(cleanup.n_bodies_before) : undefined,
      badgeN: cells.length + 1,
    });
  }

  if (rp.cleaned_mesh_stl) {
    const nAfter = cleanup?.n_bodies_after;
    cells.push({
      id: "cleaned",
      path: rp.cleaned_mesh_stl as string,
      label: t("res.cleanup"),
      units: t("v.units.canon"),
      sw: "#43a047",
      tone: "ok",
      bodies: nAfter != null ? bodyStr(nAfter) : undefined,
      badgeN: cells.length + 1,
    });
  }

  if (ps && rp.scaled_mesh_stl) {
    cells.push({
      id: "scaled",
      path: rp.scaled_mesh_stl as string,
      label: t("res.align"),
      units: t("v.units.mm"),
      sw: "#3f9f63",
      tone: "ok",
      pill: targetStr,
      pillIcon: "crosshair",
      badgeN: cells.length + 1,
    });
  }

  if (cells.length === 0) return null;

  return (
    <div className="panel reveal-2">
      <div className="panel-head">
        <div>
          <h3>{t("res.previews")}</h3>
          <p>{t("res.previewsSub")}</p>
        </div>
      </div>
      <div className="panel-pad">
        <div className="preview-grid">
          {cells.map((x) => (
            <div className="preview-cell" key={x.id}>
              {/* compare-cap: badge + swatch + label + chips (result.jsx lines 90–97) */}
              <div className="compare-cap">
                <b style={{ display: "inline-flex", alignItems: "center", gap: 9 }}>
                  <StepBadge n={x.badgeN} color={x.sw} />
                  {x.label}
                </b>
                <div className="cap-chips">
                  {x.decim ? (
                    <span className="ds-tip-wrap" tabIndex={0}>
                      <Chip tone="info" icon="scan">{t("v.downsampled")}</Chip>
                      <span className="ds-tip" role="tooltip">
                        <span className="ds-tip-row">
                          <span className="k">{t("v.faces")}</span>
                          <span className="v">{x.decim.raw.toLocaleString()} → {x.decim.kept.toLocaleString()}</span>
                        </span>
                        <span className="ds-tip-row">
                          <span className="k">{t("v.verts")}</span>
                          <span className="v">{x.decim.vRaw.toLocaleString()} → {x.decim.vKept.toLocaleString()}</span>
                        </span>
                        <span className="ds-tip-row total">
                          <span className="k">{t("v.dsReduced")}</span>
                          <span className="v">−{x.decim.pct}%</span>
                        </span>
                      </span>
                    </span>
                  ) : null}
                  <Chip tone={x.tone} icon={x.pillIcon ?? "cube3d"}>
                    {x.pill ?? x.units}
                  </Chip>
                  {x.bodies ? <Chip icon="layers">{x.bodies}</Chip> : null}
                </div>
              </div>
              <Suspense
                fallback={
                  <div className="viewer-stage" style={{ height: 420 }}>
                    <div className="viewer-loading">
                      <div className="spinner" />
                    </div>
                  </div>
                }
              >
                <MeshViewer
                  jobId={jobId}
                  path={x.path}
                  color={x.sw}
                  height={420}
                  label={x.label}
                  decim={x.decim ? { kept: x.decim.kept, vKept: x.decim.vKept, pct: x.decim.pct } : null}
                />
              </Suspense>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   CadrilleInputPreview — actual modality input fed into Cadrille.
   PC uses the backend-saved 256 sampled points; IMG can show the
   rendered 2×2 input sheet when that artifact is present.
   ============================================================ */
function isVec3(v: unknown): v is Vec3 {
  return (
    Array.isArray(v)
    && v.length === 3
    && v.every((x) => typeof x === "number" && Number.isFinite(x))
  );
}

interface CadrilleInputPreviewProps {
  jobId: string;
  job: JobSummary;
}

function CadrilleInputPreview({ jobId, job }: CadrilleInputPreviewProps) {
  const { t } = useTranslation();
  const rp = job.result_paths ?? {};
  const mode = job.request?.cadrille_mode;
  const pointsPath = mode === "pc" ? rp.cadrille_input_points ?? null : null;
  const renderGridPath = mode === "img" ? rp.cadrille_input_render_grid ?? null : null;

  const pointsQ = useQuery({
    queryKey: ["cadrilleInputPoints", jobId, pointsPath],
    queryFn: () => api.getJsonFile<CadrilleInputPoints>(jobId, pointsPath!),
    enabled: !!pointsPath,
    staleTime: Infinity,
  });

  if (!pointsPath && !renderGridPath) return null;

  const rawPoints = pointsQ.data?.points ?? [];
  const points = rawPoints.filter(isVec3);
  const displayCount = (pointsQ.data?.n_points ?? points.length) || 256;
  const candidate = pointsQ.data?.source_candidate ?? null;
  const backfilled = Boolean(pointsQ.data?.backfilled);

  return (
    <div className="panel reveal-2 cadrille-input-panel">
      <div className="panel-head">
        <div>
          <h3>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <Icon n={mode === "pc" ? "scan" : "image"} size={15} style={{ color: "var(--sig)" }} />
              {t("res.cadrilleInput")}
            </span>
          </h3>
          <p>
            {mode === "pc"
              ? t(backfilled ? "res.cadrilleInputSubPcBackfill" : "res.cadrilleInputSubPc", { n: displayCount })
              : t("res.cadrilleInputSubImg")}
          </p>
        </div>
        <div className="cap-chips">
          <Chip tone="info" icon={mode === "pc" ? "scan" : "image"}>
            {mode === "pc" ? t("res.cadrillePointCount", { n: displayCount }) : job.request.cadrille_mode_label}
          </Chip>
          {backfilled ? <Chip tone="warn" icon="info">{t("res.cadrilleBackfilled")}</Chip> : null}
          {candidate ? <Chip icon="file">{t("res.cadrilleCandidate", { v: candidate })}</Chip> : null}
        </div>
      </div>
      <div className="panel-pad">
        {mode === "pc" ? (
          pointsQ.isError ? (
            <div className="viewer-stage cadrille-input-empty" style={{ height: 320 }}>
              <div className="viewer-loading" style={{ color: "var(--bad)" }}>
                {t("res.cadrillePointsError")}
              </div>
            </div>
          ) : points.length ? (
            <Suspense
              fallback={
                <div className="viewer-stage" style={{ height: 360 }}>
                  <div className="viewer-loading">
                    <div className="spinner" />
                  </div>
                </div>
              }
            >
              <PointCloudViewer
                points={points}
                height={360}
                label={t("res.cadrilleInput")}
              />
            </Suspense>
          ) : (
            <div className="viewer-stage cadrille-input-empty" style={{ height: 320 }}>
              <div className="viewer-loading">
                <div className="spinner" />
                {t("x.loading")}
              </div>
            </div>
          )
        ) : renderGridPath ? (
          <div className="cadrille-render-grid">
            <img src={api.fileUrl(jobId, renderGridPath)} alt={t("res.cadrilleRenderAlt")} />
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* ============================================================
   RunRecap  (Reconstruction tab)
   ============================================================ */
interface RunRecapProps {
  job: JobSummary;
  metrics: Metrics;
  postscale: boolean;
}
function RunRecap({ job, metrics, postscale }: RunRecapProps) {
  const { t } = useTranslation();
  const t0 = job.started_at ?? job.created_at;
  // Use ended_at (the true end) — NOT updated_at, which the backend bumps on
  // every read, making a finished job's elapsed grow to "now - start".
  // now-fallback ticks only while the job is still running.
  const nowSec = useNow(job.status === "running");
  const t1 = job.ended_at ?? nowSec;
  const totalSec = Math.max(0, t1 - t0);
  const term = job.status === "terminated" || job.status === "failed";

  return (
    <>
      <div className="run-head reveal">
        <span className="elapsed">
          <Icon
            n="activity"
            size={16}
            style={{ verticalAlign: "-3px", marginRight: 7, color: "var(--sig)" }}
          />
          {t("pipe.elapsed") + " "}
          <b>{fmtSec(totalSec)}</b>
        </span>
        <Chip tone={term ? "warn" : "ok"} icon={term ? "ban" : "checkCircle"}>
          {t(term ? "title.terminated" : "title.result")}
        </Chip>
      </div>
      <div className="reveal-2" style={{ marginBottom: 18 }}>
        <ProcessTimeline job={job} metrics={metrics} postscale={postscale} />
      </div>
    </>
  );
}


/* ============================================================
   LogPanel — full run log in the Reconstruction tab. Uses the shared
   LogConsole (terminal shown directly, no white wrapper panel) so it
   matches the Live view exactly; here the run is finished, so the header
   reads "Reconstruction log".
   ============================================================ */
function LogPanel({ jobId }: { jobId: string }) {
  const logsQ = useQuery({
    queryKey: ["logs", jobId],
    queryFn: () => api.getLogs(jobId),
  });
  const lines = (logsQ.data?.log ?? "").split("\n").filter((l) => l.length > 0);
  return (
    <div style={{ marginTop: 18 }}>
      <LogConsole lines={lines} live={false} />
    </div>
  );
}

/* ============================================================
   ResultView (exported)
   ============================================================ */
export function ResultView({ jobId }: ResultViewProps) {
  const { t } = useTranslation();
  const [tab, setTab] = useState<"run" | "results">("results");

  /* ---- data fetching ---- */
  const jobQ = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
  });

  const metricsQ = useQuery({
    queryKey: ["metrics", jobId],
    queryFn: () => api.getMetrics(jobId),
  });

  const inputsQ = useQuery({
    queryKey: ["inputs", jobId],
    queryFn: () => api.getJobInputs(jobId),
  });

  const job = jobQ.data;
  const metrics = metricsQ.data;

  const rp = job?.result_paths ?? null;

  /* cleanup metadata — only when path is present */
  const cleanupPath = rp?.cleanup_metadata as string | undefined;
  const cleanupQ = useQuery({
    queryKey: ["cleanupMeta", jobId, cleanupPath],
    queryFn: () => api.getCleanupMeta(jobId, cleanupPath!),
    enabled: !!cleanupPath,
  });

  /* scaled metadata — only when path is present */
  const scaledPath = rp?.scaled_metadata as string | undefined;
  const scaledMetaQ = useQuery({
    queryKey: ["scaledMeta", jobId, scaledPath],
    queryFn: () => api.getJsonFile<ScaledMetadata>(jobId, scaledPath!),
    enabled: !!scaledPath,
  });

  /* per-stage IoU/CD (canonical → cleaned → aligned) — only when path present */
  const stageMetricsPath = rp?.stage_metrics as string | undefined;
  const stageMetricsQ = useQuery({
    queryKey: ["stageMetrics", jobId, stageMetricsPath],
    queryFn: () => api.getJsonFile<StageMetrics>(jobId, stageMetricsPath!),
    enabled: !!stageMetricsPath,
  });

  /* ---- loading ---- */
  if (!job || !metrics) {
    return (
      <div className="canvas wide">
        <div className="run-head reveal">
          <span className="elapsed">
            <div className="spinner" style={{ margin: 0, display: "inline-block" }} />
          </span>
        </div>
      </div>
    );
  }

  const ps = !!(job.request?.postscale_enabled);
  const cleanup = cleanupQ.data ?? null;
  const scaledMeta = scaledMetaQ.data ?? null;
  const stageMetrics = stageMetricsQ.data ?? null;

  /* tab button */
  const tabBtn = (id: "run" | "results", icon: "activity" | "layers", label: string) => (
    <button
      className={tab === id ? "on" : ""}
      onClick={() => setTab(id)}
    >
      <Icon n={icon} size={16} className="ti" />
      {label}
    </button>
  );

  /* ---- results body ---- */
  const resultsBody = (
    <div className="stack gap-20" key="results">
      {/* 2×2 pipeline preview grid (result.jsx MeshPreviews) */}
      {rp ? (
        <MeshPreviews
          jobId={jobId}
          rp={rp}
          cleanup={cleanup}
          scaledMeta={scaledMeta}
          ps={ps}
        />
      ) : null}

      {/* Post-processing disclosure panel */}
      {(cleanup || (ps && scaledMeta)) ? (
        <PostProcessSection
          cleanup={cleanup}
          meta={ps ? scaledMeta : null}
        />
      ) : null}

      {/* Quality metrics */}
      <QualityMetrics metrics={metrics} stageMetrics={stageMetrics} />

      {/* Downloads */}
      <DownloadsSection jobId={jobId} job={job} />

      {/* Output folder */}
      <OutputFolder job={job} />
    </div>
  );

  /* ---- run recap body ---- */
  const runBody = (
    <div key="run">
      <RunRecap job={job} metrics={metrics} postscale={ps} />
      {/* Reconstruction log */}
      <LogPanel jobId={jobId} />
      {/* Uploaded inputs — moved here, directly under the reconstruction log */}
      <div style={{ marginTop: 18 }}>
        <InputImagesPanel
          jobId={jobId}
          inputImage={inputsQ.data?.input_image ?? null}
          inputMask={inputsQ.data?.input_mask ?? null}
        />
      </div>
      <div style={{ marginTop: 18 }}>
        <CadrilleInputPreview jobId={jobId} job={job} />
      </div>
    </div>
  );

  return (
    <div className="canvas wide">
      <div className="result-topbar reveal">
        <div className="page-tabs">
          {tabBtn("run",     "activity", t("tab.run"))}
          {tabBtn("results", "layers",   t("tab.results"))}
        </div>
      </div>

      {tab === "run" ? runBody : resultsBody}
    </div>
  );
}
