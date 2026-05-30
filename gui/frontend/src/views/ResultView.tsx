/* ============================================================
   ResultView — pixel-accurate port of AIWS Design Reference
   aiws/result.jsx.  Wired to real backend via @tanstack/react-query.
   ============================================================ */
import { useState, lazy, Suspense } from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { fmtIoU } from "@/lib/format";
import { Icon, Chip } from "@/components/Icon";
import type { CleanupMetadata, ScaledMetadata, JobSummary, Metrics } from "@/api/types";

/* ---- lazy-load the 3-D viewer to keep it in its own chunk ---- */
const MeshViewer = lazy(() =>
  import("@/components/MeshViewer").then((m) => ({ default: m.MeshViewer }))
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
  ps:   { bar: "#b5673a", soft: "rgba(181,103,58,0.20)", cT: "#9a5530", cBg: "rgba(181,103,58,0.10)", cBd: "rgba(181,103,58,0.34)" },
};

const PTL_BACKEND_LABELS = [
  "SAM3D: Loading checkpoints",
  "SAM3D: Generating mesh",
  "Cadrille: Preparing input",
  "Cadrille: Generating CAD result",
  "Post-scaling: Aligning CAD to catalog (mm)",
];

interface PtlRow {
  key: string;
  icon: "download" | "box" | "layers" | "cube3d" | "ruler";
  tone: keyof typeof PTL_TONE;
  kind: "gpu" | "cpu";
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

  const rows: PtlRow[] = [
    { key: "s1", icon: "download", tone: "sam", kind: "gpu", sec: sam?.model_init_sec ?? null, mem: null },
    { key: "s2", icon: "box",      tone: "sam", kind: "gpu", sec: sam?.duration_sec   ?? null, mem: sam ? { r: sam.peak_memory_reserved_mb ?? null, a: sam.peak_memory_allocated_mb ?? null } : null },
    { key: "s3", icon: "layers",   tone: "cad", kind: "cpu", sec: null,                         mem: null },
    { key: "s4", icon: "cube3d",   tone: "cad", kind: "gpu", sec: cad?.duration_sec   ?? null, mem: cad ? { r: cad.peak_memory_reserved_mb ?? null, a: cad.peak_memory_allocated_mb ?? null } : null },
  ];
  if (postscale) rows.push({ key: "s5", icon: "ruler", tone: "ps", kind: "cpu", sec: null, mem: null });

  const totalSec = Math.max(1, rows.reduce((a, r) => a + (r.sec ?? 0), 0));
  const vramMb = cad?.device_total_memory_mb
    ?? (Math.max(1, ...rows.map((r) => r.mem?.r ?? 0)) * 1.18);
  const peakMem = Math.max(0, ...rows.map((r) => r.mem?.r ?? 0));

  const failed = job.status === "failed" || job.status === "terminated";
  const failedIdx = failed ? Math.max(0, PTL_BACKEND_LABELS.indexOf(job.stage_label)) : -1;
  let cur: number;
  if (job.status === "completed") cur = 99;
  else if (failed) cur = failedIdx;
  else cur = PTL_BACKEND_LABELS.indexOf(job.stage_label);

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
                <div className="fill" style={{ width: (r.sec / totalSec * 100) + "%", background: tone.bar }} />
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
                <div className="nm">{t("stage." + r.key)}</div>
                <div className="meta">
                  <span className="ptl-mod" style={{ color: tone.cT, background: tone.cBg, borderColor: tone.cBd }}>
                    {t("stage." + r.key + "g")}
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
            <b style={{ background: "var(--tx-md)" }} />
            {t("rt.allocated")}
          </span>
          <span className="lg">
            <b style={{ background: "var(--ink-300)" }} />
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
}
function QualityMetrics({ metrics }: QualityMetricsProps) {
  const { t } = useTranslation();
  const cad = metrics.cadrille;
  if (!cad?.available) return null;

  const iouStr = cad.mean_iou != null ? fmtIoU(cad.mean_iou) : "—";
  const cdStr  = cad.median_cd != null ? (Number(cad.median_cd) * 1e3).toFixed(3) : "—";

  return (
    <div className="panel panel-pad reveal-2">
      <div className="section-title" style={{ marginBottom: 4 }}>
        <span className="ix"><Icon n="target" size={16} /></span>
        <h2 style={{ fontSize: 17 }}>{t("res.quality")}</h2>
      </div>
      <p className="section-sub" style={{ margin: "0 0 16px" }}>{t("res.qualitySub")}</p>
      <MetricGroup title={t("m.quality")} icon="cube3d">
        <Metric label={t("m.meanIou")} value={iouStr} sig hint={t("m.iouHint")} />
        <Metric label={t("m.medianCd")} value={cdStr} hint={t("m.cdHint")} />
      </MetricGroup>
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
  const removedPct =
    cleanup.removed_volume_fraction != null
      ? (cleanup.removed_volume_fraction * 100).toFixed(1) + "%"
      : "—";

  return (
    <>
      <div>
        <div className="metric-group-label">
          <Icon n="box" size={14} style={{ color: "var(--sig)" }} />
          {t("cl.removedVol")}
        </div>
        <div className="pp-bignum">{removedPct}</div>
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
          <div className="eyebrow">{t("res.postEyebrow")}</div>
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
    <div className="panel panel-pad reveal-3">
      <div className="section-title" style={{ marginBottom: 4 }}>
        <span className="ix"><Icon n="download" size={16} /></span>
        <h2 style={{ fontSize: 17 }}>{t("res.downloads")}</h2>
      </div>
      <p className="section-sub" style={{ margin: "0 0 16px" }}>{t("dl.note")}</p>
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
  );
}

/* ============================================================
   InputImagesPanel — "Uploaded inputs" panel (NEW)
   Shows the photo + mask that were submitted with the job.
   ============================================================ */
interface InputImagesPanelProps {
  jobId: string;
  inputImage: string | null;
  inputMask: string | null;
}
function InputImagesPanel({ jobId, inputImage, inputMask }: InputImagesPanelProps) {
  const { t } = useTranslation();
  if (!inputImage && !inputMask) return null;

  return (
    <div className="panel reveal-2">
      <div className="panel-head">
        <div>
          <h3>{t("res.uploaded")}</h3>
          <p>{t("res.uploadedSub")}</p>
        </div>
      </div>
      <div className="panel-pad">
        <div className="upload-pair">
          {inputImage ? (
            <div className="thumb">
              <div className="thumb-img">
                <img src={api.fileUrl(jobId, inputImage)} alt={t("upl.photo")} />
              </div>
              <div className="thumb-cap">
                <b>{t("upl.photo")}</b>
                <span className="dim thumb-name" title={inputImage}>
                  {inputImage.split("/").pop()}
                </span>
              </div>
            </div>
          ) : null}
          {inputMask ? (
            <div className="thumb mask">
              <div className="thumb-img">
                <img src={api.fileUrl(jobId, inputMask)} alt={t("upl.mask")} />
              </div>
              <div className="thumb-cap">
                <b>{t("upl.mask")}</b>
                <span className="dim thumb-name" title={inputMask}>
                  {inputMask.split("/").pop()}
                </span>
              </div>
            </div>
          ) : null}
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
    } catch (_) {
      /* ignore */
    }
  };

  return (
    <div className="panel panel-pad reveal-3">
      <div className="spread" style={{ marginBottom: 12 }}>
        <div className="metric-group-label" style={{ margin: 0 }}>
          <Icon n="folder" size={14} style={{ color: "var(--sig)" }} />
          {t("res.outFolder")}
        </div>
        <button className="btn btn-ghost btn-sm" onClick={copy}>
          <Icon n="copy" size={14} />
          {t("x.copy")}
        </button>
      </div>
      <div className="path-box">
        <Icon n="terminal" size={14} />
        {root}
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
    cells.push({
      id: "sam3d",
      path: rp.sam3d_mesh_stl as string,
      label: t("v.sam3d"),
      units: t("v.units.mesh"),
      sw: "#8aa0b8",
      tone: "info",
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
                  <span
                    className="sw"
                    style={{ width: 10, height: 10, borderRadius: 2, background: x.sw }}
                  />
                  {x.label}
                </b>
                <div className="cap-chips">
                  <Chip tone={x.tone} icon={x.pillIcon ?? "cube3d"}>
                    {x.pill ?? x.units}
                  </Chip>
                  {x.bodies ? <Chip icon="layers">{x.bodies}</Chip> : null}
                </div>
              </div>
              <Suspense
                fallback={
                  <div className="viewer-stage" style={{ height: 268 }}>
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
                  height={268}
                  label={x.label}
                  badge={x.badgeN}
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
  const t1 = job.ended_at ?? Date.now() / 1000;
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
   LogPanel — full run log in the Reconstruction tab
   ============================================================ */
function logLineClass(ln: string): string {
  if (/\[(error|fail)/i.test(ln)) return "ln lv-error";
  if (/\[warn/i.test(ln)) return "ln lv-warn";
  if (/\[ok\]|\bdone\b|✓/i.test(ln)) return "ln lv-ok";
  if (/\[stage\]|\[info\]/i.test(ln)) return "ln lv-info";
  return "ln";
}
function LogPanel({ jobId }: { jobId: string }) {
  const { t } = useTranslation();
  const logsQ = useQuery({
    queryKey: ["logs", jobId],
    queryFn: () => api.getLogs(jobId),
  });
  const lines = (logsQ.data?.log ?? "").split("\n").filter((l) => l.length > 0);
  return (
    <div className="panel reveal-3" style={{ marginTop: 18 }}>
      <div className="panel-head">
        <div>
          <h3>{t("res.log")}</h3>
          <p>{t("res.logSub")}</p>
        </div>
      </div>
      <div className="panel-pad">
        <div className="log-wrap">
          <div className="log-head">
            <span className="dot-row" aria-hidden="true">
              <span className="d" style={{ background: "#ff5f57" }} />
              <span className="d" style={{ background: "#febc2e" }} />
              <span className="d" style={{ background: "#28c840" }} />
            </span>
            <span style={{ marginLeft: 6 }}>{t("log.title")}</span>
          </div>
          <div className="log">
            {lines.length === 0 ? (
              <div className="ln" style={{ opacity: 0.5 }}>—</div>
            ) : (
              lines.map((ln, i) => (
                <div key={i} className={logLineClass(ln)}>{ln}</div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   ResultView (exported)
   ============================================================ */
export function ResultView({ jobId, onNew }: ResultViewProps) {
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
      <QualityMetrics metrics={metrics} />

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
    </div>
  );

  return (
    <div className="canvas wide">
      <div className="result-topbar reveal">
        <div className="page-tabs">
          {tabBtn("run",     "activity", t("tab.run"))}
          {tabBtn("results", "layers",   t("tab.results"))}
        </div>
        <button className="btn" onClick={onNew}>
          <Icon n="plus" size={15} />
          {t("act.again")}
        </button>
      </div>

      {tab === "run" ? runBody : resultsBody}
    </div>
  );
}
