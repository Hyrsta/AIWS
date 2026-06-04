/* ============================================================
   Configure view — pixel-accurate port of AIWS Design Reference
   configure.jsx. Wired to real backend via useQuery + onStart.
   ============================================================ */
import { useState, useRef, useEffect, useCallback, Fragment } from "react";
import type { CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import type { Catalog, Health } from "@/api/types";
import type { ReconstructInput } from "@/api/types";
import { api } from "@/api/client";
import { Icon } from "@/components/Icon";
// Bundled demo pair — real pipeline inputs from a top-IoU reconstruction
// (cover_plate NEW-G140-52). Lets users try the pipeline without their own data.
import samplePhotoUrl from "@/assets/samples/sample_part.png";
import sampleMaskUrl from "@/assets/samples/sample_part_mask.png";

/* ---------- types ---------- */

interface ImageSlot {
  file: File;
  url: string;
  w: number;
  h: number;
  name: string;
}

interface MeshSlot {
  file: File;
  name: string;
  sizeBytes: number;
}

/* ---------- readImageFile ---------- */

function readImageFile(file: File): Promise<ImageSlot> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () =>
      resolve({
        file,
        url,
        w: img.naturalWidth,
        h: img.naturalHeight,
        name: file.name,
      });
    img.onerror = () =>
      resolve({
        file,
        url,
        w: 0,
        h: 0,
        name: file.name,
      });
    img.src = url;
  });
}

function readMeshFile(file: File): MeshSlot {
  return {
    file,
    name: file.name,
    sizeBytes: file.size,
  };
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

/* ---------- Toggle ---------- */

interface ToggleProps {
  on: boolean;
  onChange: (v: boolean) => void;
}

function Toggle({ on, onChange }: ToggleProps) {
  const knobStyle: CSSProperties = {
    position: "absolute",
    top: 2,
    left: on ? 20 : 2,
    width: 20,
    height: 20,
    borderRadius: "50%",
    background: on ? "var(--ink-000)" : "var(--tx-lo)",
    transition: "left .16s var(--ease)",
  };
  const trackStyle: CSSProperties = {
    width: 44,
    height: 26,
    borderRadius: 999,
    flex: "none",
    background: on ? "var(--sig)" : "var(--ink-300)",
    border: "1px solid " + (on ? "transparent" : "var(--line-2)"),
    position: "relative",
    transition: "background .16s, border-color .16s",
    padding: 0,
  };
  return (
    <button style={trackStyle} onClick={() => onChange(!on)}>
      <span style={knobStyle} />
    </button>
  );
}

/* ---------- Opt ---------- */

interface OptProps {
  title: string;
  desc: string;
  on: boolean;
  onClick: () => void;
}

function Opt({ title, desc, on, onClick }: OptProps) {
  return (
    <button className={"opt" + (on ? " on" : "")} onClick={onClick}>
      <b>{title}</b>
      <small>{desc}</small>
    </button>
  );
}

/* ---------- DropZone ---------- */

interface DropZoneProps {
  slot: "photo" | "mask";
  value: ImageSlot | null;
  onPick: (file: File) => void;
  onClear: () => void;
}

function DropZone({ slot, value, onPick, onClear }: DropZoneProps) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const handle = (files: FileList | null) => {
    if (files && files[0]) onPick(files[0]);
  };

  if (value) {
    return (
      <div className={"thumb" + (slot === "mask" ? " mask" : "")}>
        <div className="thumb-img">
          <img src={value.url} alt={slot} />
          <button
            className="x"
            title={t("upl.remove")}
            onClick={onClear}
          >
            <Icon n="x" size={20} sw={2.6} />
          </button>
        </div>
        <div className="thumb-cap">
          <b className="thumb-name" title={value.name}>{value.name}</b>
          <span className="dim">
            {value.w && value.h ? `${value.w}×${value.h}` : ""}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={"drop" + (over ? " over" : "")}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
    >
      <Icon n={slot === "mask" ? "scan" : "image"} className="di" />
      <b>{slot === "photo" ? t("upl.dropPhoto") : t("upl.dropMask")}</b>
      <small>{t("upl.formats")}</small>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        data-upload-kind={`${slot}-file`}
        style={{ display: "none" }}
        onChange={(e) => handle(e.target.files)}
      />
    </div>
  );
}

interface MeshDropZoneProps {
  value: MeshSlot | null;
  onPick: (file: File) => void;
  onClear: () => void;
}

function MeshDropZone({ value, onPick, onClear }: MeshDropZoneProps) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const handle = (files: FileList | null) => {
    if (files && files[0]) onPick(files[0]);
  };

  if (value) {
    return (
      <div className="thumb">
        <div className="thumb-img" style={{ display: "grid", placeItems: "center" }}>
          <Icon n="cube3d" size={58} style={{ color: "var(--sig)" }} />
          <button className="x" title={t("upl.remove")} onClick={onClear}>
            <Icon n="x" size={20} sw={2.6} />
          </button>
        </div>
        <div className="thumb-cap">
          <b className="thumb-name" title={value.name}>{value.name}</b>
          <span className="dim">{formatBytes(value.sizeBytes)}</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={"drop" + (over ? " over" : "")}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
    >
      <Icon n="cube3d" className="di" />
      <b>{t("upl.dropMesh")}</b>
      <small>{t("upl.meshFormats")}</small>
      <input
        ref={inputRef}
        type="file"
        accept=".stl,.glb,.obj,.ply"
        data-upload-kind="mesh-file"
        style={{ display: "none" }}
        onChange={(e) => handle(e.target.files)}
      />
    </div>
  );
}

/* ---------- ConfigureView ---------- */

interface ConfigureViewProps {
  onStart: (input: ReconstructInput) => void;
}

export function ConfigureView({ onStart }: ConfigureViewProps) {
  const { t } = useTranslation();

  const catalogQ = useQuery<Catalog>({
    queryKey: ["catalog"],
    queryFn: api.catalog,
  });
  const catalog = catalogQ.data;

  // GPU list for the device picker (from /health). Refreshed periodically so
  // the free-memory readout stays roughly current on this shared box.
  const healthQ = useQuery<Health>({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15000,
  });
  const gpus = healthQ.data?.gpus ?? [];
  const gpuLoading = healthQ.isFetching;
  const gpuTs = healthQ.dataUpdatedAt || null;
  // Device name + total VRAM for the section-header pill (all GPUs share a model here).
  const gpu0 = gpus[0];
  const commonGpuName =
    gpus.length > 0 && gpus.every((g) => g.name === gpu0.name) ? gpu0.name : null;
  const gpuTotalGb = gpu0 ? Math.round(gpu0.memory_total_mb / 1024) : null;
  const gpuDeviceLabel =
    gpuTotalGb != null
      ? commonGpuName
        ? `${commonGpuName} · ${gpuTotalGb} GB`
        : `${gpuTotalGb} GB`
      : null;
  const fmtClock = (ms: number) => {
    const d = new Date(ms);
    const p = (n: number) => String(n).padStart(2, "0");
    return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  };

  const [inputMode, setInputMode] = useState<"image_mask" | "mesh">("image_mask");
  const [photo, setPhoto] = useState<ImageSlot | null>(null);
  const [mask, setMask] = useState<ImageSlot | null>(null);
  const [mesh, setMesh] = useState<MeshSlot | null>(null);
  const [ckpt, setCkpt] = useState<"RL" | "SFT">("RL");
  const [mode, setMode] = useState<"PC" | "IMG">("PC");
  const [gpu, setGpu] = useState<string>(""); // "" until a GPU is auto-selected
  const [gpuTouched, setGpuTouched] = useState(false); // user picked manually?
  const [psOn, setPsOn] = useState(false);
  const [wclass, setWclass] = useState("");
  const [model, setModel] = useState("");
  const [sampleBusy, setSampleBusy] = useState(false); // loading the demo pair?

  // Auto-select the least-busy GPU (most free memory) once the list loads, as
  // long as the user hasn't picked one. Keeps a sensible default without a
  // dedicated "Auto" cell.
  useEffect(() => {
    if (gpuTouched || gpus.length === 0) return;
    const best = [...gpus].sort((a, b) => b.memory_free_mb - a.memory_free_mb)[0];
    setGpu(String(best.index));
  }, [gpus, gpuTouched]);
  const pickGpu = (idx: string) => { setGpuTouched(true); setGpu(idx); };

  /* revoke object URLs on clear / unmount to avoid leaks */
  const revokeSlot = useCallback((slot: ImageSlot | null) => {
    if (slot) URL.revokeObjectURL(slot.url);
  }, []);

  useEffect(() => {
    return () => {
      if (photo) URL.revokeObjectURL(photo.url);
      if (mask) URL.revokeObjectURL(mask.url);
    };
    // eslint intentional: only runs on unmount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const classes = catalog ? Object.keys(catalog.classes) : [];
  const models =
    psOn && wclass && catalog ? catalog.classes[wclass].models : [];
  const targetMm =
    psOn && wclass && model && catalog
      ? catalog.classes[wclass].entries[model]?.bbox_mm ?? null
      : null;

  const dimMatch =
    photo &&
    mask &&
    photo.w > 0 &&
    mask.w > 0 &&
    photo.w === mask.w &&
    photo.h === mask.h;

  const dimMismatch =
    photo &&
    mask &&
    photo.w > 0 &&
    mask.w > 0 &&
    !(photo.w === mask.w && photo.h === mask.h);

  const imageReady = photo !== null && mask !== null && !dimMismatch;
  const meshReady = mesh !== null;
  const ready =
    (inputMode === "mesh" ? meshReady : imageReady) &&
    (!psOn || (wclass !== "" && model !== ""));

  const pickPhoto = async (f: File) => setPhoto(await readImageFile(f));
  const pickMask = async (f: File) => setMask(await readImageFile(f));
  const pickMesh = (f: File) => setMesh(readMeshFile(f));

  // Populate both slots with the bundled demo pair. Fetches the bundled assets
  // and rebuilds real File objects (onStart requires actual Files, not URLs),
  // then loads them like a normal upload. The pair is dimension-matched, so the
  // green "dimensions match" validation lights up automatically.
  const useSample = async () => {
    if (sampleBusy) return;
    setSampleBusy(true);
    try {
      const [pBlob, mBlob] = await Promise.all([
        fetch(samplePhotoUrl).then((r) => r.blob()),
        fetch(sampleMaskUrl).then((r) => r.blob()),
      ]);
      const [pSlot, mSlot] = await Promise.all([
        readImageFile(new File([pBlob], "sample_part.png", { type: "image/png" })),
        readImageFile(new File([mBlob], "sample_part_mask.png", { type: "image/png" })),
      ]);
      revokeSlot(photo); // drop any prior object URLs before replacing
      revokeSlot(mask);
      setPhoto(pSlot);
      setMask(mSlot);
    } catch (err) {
      console.error("[useSample] failed to load sample pair", err);
    } finally {
      setSampleBusy(false);
    }
  };

  const clearPhoto = () => {
    revokeSlot(photo);
    setPhoto(null);
  };
  const clearMask = () => {
    revokeSlot(mask);
    setMask(null);
  };
  const clearMesh = () => setMesh(null);

  const handleWclassChange = (v: string) => {
    setWclass(v);
    setModel("");
  };

  const onStartClick = () => {
    if (!ready) return;
    const base = {
      cadrille_checkpoint_preset: ckpt,
      cadrille_mode: mode,
      workpiece_class: psOn ? wclass : null,
      model_code: psOn ? model : null,
      gpu_index: gpu === "" ? null : Number(gpu),
    };
    if (inputMode === "mesh") {
      if (!mesh) return;
      onStart({ input_mode: "mesh", mesh: mesh.file, ...base });
      return;
    }
    if (!photo || !mask) return;
    onStart({
      input_mode: "image_mask",
      image: photo.file,
      mask: mask.file,
      ...base,
    });
  };

  return (
    <div className="canvas">
      {/* step strip — anchors the page (replaces the orphaned floating subtitle) */}
      <div className="cfg-steps reveal">
        {([
          { n: "01", icon: "image" as const, title: t("cfg.step1"), sub: t("cfg.step1sub") },
          { n: "02", icon: "sliders" as const, title: t("cfg.step2"), sub: t("cfg.step2sub") },
          { n: "03", icon: "play" as const, title: t("cfg.step3"), sub: t("cfg.step3sub") },
        ]).map((s, i, arr) => (
          <Fragment key={s.n}>
            <div className="cfg-step">
              <span className="sn">{s.n}</span>
              <span className="si"><Icon n={s.icon} size={16} /></span>
              <span className="st">
                <b>{s.title}</b>
                <span>{s.sub}</span>
              </span>
            </div>
            {i < arr.length - 1 ? (
              <span className="cfg-step-sep"><Icon n="chevR" size={15} /></span>
            ) : null}
          </Fragment>
        ))}
      </div>

      <div className="cfg-grid">
        {/* ---- left: inputs ---- */}
        <div
          className={
            "panel reveal-2 cfg-input-panel" +
            (inputMode === "image_mask" ? " has-sample-action" : "")
          }
        >
          <div className="panel-head">
            <div>
              <h3>{t("upl.inputs")}</h3>
              <p>{t(inputMode === "mesh" ? "upl.meshFormats" : "upl.formats")}</p>
            </div>
            {inputMode === "image_mask" ? (
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => { void useSample(); }}
                disabled={sampleBusy}
              >
                <Icon
                  n={sampleBusy ? "refresh" : "image"}
                  size={14}
                  className={sampleBusy ? "spin-ico" : undefined}
                />
                {t("upl.usePair")}
              </button>
            ) : null}
          </div>

          <div className={"panel-pad" + (inputMode === "mesh" ? " mesh-input-pad" : "")}>
            <div className="field">
              <label>{t("upl.inputSource")}</label>
              <div className="opt-row">
                <Opt
                  title={t("upl.imageMask")}
                  desc={t("upl.imageMaskDesc")}
                  on={inputMode === "image_mask"}
                  onClick={() => setInputMode("image_mask")}
                />
                <Opt
                  title={t("upl.meshInput")}
                  desc={t("upl.meshInputDesc")}
                  on={inputMode === "mesh"}
                  onClick={() => setInputMode("mesh")}
                />
              </div>
            </div>

            {inputMode === "image_mask" ? (
              <>
                <div className="upload-pair">
                  <div>
                    <div className="field" style={{ marginBottom: 8 }}>
                      <label>
                        <Icon n="image" size={14} />
                        {t("upl.photo")}
                      </label>
                    </div>
                    <DropZone
                      slot="photo"
                      value={photo}
                      onPick={(f) => { void pickPhoto(f); }}
                      onClear={clearPhoto}
                    />
                  </div>
                  <div>
                    <div className="field" style={{ marginBottom: 8 }}>
                      <label>
                        <Icon n="scan" size={14} />
                        {t("upl.mask")}
                      </label>
                    </div>
                    <DropZone
                      slot="mask"
                      value={mask}
                      onPick={(f) => { void pickMask(f); }}
                      onClear={clearMask}
                    />
                  </div>
                </div>

                {dimMatch ? (
                  <div className="validate ok mt-16">
                    <Icon n="checkCircle" size={16} />
                    {t("upl.dimMatch")}
                  </div>
                ) : null}
                {dimMismatch ? (
                  <div className="validate bad mt-16">
                    <Icon n="alert" size={16} />
                    {t("upl.dimMismatch")}
                  </div>
                ) : null}
              </>
            ) : (
              <>
                <div className="field" style={{ marginBottom: 8 }}>
                  <label>
                    <Icon n="cube3d" size={14} />
                    {t("upl.mesh")}
                  </label>
                </div>
                <div className="mesh-fill-area">
                  <MeshDropZone
                    value={mesh}
                    onPick={pickMesh}
                    onClear={clearMesh}
                  />
                </div>
              </>
            )}
          </div>
        </div>

        {/* ---- right: settings ---- */}
        <div className="stack gap-20">
          <div className="panel reveal-2 cfg-settings-panel">
            <div className="panel-head">
              <div>
                <h3>{t("set.cadrille")}</h3>
              </div>
            </div>
            <div className="panel-pad">
              <div className="field">
                <label>{t("set.checkpoint")}</label>
                <div className="opt-row">
                  <Opt
                    title="RL"
                    desc={t("ck.rl.desc")}
                    on={ckpt === "RL"}
                    onClick={() => setCkpt("RL")}
                  />
                  <Opt
                    title="SFT"
                    desc={t("ck.sft.desc")}
                    on={ckpt === "SFT"}
                    onClick={() => setCkpt("SFT")}
                  />
                </div>
              </div>
              <div className="field">
                <label>{t("set.modality")}</label>
                <div className="opt-row">
                  <Opt
                    title="PC"
                    desc={t("md.pc.desc")}
                    on={mode === "PC"}
                    onClick={() => setMode("PC")}
                  />
                  <Opt
                    title="IMG"
                    desc={t("md.img.desc")}
                    on={mode === "IMG"}
                    onClick={() => setMode("IMG")}
                  />
                </div>
                <p className="hint cfg-settings-hint" style={{ margin: "8px 0 0" }}>
                  {t(mode === "IMG" ? "md.hint.img" : "md.hint.pc")}
                </p>
              </div>
              <div className="field" style={{ marginBottom: 0 }}>
                <div className="gpu-label-row">
                  <label style={{ margin: 0 }}>
                    <Icon n="cpu" size={14} />
                    {t("set.gpu")}
                    {gpuDeviceLabel ? (
                      <span className="gpu-device-name mono">{gpuDeviceLabel}</span>
                    ) : null}
                  </label>
                  <div className="gpu-status-meta">
                    <span className="gpu-ts">
                      {gpuLoading
                        ? t("set.gpuRefreshing")
                        : gpuTs
                          ? t("set.gpuUpdated", { v: fmtClock(gpuTs) })
                          : t("set.gpuNever")}
                    </span>
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm gpu-refresh"
                      disabled={gpuLoading}
                      onClick={() => { void healthQ.refetch(); }}
                    >
                      <Icon n="refresh" size={14} className={gpuLoading ? "spin-ico" : undefined} />
                      {t("set.gpuRefresh")}
                    </button>
                  </div>
                </div>
                <div className="gpu-seg">
                  {/* One card per GPU. The least-busy is auto-selected on load
                      (see effect); the user can override. */}
                  {gpus.map((g) => {
                    const usedMb = g.memory_total_mb - g.memory_free_mb;
                    const memFrac = g.memory_total_mb > 0 ? usedMb / g.memory_total_mb : 0;
                    const memTone = memFrac >= 0.85 ? " hot" : memFrac >= 0.55 ? " warm" : "";
                    const util = Math.round(g.utilization);
                    const idle = util <= 1 && memFrac < 0.1;
                    const sel = gpu === String(g.index);
                    return (
                      <button
                        key={g.index}
                        type="button"
                        className={"gpu-cell" + (sel ? " on" : "") + (gpuLoading ? " loading" : "")}
                        aria-pressed={sel}
                        onClick={() => pickGpu(String(g.index))}
                      >
                        <div className="gpu-cell-top">
                          <span className="gpu-id">
                            <span className="gpu-pre">cuda:</span>
                            <span className="gpu-n">{g.index}</span>
                          </span>
                          {idle ? <span className="gpu-idle">{t("set.gpuFree")}</span> : null}
                        </div>
                        {/* compute utilisation — steady teal, value in % */}
                        <div className="gpu-row">
                          <span className="gpu-rl">{t("set.gpuUtil")}</span>
                          <div className="gpu-bar">
                            <span
                              className="gpu-bar-fill util"
                              style={{ width: Math.max(util > 0 ? 2 : 0, util) + "%" }}
                            />
                          </div>
                          <span className="gpu-rv">{util}%</span>
                        </div>
                        {/* VRAM — ramps neutral→warm→hot, value in GB (used / total) */}
                        <div className="gpu-row">
                          <span className="gpu-rl">{t("set.gpuVram")}</span>
                          <div className="gpu-bar">
                            <span
                              className={"gpu-bar-fill" + memTone}
                              style={{ width: Math.max(2, Math.round(memFrac * 100)) + "%" }}
                            />
                          </div>
                          <span className={"gpu-rv" + memTone}>
                            {(usedMb / 1024).toFixed(1)} / {Math.round(g.memory_total_mb / 1024)}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* ---- post-scale ---- */}
          {/*
           * The body is ALWAYS rendered so toggling the panel never changes
           * the column height (which would otherwise stretch the Inputs panel
           * and balloon the dropzones). When off, the controls are shown in a
           * dimmed, non-interactive preview state so the layout is symmetric +
           * stable.
           */}
          <div className="panel reveal-3 cfg-post-panel">
            <div className="panel-head">
              <div>
                <h3>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <Icon n="ruler" size={15} style={{ color: "var(--sig)" }} />
                    {t("ps.title")}
                  </span>
                </h3>
                <p>{t("ps.sub")}</p>
              </div>
              <Toggle on={psOn} onChange={setPsOn} />
            </div>
            <div
              className={"panel-pad ps-body" + (psOn ? "" : " is-off")}
              aria-hidden={psOn ? undefined : "true"}
            >
              <div
                className="opt-row"
                style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}
              >
                <div className="field" style={{ marginBottom: 0 }}>
                  <label>{t("ps.class")}</label>
                  <select
                    className="sel"
                    value={wclass}
                    disabled={!psOn}
                    tabIndex={psOn ? undefined : -1}
                    onChange={(e) => handleWclassChange(e.target.value)}
                  >
                    <option value="">—</option>
                    {classes.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field" style={{ marginBottom: 0 }}>
                  <label>{t("ps.model")}</label>
                  <select
                    className="sel"
                    value={model}
                    disabled={!psOn || !wclass}
                    tabIndex={psOn ? undefined : -1}
                    onChange={(e) => setModel(e.target.value)}
                  >
                    <option value="">—</option>
                    {models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Readout row is always present (height-stable). Shows the
                  catalog target when resolved, else a muted placeholder. */}
              <div
                className={
                  "bbox-readout" + (psOn && targetMm ? "" : " muted")
                }
              >
                <Icon
                  n="target"
                  size={14}
                  style={{
                    color:
                      psOn && targetMm ? "var(--sig)" : "var(--tx-dim)",
                  }}
                />
                {psOn && targetMm ? (
                  <>
                    <span>{t("ps.target")}:</span>
                    <span className="v">{targetMm.join(" × ")} mm</span>
                  </>
                ) : (
                  <span>
                    {psOn ? `${t("ps.target")}: —` : t("ps.skip")}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ---- action bar ---- */}
      <div
        className="row reveal-3"
        style={{ marginTop: 24, justifyContent: "flex-end" }}
      >
        <button
          className="btn btn-primary lg"
          disabled={!ready}
          onClick={onStartClick}
        >
          <Icon n="play" size={16} />
          {t("act.start")}
        </button>
      </div>
    </div>
  );
}
