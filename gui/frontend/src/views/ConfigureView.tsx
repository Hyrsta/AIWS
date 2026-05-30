/* ============================================================
   Configure view — pixel-accurate port of AIWS Design Reference
   configure.jsx. Wired to real backend via useQuery + onStart.
   ============================================================ */
import { useState, useRef, useEffect, useCallback } from "react";
import type { CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import type { Catalog } from "@/api/types";
import type { ReconstructInput } from "@/api/types";
import { api } from "@/api/client";
import { Icon } from "@/components/Icon";

/* ---------- types ---------- */

interface ImageSlot {
  file: File;
  url: string;
  w: number;
  h: number;
  name: string;
}

/* ---------- readImageFile ---------- */

function readImageFile(file: File): Promise<ImageSlot> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () =>
      resolve({ file, url, w: img.naturalWidth, h: img.naturalHeight, name: file.name });
    img.onerror = () =>
      resolve({ file, url, w: 0, h: 0, name: file.name });
    img.src = url;
  });
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
            <Icon n="x" size={14} />
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

  const [photo, setPhoto] = useState<ImageSlot | null>(null);
  const [mask, setMask] = useState<ImageSlot | null>(null);
  const [ckpt, setCkpt] = useState<"RL" | "SFT">("RL");
  const [mode, setMode] = useState<"PC" | "IMG">("PC");
  const [psOn, setPsOn] = useState(false);
  const [wclass, setWclass] = useState("");
  const [model, setModel] = useState("");

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

  const ready =
    photo !== null &&
    mask !== null &&
    !dimMismatch &&
    (!psOn || (wclass !== "" && model !== ""));

  const pickPhoto = async (f: File) => setPhoto(await readImageFile(f));
  const pickMask = async (f: File) => setMask(await readImageFile(f));

  const clearPhoto = () => {
    revokeSlot(photo);
    setPhoto(null);
  };
  const clearMask = () => {
    revokeSlot(mask);
    setMask(null);
  };

  const handleWclassChange = (v: string) => {
    setWclass(v);
    setModel("");
  };

  const onStartClick = () => {
    if (!ready || !photo || !mask) return;
    onStart({
      image: photo.file,
      mask: mask.file,
      cadrille_checkpoint_preset: ckpt,
      cadrille_mode: mode,
      workpiece_class: psOn ? wclass : null,
      model_code: psOn ? model : null,
    });
  };

  return (
    <div className="canvas">
      <p className="section-sub reveal" style={{ marginTop: 0 }}>
        {t("sub.configure")}
      </p>

      <div className="cfg-grid">
        {/* ---- left: inputs ---- */}
        <div className="panel reveal-2">
          <div className="panel-head">
            <div>
              <h3>{t("upl.inputs")}</h3>
              <p>{t("upl.formats")}</p>
            </div>
            {/* "Use sample pair" button removed — no SAMPLE_PAIR fallback */}
          </div>

          <div className="panel-pad">
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
          </div>
        </div>

        {/* ---- right: settings ---- */}
        <div className="stack gap-20">
          <div className="panel reveal-2">
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
              <div className="field" style={{ marginBottom: 0 }}>
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
          <div className="panel reveal-3">
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
