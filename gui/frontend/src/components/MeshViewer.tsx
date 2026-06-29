/* ============================================================
   MeshViewer — interactive r3f 3-D preview, pixel-accurate port
   of viewer.jsx panel chrome.  Lazy-loaded by ResultView.
   ============================================================ */
import { Suspense, useEffect, useMemo, useCallback, useState } from "react";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import { OrbitControls, Center } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Icon } from "@/components/Icon";
import { useTranslation } from "react-i18next";
import type { BufferGeometry, PerspectiveCamera } from "three";

/* ---------- shared auto-spin clock ----------
   Every viewer derives its azimuth from ONE wall clock, so all four preview
   canvases spin in phase ("when one is in front, the rest are too") regardless
   of when each mounted or finished loading. Manual interaction detaches a
   viewer (its spin stops); the reset button re-attaches it — snapping back into
   the shared phase so it lines up with the others again. */
const FOV = 38;
const POLAR = 1.05; // camera elevation (reference viewer phi) — kept above the grid
const BASE_AZIMUTH = 0.7; // 3/4-view starting azimuth (reference viewer theta)
const ROT_SPEED = 0.22; // rad/s — gentle shared spin rate
const FRAME_MARGIN = 1.32; // distance padding around the object (↑ = further away)
const GRID_SIZE_MULT = 4.0; // grid size relative to the mesh's largest dimension (incl. height) — matches reference's expansive floor
const ROT_T0 = typeof performance !== "undefined" ? performance.now() : 0;

function sharedAzimuth(): number {
  return BASE_AZIMUTH + ((performance.now() - ROT_T0) / 1000) * ROT_SPEED;
}

/* ---------- mesh stats shape ---------- */

interface MeshStats {
  verts: number;
  faces: number;
  extStr: string;
}

/* ---------- camera rig: shared spin + size-calibrated framing ----------
   Owns the camera while `synced`. Frames the object at a distance derived from
   its bounding radius (so it fills the view consistently at any scale — mm,
   canonical, or normalized units) and advances only the azimuth from the shared
   clock, leaving the calibrated radius + elevation fixed. When detached it does
   nothing and OrbitControls drives freely. */
function SpinRig({ synced, dist }: { synced: boolean; dist: number }) {
  const camera = useThree((s) => s.camera) as PerspectiveCamera;
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null;

  // Clip planes calibrated to the framing distance so any unit scale stays in view.
  // Mutating the three.js camera in an effect is the idiomatic r3f pattern; the
  // react-hooks/immutability compiler check flags it conservatively here.
  useEffect(() => {
    /* eslint-disable react-hooks/immutability */
    camera.near = Math.max(dist / 100, 0.001);
    camera.far = dist * 100;
    camera.updateProjectionMatrix();
    /* eslint-enable react-hooks/immutability */
  }, [camera, dist]);

  useFrame(() => {
    if (!synced || !controls) return;
    const az = sharedAzimuth();
    const sinP = Math.sin(POLAR);
    controls.target.set(0, 0, 0); // object is Center'd at the origin
    camera.position.set(
      dist * sinP * Math.sin(az),
      dist * Math.cos(POLAR),
      dist * sinP * Math.cos(az),
    );
    camera.lookAt(0, 0, 0);
    controls.update(); // keep controls' internal state in sync for smooth manual takeover
  });

  return null;
}

/* ---------- inner scene ---------- */

function Scene({
  buf,
  color,
  synced,
  onStats,
}: {
  buf: ArrayBuffer;
  color: string;
  synced: boolean;
  onStats: (s: MeshStats) => void;
}) {
  const geo = useMemo<BufferGeometry>(() => {
    const g = new STLLoader().parse(buf);
    g.computeVertexNormals();
    g.computeBoundingBox();
    return g;
  }, [buf]);

  // Framing distance + floor-grid sizing, derived from the bounding box so they
  // scale with any unit system. The grid is laid just beneath the object's base
  // so the part reads as resting on a surface (mirrors the reference viewer's
  // THREE.GridHelper floor) rather than floating in space.
  const { dist, gridSize, gridY } = useMemo(() => {
    const bb = geo.boundingBox;
    if (!bb) return { dist: 3, gridSize: 2, gridY: -0.5 };
    const w = bb.max.x - bb.min.x;
    const h = bb.max.y - bb.min.y;
    const d = bb.max.z - bb.min.z;
    const radius = 0.5 * Math.sqrt(w * w + h * h + d * d) || 1; // sphere around bbox center
    const maxDim = Math.max(w, h, d) || 1; // largest extent — drives the floor size
    return {
      dist: (radius / Math.sin(((FOV * Math.PI) / 180) / 2)) * FRAME_MARGIN,
      gridSize: maxDim * GRID_SIZE_MULT, // scales with the whole mesh, not just its footprint
      gridY: -h / 2 - maxDim * 0.02,
    };
  }, [geo]);

  useEffect(() => {
    /* compute stats once the geometry is ready (true extents, unscaled) */
    geo.computeBoundingBox();
    const bb = geo.boundingBox;
    let extStr = "—";
    if (bb) {
      const sx = (bb.max.x - bb.min.x).toFixed(2);
      const sy = (bb.max.y - bb.min.y).toFixed(2);
      const sz = (bb.max.z - bb.min.z).toFixed(2);
      extStr = `${sx} × ${sy} × ${sz}`;
    }
    const pos = geo.attributes.position;
    const verts = pos ? pos.count : 0;
    const faces = geo.index
      ? Math.round(geo.index.count / 3)
      : pos
        ? Math.round(pos.count / 3)
        : 0;
    onStats({ verts, faces, extStr });
  }, [geo, onStats]);

  useEffect(() => () => geo.dispose(), [geo]);

  return (
    <>
      <Center>
        <mesh geometry={geo}>
          <meshStandardMaterial color={color} flatShading metalness={0.18} roughness={0.6} />
        </mesh>
      </Center>
      {/* ground plane — grounds the part on a surface (reference colors/opacity) */}
      <gridHelper
        args={[gridSize, 24, 0x8ea4c4, 0xc0c8d3]}
        position={[0, gridY, 0]}
        material-transparent
        material-opacity={0.5}
      />
      <SpinRig synced={synced} dist={dist} />
    </>
  );
}

/* ---------- exported viewer ---------- */

/** Decimation stats for the downsampled SAM3D preview (backend-provided, indexed counts). */
export interface MeshDecim {
  kept: number;
  vKept: number;
  pct: number;
}

export interface MeshViewerProps {
  jobId: string;
  path: string;
  color?: string;
  height?: number;
  label?: string;
  /** Optional caption shown in viewer-badge (top-left overlay) */
  badge?: React.ReactNode;
  /** When set, the HUD shows the decimated face/vert counts + reduction %. */
  decim?: MeshDecim | null;
}

export function MeshViewer({
  jobId,
  path,
  color = "#9aa0a6",
  height = 320,
  label,
  badge,
  decim,
}: MeshViewerProps) {
  const { t } = useTranslation();
  const [stats, setStats] = useState<MeshStats | null>(null);
  // Auto-spin on by default; manual interaction detaches this viewer (spin
  // stops); the reset button re-attaches it to the shared spin.
  const [synced, setSynced] = useState(true);

  const q = useQuery({
    queryKey: ["mesh", jobId, path],
    queryFn: () => api.getMeshBytes(jobId, path),
    staleTime: Infinity,
  });

  const handleStats = useCallback((s: MeshStats) => {
    setStats(s);
  }, []);

  // Reset: re-attach to the shared spin. The rig re-frames the object at the
  // calibrated distance and snaps the azimuth back into phase with the others.
  const resetView = useCallback(() => setSynced(true), []);

  /* error state */
  if (q.isError) {
    return (
      <div
        className="viewer-stage"
        style={{ height }}
        role="img"
        aria-label={label ?? "3D mesh preview"}
      >
        <div className="viewer-loading" style={{ color: "var(--bad)" }}>
          {t("x.meshError")}
        </div>
      </div>
    );
  }

  return (
    <div
      className="viewer-stage"
      style={{ height }}
      role="img"
      aria-label={label ?? "3D mesh preview"}
    >
      {/* canvas fills the stage via position:absolute inset:0 */}
      <div style={{ position: "absolute", inset: 0 }}>
        <Canvas
          camera={{ position: [1.68, 1.49, 1.99], fov: FOV }}
          gl={{ antialias: true, alpha: true }}
          style={{ width: "100%", height: "100%" }}
          onCreated={({ gl }) => {
            gl.setClearColor(0x000000, 0);
          }}
        >
          <ambientLight intensity={0.85} color={0xbfefff} />
          <directionalLight position={[2.5, 4, 3]} intensity={1.05} />
          <directionalLight position={[-3, 1, -2]} intensity={0.45} color={0x6fb8ff} />
          <directionalLight position={[0, -2, -3]} intensity={0.5} color={0x2f5fa8} />
          {q.data && (
            <Suspense fallback={null}>
              <Scene buf={q.data} color={color} synced={synced} onStats={handleStats} />
            </Suspense>
          )}
          <OrbitControls
            makeDefault
            enableDamping={false}
            onStart={() => setSynced(false)}
          />
        </Canvas>
      </div>

      {/* loading overlay while mesh bytes fetch */}
      {q.isLoading && (
        <div className="viewer-loading">
          <div className="spinner" />
          {t("x.loading")}
        </div>
      )}

      {/* optional top-left badge (step number + color swatch) */}
      {badge != null && (
        <div className="viewer-badge">{badge}</div>
      )}

      {/* top-right reset control */}
      <div className="viewer-tools">
        <button className="icon-btn" title={t("x.reset")} onClick={resetView}>
          <Icon n="refresh" />
        </button>
      </div>

      {/* bottom HUD — extents + face/vert counts (matches viewer.jsx lines 186–191) */}
      <div className="viewer-hud">
        <div className="ext">
          {t("v.extents")}{" "}{stats ? stats.extStr : "—"}
        </div>
        <div className="hud-fv">
          {!stats ? (
            "—"
          ) : decim ? (
            <>
              <b>{decim.kept.toLocaleString()}</b> {t("v.faces")} ·{" "}
              <b>{decim.vKept.toLocaleString()}</b> {t("v.verts")} ·{" "}
              <span className="ds-cut">−{decim.pct}%</span>
            </>
          ) : (
            `${stats.faces.toLocaleString()} ${t("v.faces")} · ${stats.verts.toLocaleString()} ${t("v.verts")}`
          )}
        </div>
      </div>
    </div>
  );
}
