/* ============================================================
   MeshViewer — interactive r3f 3-D preview, pixel-accurate port
   of viewer.jsx panel chrome.  Lazy-loaded by ResultView.
   ============================================================ */
import { Suspense, useEffect, useMemo, useCallback, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Bounds, Center } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Icon } from "@/components/Icon";
import { useTranslation } from "react-i18next";
import type { BufferGeometry } from "three";

/* ---------- mesh stats shape ---------- */

interface MeshStats {
  verts: number;
  faces: number;
  extStr: string;
}

/* ---------- inner scene ---------- */

function Geom({
  buf,
  color,
  onStats,
}: {
  buf: ArrayBuffer;
  color: string;
  onStats: (s: MeshStats) => void;
}) {
  const geo = useMemo<BufferGeometry>(() => {
    const g = new STLLoader().parse(buf);
    g.computeVertexNormals();
    return g;
  }, [buf]);

  useEffect(() => {
    /* compute stats once the geometry is ready */
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
    <Bounds fit clip observe margin={1.2}>
      <Center>
        <mesh geometry={geo}>
          <meshStandardMaterial color={color} flatShading metalness={0.18} roughness={0.6} />
        </mesh>
      </Center>
    </Bounds>
  );
}

/* ---------- exported viewer ---------- */

export interface MeshViewerProps {
  jobId: string;
  path: string;
  color?: string;
  height?: number;
  label?: string;
  /** Optional caption shown in viewer-badge (top-left overlay) */
  badge?: React.ReactNode;
}

export function MeshViewer({
  jobId,
  path,
  color = "#9aa0a6",
  height = 320,
  label,
  badge,
}: MeshViewerProps) {
  const { t } = useTranslation();
  const [stats, setStats] = useState<MeshStats | null>(null);

  const q = useQuery({
    queryKey: ["mesh", jobId, path],
    queryFn: () => api.getMeshBytes(jobId, path),
    staleTime: Infinity,
  });

  const handleStats = useCallback((s: MeshStats) => {
    setStats(s);
  }, []);

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
          camera={{ position: [0, 0, 3], fov: 38 }}
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
              <Geom buf={q.data} color={color} onStats={handleStats} />
            </Suspense>
          )}
          <OrbitControls makeDefault enableDamping={false} />
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
        <button
          className="icon-btn"
          title={t("x.reset")}
          onClick={() => {
            /* OrbitControls reset is handled via the controls ref in
               a future enhancement; for now this is a no-op placeholder
               that matches the viewer.jsx chrome */
          }}
        >
          <Icon n="refresh" />
        </button>
      </div>

      {/* bottom HUD — extents + face/vert counts (matches viewer.jsx lines 186–191) */}
      <div className="viewer-hud">
        <div className="ext">
          {t("v.extents")}{" "}{stats ? stats.extStr : "—"}
        </div>
        <div className="hud-fv">
          {stats
            ? `${stats.faces.toLocaleString()} ${t("v.faces")} · ${stats.verts.toLocaleString()} ${t("v.verts")}`
            : "—"}
        </div>
      </div>
    </div>
  );
}
