/* ============================================================
   PointCloudViewer — interactive r3f preview for the exact
   Cadrille point-cloud batch saved by the backend.
   ============================================================ */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import { useTranslation } from "react-i18next";
import { Icon } from "@/components/Icon";
import type { Vec3 } from "@/api/types";
import type { PerspectiveCamera } from "three";

const FOV = 38;
const POLAR = 1.02;
const BASE_AZIMUTH = 0.7;
const ROT_SPEED = 0.16;
const FRAME_MARGIN = 1.44;
const GRID_SIZE_MULT = 3.2;
const ROT_T0 = typeof performance !== "undefined" ? performance.now() : 0;

function sharedAzimuth(): number {
  return BASE_AZIMUTH + ((performance.now() - ROT_T0) / 1000) * ROT_SPEED;
}

interface PointStats {
  count: number;
  extStr: string;
}

function SpinRig({ synced, dist }: { synced: boolean; dist: number }) {
  const camera = useThree((s) => s.camera) as PerspectiveCamera;
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null;

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
    controls.target.set(0, 0, 0);
    camera.position.set(
      dist * sinP * Math.sin(az),
      dist * Math.cos(POLAR),
      dist * sinP * Math.cos(az),
    );
    camera.lookAt(0, 0, 0);
    controls.update();
  });

  return null;
}

function PointScene({
  points,
  color,
  synced,
  onStats,
}: {
  points: Vec3[];
  color: string;
  synced: boolean;
  onStats: (s: PointStats) => void;
}) {
  const geometry = useMemo(() => {
    const data = new Float32Array(points.length * 3);
    points.forEach(([x, y, z], i) => {
      const j = i * 3;
      data[j] = x;
      data[j + 1] = y;
      data[j + 2] = z;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data, 3));
    g.computeBoundingBox();
    g.computeBoundingSphere();
    return g;
  }, [points]);

  const frame = useMemo(() => {
    const bb = geometry.boundingBox;
    if (!bb) {
      return {
        center: new THREE.Vector3(),
        dist: 3,
        gridSize: 2,
        gridY: -0.5,
        pointSize: 0.035,
      };
    }
    const center = new THREE.Vector3();
    bb.getCenter(center);
    const w = bb.max.x - bb.min.x;
    const h = bb.max.y - bb.min.y;
    const d = bb.max.z - bb.min.z;
    const radius = 0.5 * Math.sqrt(w * w + h * h + d * d) || 1;
    const maxDim = Math.max(w, h, d) || 1;
    return {
      center,
      dist: (radius / Math.sin(((FOV * Math.PI) / 180) / 2)) * FRAME_MARGIN,
      gridSize: maxDim * GRID_SIZE_MULT,
      gridY: bb.min.y - center.y - maxDim * 0.03,
      pointSize: Math.max(maxDim * 0.018, 0.015),
    };
  }, [geometry]);

  useEffect(() => {
    const bb = geometry.boundingBox;
    let extStr = "—";
    if (bb) {
      const sx = (bb.max.x - bb.min.x).toFixed(2);
      const sy = (bb.max.y - bb.min.y).toFixed(2);
      const sz = (bb.max.z - bb.min.z).toFixed(2);
      extStr = `${sx} × ${sy} × ${sz}`;
    }
    onStats({ count: points.length, extStr });
  }, [geometry, onStats, points.length]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <>
      <points geometry={geometry} position={[-frame.center.x, -frame.center.y, -frame.center.z]}>
        <pointsMaterial color={color} size={frame.pointSize} sizeAttenuation />
      </points>
      <gridHelper
        args={[frame.gridSize, 16, 0x8ea4c4, 0xc0c8d3]}
        position={[0, frame.gridY, 0]}
        material-transparent
        material-opacity={0.45}
      />
      <SpinRig synced={synced} dist={frame.dist} />
    </>
  );
}

export interface PointCloudViewerProps {
  points: Vec3[];
  color?: string;
  height?: number;
  label?: string;
}

export function PointCloudViewer({
  points,
  color = "#5f80a8",
  height = 360,
  label,
}: PointCloudViewerProps) {
  const { t } = useTranslation();
  const [stats, setStats] = useState<PointStats | null>(null);
  const [synced, setSynced] = useState(true);
  const resetView = useCallback(() => setSynced(true), []);

  return (
    <div
      className="viewer-stage point-cloud-stage"
      style={{ height }}
      role="img"
      aria-label={label ?? "Cadrille point-cloud preview"}
    >
      <div style={{ position: "absolute", inset: 0 }}>
        <Canvas
          camera={{ position: [1.68, 1.49, 1.99], fov: FOV }}
          gl={{ antialias: true, alpha: true }}
          style={{ width: "100%", height: "100%" }}
          onCreated={({ gl }) => {
            gl.setClearColor(0x000000, 0);
          }}
        >
          <ambientLight intensity={0.9} color={0xbfefff} />
          <directionalLight position={[2.5, 4, 3]} intensity={0.9} />
          <directionalLight position={[-3, 1, -2]} intensity={0.35} color={0x6fb8ff} />
          <PointScene points={points} color={color} synced={synced} onStats={setStats} />
          <OrbitControls makeDefault enableDamping={false} onStart={() => setSynced(false)} />
        </Canvas>
      </div>

      <div className="viewer-tools">
        <button className="icon-btn" title={t("x.reset")} onClick={resetView}>
          <Icon n="refresh" />
        </button>
      </div>

      <div className="viewer-hud">
        <div className="ext">
          {t("v.extents")} {stats ? stats.extStr : "—"}
        </div>
        <div className="hud-fv">
          {stats ? t("res.cadrillePointCount", { n: stats.count.toLocaleString() }) : "—"}
        </div>
      </div>
    </div>
  );
}
