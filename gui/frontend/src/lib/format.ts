export const fmtIoU = (v?: number) => (v == null || !Number.isFinite(v) ? "—" : (v * 100).toFixed(2) + "%");
export const fmtCd = (v?: number) => {
  if (v == null || !Number.isFinite(v)) return "—";
  const e = v.toExponential(3); const [m, exp] = e.split("e");
  return `${(+m).toFixed(3)}e${parseInt(exp, 10)}`;
};
export const fmtMm = (v?: number) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(2));
export function relTime(nowSec: number, thenSec: number): string {
  const d = Math.max(0, Math.floor(nowSec - thenSec));
  if (d < 60) return `${d}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}
export const fmtElapsed = (sec: number) => `${sec.toFixed(1)}s`;
