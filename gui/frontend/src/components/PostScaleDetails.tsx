import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fmtMm, fmtCd } from "@/lib/format";
import type { Metrics, ScaledMetadata } from "@/api/types";

interface PostScaleDetailsProps {
  metrics: Metrics;
  scaledMeta?: ScaledMetadata | null;
}

function fmtVec3(v: [number, number, number] | undefined): string {
  if (!v) return "—";
  return v.map((x) => fmtMm(x)).join(" × ");
}

export function PostScaleDetails({ metrics, scaledMeta }: PostScaleDetailsProps) {
  const { t } = useTranslation();

  if (!metrics.postscale.available) return null;

  const ps = metrics.postscale;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">{t("result.postScaleTitle")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {/* Summary from metrics */}
        <div className="grid grid-cols-2 gap-x-6 gap-y-1">
          <div className="text-muted-foreground">{t("result.rewriteMode")}</div>
          <div className="font-mono">{ps.rewrite_mode ?? "—"}</div>

          <div className="text-muted-foreground">{t("result.match")}</div>
          <div
            className={
              ps.match_ok === true
                ? "text-green-600 font-medium"
                : ps.match_ok === false
                  ? "text-destructive font-medium"
                  : ""
            }
          >
            {ps.match_ok === true ? "✓ OK" : ps.match_ok === false ? "✗ Failed" : "—"}
          </div>

          <div className="text-muted-foreground">{t("result.maxRelError")}</div>
          <div className="font-mono">
            {ps.max_rel_error != null ? fmtCd(ps.max_rel_error) : "—"}
          </div>

          <div className="text-muted-foreground">{t("result.canonicalExtents")}</div>
          <div className="font-mono">{fmtVec3(ps.canonical_extents)}</div>

          <div className="text-muted-foreground">{t("result.catalogTarget")}</div>
          <div className="font-mono">{fmtVec3(ps.catalog_target_mm)}</div>

          <div className="text-muted-foreground">{t("result.afterScale")}</div>
          <div className="font-mono">{fmtVec3(ps.after_scale_mm)}</div>
        </div>

        {/* Matrix and axis pairing from scaled metadata file */}
        {scaledMeta && (
          <>
            {/* 3×3 transform matrix */}
            {scaledMeta.scale.matrix_3x3 && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
                  {t("result.scaleMatrix")}
                </div>
                <div
                  className="grid font-mono text-xs gap-x-4 gap-y-0.5"
                  style={{ gridTemplateColumns: "repeat(3, max-content)" }}
                >
                  {scaledMeta.scale.matrix_3x3.flatMap((row, ri) =>
                    row.map((v, ci) => (
                      <span
                        key={`${ri}-${ci}`}
                        className={ri === ci ? "text-primary font-semibold" : "text-muted-foreground"}
                      >
                        {v.toFixed(4)}
                      </span>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Per-axis pairing/scales */}
            {scaledMeta.scale.scale_on_catalog_axes && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
                  {t("result.scaleOnAxes")}
                </div>
                <table className="text-xs w-full border-collapse">
                  <thead>
                    <tr className="text-muted-foreground border-b">
                      <th className="text-left py-1 font-medium w-16">{t("result.axis")}</th>
                      <th className="text-right py-1 font-medium font-mono">{t("result.scale")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(["X", "Y", "Z"] as const).map((ax) => {
                      const val = scaledMeta.scale.scale_on_catalog_axes?.[ax];
                      return (
                        <tr key={ax} className="border-b border-border/50">
                          <td className="py-1 font-medium">{ax}</td>
                          <td className="py-1 text-right font-mono">
                            {val != null ? val.toFixed(6) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
