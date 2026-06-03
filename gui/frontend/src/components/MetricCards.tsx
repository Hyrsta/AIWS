import { useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fmtIoU, fmtCd, fmtMm } from "@/lib/format";
import type { Metrics } from "@/api/types";

export function MetricCards({ metrics }: { metrics: Metrics }) {
  const { t } = useTranslation();
  const cards: React.ReactNode[] = [];

  if (metrics.cadrille.available) {
    cards.push(
      <Card key="iou">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            {t("result.iou")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold tabular-nums">
            {fmtIoU(metrics.cadrille.mean_iou)}
          </div>
          <p className="text-xs text-muted-foreground mt-1">{t("metric.meanIou")}</p>
        </CardContent>
      </Card>
    );

    cards.push(
      <Card key="cd">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            {t("result.cd")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold tabular-nums font-mono">
            {fmtCd(metrics.cadrille.median_cd)}
          </div>
          <p className="text-xs text-muted-foreground mt-1">{t("metric.medianChamfer")}</p>
        </CardContent>
      </Card>
    );
  }

  if (metrics.postscale.available) {
    const { catalog_target_mm, after_scale_mm, match_ok } = metrics.postscale;
    const fmtVec = (v?: [number, number, number]) =>
      v ? v.map((x) => fmtMm(x)).join(" × ") : "—";

    cards.push(
      <Card key="bbox">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            {t("result.bbox")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm font-bold tabular-nums">
            <span
              className={
                match_ok === true
                  ? "text-green-600"
                  : match_ok === false
                    ? "text-destructive"
                    : ""
              }
            >
              {match_ok === true ? "✓" : match_ok === false ? "✗" : "—"}
            </span>
          </div>
          <div className="mt-1 text-xs text-muted-foreground space-y-0.5">
            <div>
              <span className="font-medium">{t("metric.target")}</span> {fmtVec(catalog_target_mm)}
            </div>
            <div>
              <span className="font-medium">{t("metric.after")}</span> {fmtVec(after_scale_mm)}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (cards.length === 0) return null;

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {cards}
    </div>
  );
}
