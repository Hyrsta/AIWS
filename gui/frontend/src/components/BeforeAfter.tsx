import { lazy, Suspense } from "react";
import { useTranslation } from "react-i18next";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { ResultPaths, CleanupMetadata } from "@/api/types";

const MeshViewer = lazy(() => import("@/components/MeshViewer").then((m) => ({ default: m.MeshViewer })));

interface BeforeAfterProps {
  jobId: string;
  resultPaths: ResultPaths;
  cleanup: CleanupMetadata | null | undefined;
}

export function BeforeAfter({ jobId, resultPaths, cleanup }: BeforeAfterProps) {
  const { t } = useTranslation();

  // Early return BEFORE constructing any viewer
  if (!resultPaths.cleaned_mesh_stl) return null;

  const nBefore =
    cleanup?.n_bodies_before ?? resultPaths.n_bodies_before ?? "?";
  const nAfter =
    cleanup?.n_bodies_after ?? resultPaths.n_bodies_after ?? "?";
  const nRemoved =
    cleanup?.n_bodies_removed ??
    (typeof nBefore === "number" && typeof nAfter === "number"
      ? nBefore - nAfter
      : "?");

  const hasConfidenceFlag = !!cleanup?.confidence_flag;
  const reasons = cleanup?.confidence_reasons ?? [];

  return (
    <div className="space-y-4">
      {hasConfidenceFlag && (
        <Alert variant="destructive">
          <AlertTitle>{t("result.confidence")}</AlertTitle>
          {reasons.length > 0 && (
            <AlertDescription>
              <ul className="mt-1 list-disc pl-4 space-y-0.5 text-sm">
                {reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </AlertDescription>
          )}
        </Alert>
      )}

      <div className="grid grid-cols-2 gap-4">
        {/* Before */}
        <div className="space-y-1">
          <div className="text-sm font-medium text-muted-foreground">
            {t("result.cleanupBefore")} — {nBefore}{" "}
            {typeof nBefore === "number" && nBefore === 1 ? "body" : "bodies"}
          </div>
          {resultPaths.selected_mesh && (
            <Suspense fallback={<div className="animate-pulse rounded-md border bg-card" style={{ height: 280 }} />}>
              <MeshViewer
                jobId={jobId}
                path={resultPaths.selected_mesh}
                color="#9aa0a6"
                height={280}
                label="Before cleanup"
              />
            </Suspense>
          )}
        </div>

        {/* After */}
        <div className="space-y-1">
          <div className="text-sm font-medium text-muted-foreground">
            {t("result.cleanupAfter")} — {nAfter}{" "}
            {typeof nAfter === "number" && nAfter === 1 ? "body" : "bodies"}{" "}
            <span className="text-xs">({nRemoved} removed)</span>
          </div>
          <Suspense fallback={<div className="animate-pulse rounded-md border bg-card" style={{ height: 280 }} />}>
            <MeshViewer
              jobId={jobId}
              path={resultPaths.cleaned_mesh_stl}
              color="#43a047"
              height={280}
              label="After cleanup"
            />
          </Suspense>
        </div>
      </div>
    </div>
  );
}
