/* ============================================================
   InputImagesPanel — "Uploaded inputs" panel.
   Shows the photo + mask submitted with a job. Used in both the
   Live (during generation) and Result views.
   ============================================================ */
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { api } from "@/api/client";

interface InputImagesPanelProps {
  jobId: string;
  /** When omitted, the panel fetches /jobs/{id}/inputs itself. */
  inputImage?: string | null;
  inputMask?: string | null;
}

export function InputImagesPanel({ jobId, inputImage, inputMask }: InputImagesPanelProps) {
  const { t } = useTranslation();

  // If the caller didn't pass paths, resolve them ourselves. Polled lightly so
  // it appears as soon as the backend has written the uploaded inputs.
  const needFetch = inputImage === undefined && inputMask === undefined;
  const inputsQ = useQuery({
    queryKey: ["inputs", jobId],
    queryFn: () => api.getJobInputs(jobId),
    enabled: needFetch,
    refetchInterval: (q) =>
      q.state.data && (q.state.data.input_image || q.state.data.input_mask) ? false : 4000,
  });

  const img = needFetch ? inputsQ.data?.input_image ?? null : inputImage ?? null;
  const msk = needFetch ? inputsQ.data?.input_mask ?? null : inputMask ?? null;

  if (!img && !msk) return null;

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
          {img ? (
            <div className="thumb">
              <div className="thumb-img">
                <img src={api.fileUrl(jobId, img)} alt={t("upl.photo")} />
              </div>
              <div className="thumb-cap">
                <b>{t("upl.photo")}</b>
                <span className="dim thumb-name" title={img}>
                  {img.split("/").pop()}
                </span>
              </div>
            </div>
          ) : null}
          {msk ? (
            <div className="thumb mask">
              <div className="thumb-img">
                <img src={api.fileUrl(jobId, msk)} alt={t("upl.mask")} />
              </div>
              <div className="thumb-cap">
                <b>{t("upl.mask")}</b>
                <span className="dim thumb-name" title={msk}>
                  {msk.split("/").pop()}
                </span>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
