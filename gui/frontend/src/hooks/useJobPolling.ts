import { useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { JobSummary } from "@/api/types";

export function useJobPolling(jobId: string, onCompleted: () => void) {
  const qc = useQueryClient();
  const calledRef = useRef<string | null>(null);
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
    refetchInterval: (q) =>
      q.state.data && ["completed", "failed", "terminated"].includes(q.state.data.status)
        ? false
        : 1000,
  });
  const logs = useQuery({
    queryKey: ["logs", jobId],
    queryFn: () => api.getLogs(jobId),
    refetchInterval: () => {
      const s = qc.getQueryData<JobSummary>(["job", jobId])?.status;
      return s && s !== "running" && s !== "queued" ? false : 1200;
    },
  });
  useEffect(() => {
    if (job.data?.status === "completed" && calledRef.current !== jobId) {
      calledRef.current = jobId;
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["metrics", jobId] });
      onCompleted();
    }
  }, [job.data?.status, onCompleted, qc, jobId]);
  return { job: job.data, logText: logs.data?.log ?? "" };
}
