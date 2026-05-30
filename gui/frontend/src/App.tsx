/* ============================================================
   App — root state machine. Ported from AIWS Design Reference aiws/app.jsx.
   Data layer: real FastAPI backend via src/api/client.ts.
   ============================================================ */
import { useState, useCallback, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { ReconstructInput } from "@/api/types";
import i18n from "@/i18n";
import { Rail } from "@/components/Rail";
import { TopBar } from "@/components/TopBar";
import { ConfigureView } from "@/views/ConfigureView";
import { LiveView } from "@/views/LiveView";
import { ResultView } from "@/views/ResultView";
import { Icon } from "@/components/Icon";
import type { JobSummary } from "@/api/types";

type View = "configure" | "live" | "result";

// ---- Toast ----
function Toast({ msg }: { msg: string }) {
  if (!msg) return null;
  return (
    <div
      style={{
        position: "fixed",
        bottom: 22,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 100,
        background: "#262217",
        border: "1px solid #3c352a",
        color: "#f3efe6",
        padding: "10px 18px",
        borderRadius: 10,
        fontSize: 14,
        boxShadow: "var(--shadow-pop)",
        display: "flex",
        alignItems: "center",
        gap: 9,
      }}
    >
      <Icon n="info" size={15} style={{ color: "var(--sig)" }} />
      {msg}
    </div>
  );
}

// ---- BackendGate restyled with prototype CSS ----
function BackendGate({ children }: { children: React.ReactNode }) {
  const q = useQuery({ queryKey: ["health"], queryFn: api.health });
  if (q.isLoading) {
    return (
      <div className="fullscreen-msg">
        <div className="card">
          <div className="section-sub">Connecting to backend…</div>
        </div>
      </div>
    );
  }
  if (q.isError) {
    return (
      <div className="fullscreen-msg">
        <div className="card">
          <div className="section-title">
            <span className="ix" style={{ color: "var(--bad)" }}>
              <Icon n="alert" size={17} />
            </span>
            <h2>Backend unavailable</h2>
          </div>
          <p className="section-sub">
            The AIWS backend could not be reached. Start the server and refresh.
          </p>
        </div>
      </div>
    );
  }
  return <>{children}</>;
}

// ---- App shell ----
export default function App() {
  const qc = useQueryClient();
  const [lang, setLangState] = useState<string>(
    () => localStorage.getItem("aiws.lang") ?? "en",
  );
  const [view, setView] = useState<View>("configure");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [toast, setToast] = useState("");

  // Keep i18n in sync with lang state
  useEffect(() => {
    void i18n.changeLanguage(lang);
    localStorage.setItem("aiws.lang", lang);
  }, [lang]);

  const setLang = useCallback((l: string) => {
    setLangState(l);
  }, []);

  // Jobs list — polled every 5s
  const jobsQ = useQuery({
    queryKey: ["jobs"],
    queryFn: api.listJobs,
    refetchInterval: 5000,
  });
  const jobs: JobSummary[] = jobsQ.data ?? [];

  // Active job data (for TopBar)
  const activeJobQ = useQuery({
    queryKey: ["job", activeId],
    queryFn: () => api.getJob(activeId!),
    enabled: !!activeId,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s && ["completed", "failed", "terminated"].includes(s) ? false : 2000;
    },
  });
  const activeJob = activeJobQ.data ?? null;

  // Toast helpers
  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(""), 2400);
  }, []);

  // Handlers
  const onStart = useCallback(
    async (input: ReconstructInput) => {
      try {
        const job = await api.createSimpleReconstruct(input);
        setActiveId(job.job_id);
        setView("live");
        void qc.invalidateQueries({ queryKey: ["jobs"] });
      } catch (err) {
        showToast(err instanceof Error ? err.message : String(err));
      }
    },
    [qc, showToast],
  );

  const onSelect = useCallback((id: string) => {
    setActiveId(id);
    // Determine view from cached job data if available
    const cached = qc.getQueryData<JobSummary>(["job", id]);
    if (cached?.status === "completed") {
      setView("result");
    } else {
      setView("live");
    }
  }, [qc]);

  const onNew = useCallback(() => {
    setActiveId(null);
    setView("configure");
  }, []);

  const onCompleted = useCallback(() => {
    setView("result");
    void qc.invalidateQueries({ queryKey: ["jobs"] });
  }, [qc]);

  return (
    <BackendGate>
      <div className="bg-field" />
      <div className="bg-grid" />
      <div className="app">
        <Rail
          jobs={jobs}
          activeId={activeId}
          onNew={onNew}
          onSelect={onSelect}
        />
        <TopBar view={view} job={activeJob} lang={lang} setLang={setLang} />
        <main className="main" key={view + (activeId ?? "")}>
          {view === "configure" && <ConfigureView onStart={onStart} />}
          {view === "live" && activeId && (
            <LiveView jobId={activeId} onCompleted={onCompleted} />
          )}
          {view === "result" && activeId && (
            <ResultView jobId={activeId} onNew={onNew} />
          )}
        </main>
      </div>
      <Toast msg={toast} />
    </BackendGate>
  );
}
