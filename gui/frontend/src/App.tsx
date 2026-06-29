/* ============================================================
   App — root state machine. Ported from AIWS Design Reference aiws/app.jsx.
   Data layer: real FastAPI backend via src/api/client.ts.
   ============================================================ */
import { useState, useCallback, useEffect, useMemo } from "react";
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

interface RouteState {
  view: View;
  activeId: string | null;
}

function readRouteState(): RouteState {
  if (typeof window === "undefined") return { view: "configure", activeId: null };
  const raw = window.location.hash.replace(/^#/, "");
  const params = new URLSearchParams(raw);
  const view = params.get("view");
  const activeId = params.get("job");
  if ((view === "live" || view === "result") && activeId) {
    return { view, activeId };
  }
  return { view: "configure", activeId: null };
}

function writeRouteState(view: View, activeId: string | null) {
  if (typeof window === "undefined") return;
  const params = new URLSearchParams();
  if (activeId && view !== "configure") {
    params.set("view", view);
    params.set("job", activeId);
  }
  const nextHash = params.toString() ? `#${params.toString()}` : "";
  if (window.location.hash === nextHash) return;
  window.history.replaceState(
    null,
    "",
    `${window.location.pathname}${window.location.search}${nextHash}`,
  );
}

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

// ---- ConfirmDialog (delete confirmation modal) ----
function ConfirmDialog({
  open, title, body, meta, confirmLabel, cancelLabel, onConfirm, onCancel,
}: {
  open: boolean;
  title: string;
  body: string;
  meta?: string;
  confirmLabel: string;
  cancelLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      else if (e.key === "Enter") onConfirm();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onConfirm, onCancel]);
  if (!open) return null;
  return (
    <div className="modal-scrim" onMouseDown={onCancel}>
      <div
        className="modal-card"
        role="alertdialog"
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h3 className="modal-title">{title}</h3>
        <p className="modal-body">{body}</p>
        {meta ? <div className="modal-meta mono">{meta}</div> : null}
        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onCancel}>{cancelLabel}</button>
          <button className="btn btn-danger" onClick={onConfirm} autoFocus>
            <Icon n="trash" size={15} />
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---- App shell ----
export default function App() {
  const qc = useQueryClient();
  const initialRoute = readRouteState();
  const [lang, setLangState] = useState<string>(
    () => localStorage.getItem("aiws.lang") ?? "en",
  );
  const [view, setView] = useState<View>(initialRoute.view);
  const [activeId, setActiveId] = useState<string | null>(initialRoute.activeId);
  const [toast, setToast] = useState("");
  const [pendingDel, setPendingDel] = useState<string | null>(null);

  // Keep i18n in sync with lang state
  useEffect(() => {
    void i18n.changeLanguage(lang);
    localStorage.setItem("aiws.lang", lang);
  }, [lang]);

  const setLang = useCallback((l: string) => {
    setLangState(l);
  }, []);

  // Keep the selected page/job in the URL so a browser reload restores it.
  useEffect(() => {
    writeRouteState(view, activeId);
  }, [view, activeId]);

  useEffect(() => {
    const onHashChange = () => {
      const next = readRouteState();
      setView(next.view);
      setActiveId(next.activeId);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Jobs list — polled every 5s
  const jobsQ = useQuery({
    queryKey: ["jobs"],
    queryFn: api.listJobs,
    refetchInterval: 5000,
  });
  // Memoized so its identity is stable for the useCallback deps below.
  const jobs: JobSummary[] = useMemo(() => jobsQ.data ?? [], [jobsQ.data]);

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
    const selected = jobs.find((j) => j.job_id === id) ?? qc.getQueryData<JobSummary>(["job", id]);
    if (selected?.status === "completed") {
      setView("result");
    } else {
      setView("live");
    }
  }, [jobs, qc]);

  const onNew = useCallback(() => {
    setActiveId(null);
    setView("configure");
  }, []);

  const onCompleted = useCallback(() => {
    setView("result");
    void qc.invalidateQueries({ queryKey: ["jobs"] });
  }, [qc]);

  // Delete flow — request opens a confirmation; confirm removes the job.
  const onRequestDelete = useCallback((id: string) => setPendingDel(id), []);
  const onCancelDelete = useCallback(() => setPendingDel(null), []);
  const onConfirmDelete = useCallback(async () => {
    const id = pendingDel;
    if (!id) return;
    setPendingDel(null);
    try {
      await api.deleteJob(id);
      // If the deleted job is on screen, fall back to the configure view.
      if (activeId === id) {
        setActiveId(null);
        setView("configure");
      }
      void qc.invalidateQueries({ queryKey: ["jobs"] });
      showToast(i18n.t("del.toast"));
    } catch (err) {
      showToast(err instanceof Error ? err.message : i18n.t("del.failed"));
    }
  }, [pendingDel, activeId, qc, showToast]);

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
          onDelete={onRequestDelete}
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
      <ConfirmDialog
        open={pendingDel !== null}
        title={i18n.t("del.title")}
        body={i18n.t("del.body")}
        meta={pendingDel ?? undefined}
        confirmLabel={i18n.t("del.confirm")}
        cancelLabel={i18n.t("del.cancel")}
        onConfirm={() => { void onConfirmDelete(); }}
        onCancel={onCancelDelete}
      />
    </BackendGate>
  );
}
