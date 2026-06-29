/* ============================================================
   LogConsole — shared terminal-style log panel.
   Used by both the Live (during generation) and Result views so the
   in-generation and finished views are identical. The only difference is
   the header title: "Live log" while running, "Reconstruction log" once
   finished. No white wrapper panel — the terminal is shown directly.
   ============================================================ */
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

interface LogConsoleProps {
  lines: string[];
  /** True while the job is still running → header reads "Live log". */
  live?: boolean;
}

export function LogConsole({ lines, live = false }: LogConsoleProps) {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever the line count changes.
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines.length]);

  function renderLine(line: string, i: number) {
    // parse "[mm:ss.s] [level] message"
    const m = line.match(/^(\[[\d:.]+\])\s+(?:\[(\w+)\]\s+)?(.*)$/);
    if (!m) {
      return <div className="ln" key={i}>{line}</div>;
    }
    const [, ts, lv, msg] = m;
    const lvClass =
      lv === "ok" ? "lv-ok"
      : lv === "warn" ? "lv-warn"
      : lv === "error" || lv === "fail" ? "lv-error"
      : lv === "info" ? "lv-info"
      : lv === "stage" ? "lv-stage"
      : "";
    return (
      <div className="ln" key={i}>
        <span className="ts">{ts + " "}</span>
        {lv ? <span className={lvClass}>{"[" + lv + "] "}</span> : "      "}
        <span className={lv === "stage" ? "lv-stage" : ""}>{msg}</span>
      </div>
    );
  }

  return (
    <div className="log-wrap">
      <div className="log-head">
        <div className="row gap-8">
          <span className="dots">
            <i style={{ background: "#ff5f57" }} />
            <i style={{ background: "#febc2e" }} />
            <i style={{ background: "#28c840" }} />
          </span>
          <span style={{ marginLeft: 6 }}>{live ? t("log.title") : t("res.log")}</span>
        </div>
        <div className="row gap-8">
          <span className="mono" style={{ fontSize: 12 }}>
            {t("log.lines", { n: lines.length })}
          </span>
        </div>
      </div>
      <div className="log" ref={ref}>
        {lines.length === 0
          ? <div className="ln" style={{ opacity: 0.5 }}>—</div>
          : lines.map(renderLine)}
      </div>
    </div>
  );
}
