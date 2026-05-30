import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

interface LogConsoleProps {
  text: string;
}

export function LogConsole({ text }: LogConsoleProps) {
  const { t } = useTranslation();
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [text]);

  return (
    <div className="rounded-md border bg-card overflow-hidden">
      <div className="flex items-center gap-2 border-b bg-muted/40 px-3 py-1.5">
        <div className="flex gap-1.5" aria-hidden="true">
          <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
          <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
          <span className="h-3 w-3 rounded-full bg-[#28c840]" />
        </div>
        <span className="text-xs text-muted-foreground">{t("live.log")}</span>
      </div>
      <pre
        ref={ref}
        className="h-64 overflow-y-auto p-3 text-xs font-mono leading-relaxed text-foreground whitespace-pre-wrap break-all"
      >
        {text || ""}
      </pre>
    </div>
  );
}
