import { Download } from "lucide-react";
import { api } from "@/api/client";

export interface DownloadItem {
  label: string;
  path: string | undefined;
}

interface DownloadGroupProps {
  jobId: string;
  title: string;
  items: DownloadItem[];
}

function fileExt(path: string): string {
  const m = path.match(/\.(\w+)$/);
  return m ? m[1].toUpperCase().slice(0, 4) : "FILE";
}

function extColorClass(ext: string): string {
  const e = ext.toLowerCase();
  if (e === "stl") return "bg-blue-100 text-blue-700 border-blue-200";
  if (e === "step" || e === "stp") return "bg-green-100 text-green-700 border-green-200";
  if (e === "py") return "bg-yellow-100 text-yellow-700 border-yellow-200";
  if (e === "json") return "bg-purple-100 text-purple-700 border-purple-200";
  return "bg-muted text-muted-foreground border-border";
}

export function DownloadGroup({ jobId, title, items }: DownloadGroupProps) {
  const visible = items.filter((item) => !!item.path);
  if (visible.length === 0) return null;

  return (
    <div>
      <h4 className="text-sm font-semibold mb-2 text-foreground">{title}</h4>
      <div className="space-y-1">
        {visible.map((item) => {
          const path = item.path!;
          const ext = fileExt(path);
          const filename = path.split("/").pop() ?? path;
          const href = api.fileUrl(jobId, path);
          return (
            <a
              key={path}
              href={href}
              download={filename}
              className="flex items-center gap-3 rounded-md border px-3 py-2 text-sm hover:bg-muted transition-colors no-underline text-foreground"
            >
              <span
                className={`inline-flex items-center justify-center rounded border px-1.5 py-0.5 text-[10px] font-bold font-mono min-w-[36px] ${extColorClass(ext)}`}
              >
                {ext}
              </span>
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{item.label}</div>
                <div className="text-xs text-muted-foreground truncate font-mono">{filename}</div>
              </div>
              <Download className="size-4 shrink-0 text-muted-foreground" />
            </a>
          );
        })}
      </div>
    </div>
  );
}
