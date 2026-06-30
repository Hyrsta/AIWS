import { useEffect, useRef, useState, useCallback } from "react";
import { api } from "../api/client";
import type { RefinePoint } from "../api/types";

export function canvasToImage(cx: number, cy: number, scale: number) {
  return { x: Math.round(cx / scale), y: Math.round(cy / scale) };
}

export function pngDataUrlToFile(dataUrl: string, name: string): File {
  const b64 = dataUrl.split(",")[1] ?? dataUrl;
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new File([arr], name, { type: "image/png" });
}

type Props = { image: File; defaultPrompt: string; onMaskChange: (m: File | null) => void };

export function SegmentRefineCanvas({ image, defaultPrompt, onMaskChange }: Props) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [maskB64, setMaskB64] = useState<string | null>(null);
  const [points, setPoints] = useState<RefinePoint[]>([]);
  const [busy, setBusy] = useState(false);
  const [hint, setHint] = useState<string>("");
  const imgUrlRef = useRef<string>("");
  const imgSizeRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const wrapRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x0: number; y0: number } | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  // Object URL for the uploaded image preview.
  useEffect(() => {
    const url = URL.createObjectURL(image);
    imgUrlRef.current = url;
    const im = new Image();
    im.onload = () => { imgSizeRef.current = { w: im.width, h: im.height }; };
    im.src = url;
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const publishMask = useCallback((b64: string | null) => {
    setMaskB64(b64);
    onMaskChange(b64 ? pngDataUrlToFile(b64, "refined_mask.png") : null);
  }, [onMaskChange]);

  // Create the session + initial auto-mask on mount / image change.
  useEffect(() => {
    let alive = true;
    setBusy(true); setHint("");
    api.segmentSession(image, prompt).then((s) => {
      if (!alive) return;
      setSessionId(s.session_id);
      sessionIdRef.current = s.session_id;
      setPoints([]);
      publishMask(s.mask_png_base64);
      if (!s.detected) setHint("No object detected. Try a different prompt or click a point.");
    }).catch(() => { if (alive) setHint("Segmentation service unavailable."); })
      .finally(() => { if (alive) setBusy(false); });
    return () => {
      alive = false;
      if (sessionIdRef.current) api.segmentRelease(sessionIdRef.current).catch(() => {}); sessionIdRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image]);

  async function callRefine(edit: { prompt?: string; points?: RefinePoint[]; box?: number[]; reset?: boolean }) {
    if (!sessionId) return;
    setBusy(true);
    try {
      const r = await api.segmentRefine(sessionId, edit);
      publishMask(r.mask_png_base64);
      setHint("");
    } catch (e) {
      const status = (e as Error & { status?: number }).status;
      if (status === 409) {
        // session expired: re-create from the image we still hold, then retry once
        try {
          const s = await api.segmentSession(image, prompt);
          setSessionId(s.session_id);
          sessionIdRef.current = s.session_id;
          const r = await api.segmentRefine(s.session_id, edit);
          publishMask(r.mask_png_base64);
        } catch {
          setHint("Refine failed. Try again.");
        }
      } else if (status === 422) {
        setHint("No match for that prompt. Keeping the current mask.");
      } else {
        setHint("Refine failed. Try again.");
      }
    } finally { setBusy(false); }
  }

  const scale = () => {
    const w = wrapRef.current?.clientWidth ?? imgSizeRef.current.w;
    return imgSizeRef.current.w ? w / imgSizeRef.current.w : 1;
  };

  function onClickPoint(ev: React.MouseEvent) {
    if (busy) return;
    const rect = wrapRef.current!.getBoundingClientRect();
    const { x, y } = canvasToImage(ev.clientX - rect.left, ev.clientY - rect.top, scale());
    const label: 1 | 0 = ev.altKey ? 0 : 1;
    const next = [...points, { x, y, label }];
    setPoints(next);
    callRefine({ points: [{ x, y, label }] });
  }

  function onMouseDown(ev: React.MouseEvent) {
    const rect = wrapRef.current!.getBoundingClientRect();
    dragRef.current = { x0: ev.clientX - rect.left, y0: ev.clientY - rect.top };
  }
  function onMouseUp(ev: React.MouseEvent) {
    const d = dragRef.current; dragRef.current = null;
    if (!d) return;
    const rect = wrapRef.current!.getBoundingClientRect();
    const x1 = ev.clientX - rect.left, y1 = ev.clientY - rect.top;
    if (Math.abs(x1 - d.x0) < 6 && Math.abs(y1 - d.y0) < 6) { onClickPoint(ev); return; }
    const s = scale();
    const a = canvasToImage(Math.min(d.x0, x1), Math.min(d.y0, y1), s);
    const b = canvasToImage(Math.max(d.x0, x1), Math.max(d.y0, y1), s);
    callRefine({ box: [a.x, a.y, b.x, b.y] });
  }

  return (
    <div className="seg-refine">
      <div className="seg-canvas-wrap" ref={wrapRef}
           onMouseDown={onMouseDown} onMouseUp={onMouseUp}
           onContextMenu={(e) => e.preventDefault()}>
        {imgUrlRef.current && <img className="seg-base" src={imgUrlRef.current} alt="input" />}
        {maskB64 && <img className="seg-mask" src={`data:image/png;base64,${maskB64}`} alt="mask" />}
        {busy && <div className="seg-busy">...</div>}
      </div>
      <div className="seg-controls">
        <input className="seg-prompt" value={prompt}
               onChange={(e) => setPrompt(e.target.value)}
               placeholder="workpiece. metal part." />
        <button type="button" disabled={busy} onClick={() => callRefine({ prompt })}>Re-detect</button>
        <button type="button" disabled={busy || !points.length}
                onClick={() => { const n = points.slice(0, -1); setPoints(n);
                  callRefine({ reset: true }).then(() => { if (n.length) callRefine({ points: n }); }); }}>Undo point</button>
        <button type="button" disabled={busy}
                onClick={() => { setPoints([]); callRefine({ reset: true }).then(() => callRefine({ prompt })); }}>Reset to auto</button>
      </div>
      {hint && <div className="seg-hint">{hint}</div>}
    </div>
  );
}
