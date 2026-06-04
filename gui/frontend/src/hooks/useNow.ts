import { useEffect, useState } from "react";

/**
 * Ticking wall-clock time in seconds.
 *
 * While `active` is true the value updates every `intervalMs` (default 1s);
 * when inactive it holds its last value. The clock is only read inside an
 * effect/interval (never during render), which keeps consuming components pure
 * (no `Date.now()` in render) and is the supported way to drive live "elapsed"
 * displays. Callers gate `active` on the running state, so the held value is
 * only consumed for in-flight jobs.
 */
export function useNow(active: boolean, intervalMs = 1000): number {
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNowMs(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [active, intervalMs]);
  return nowMs / 1000;
}
