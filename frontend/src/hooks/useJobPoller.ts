import { useEffect, useRef, useState, useCallback } from "react";
import { fetchJobStatus } from "../api/client";
import type { JobStatus } from "../types";

export function useJobPoller(jobId: string | null, intervalMs = 1500) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setError(null);
      return;
    }

    let active = true;

    const poll = async () => {
      try {
        const status = await fetchJobStatus(jobId);
        if (!active) return;
        setJob(status);
        if (status.status === "completed" || status.status === "failed") {
          stop();
          if (status.status === "failed") {
            setError(status.error ?? "Analysis failed");
          }
        }
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Polling failed");
        stop();
      }
    };

    poll();
    timerRef.current = setInterval(poll, intervalMs);

    return () => {
      active = false;
      stop();
    };
  }, [jobId, intervalMs, stop]);

  return { job, error };
}
