// Upload a video, then poll its analysis job until it finishes (POST /api/jobs, GET /api/jobs/{id}).
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorText, isAbort, uploadVideo, type UploadHandle } from "./api";
import { stageLabel } from "./i18n";
import type { Job } from "./types";

export type JobState =
  | { phase: "idle" }
  | { phase: "uploading"; file: string; fraction: number }
  | { phase: "processing"; file: string; jobId: string; job: Job | null; transientError: string | null }
  | { phase: "done"; file: string; job: Job }
  | { phase: "error"; file: string; message: string };

const MAX_POLL_FAILURES = 5;

/** Job.progress may be a 0..1 fraction or a 0..100 percentage; normalise to 0..1. */
export function progressFraction(p: number): number {
  if (!Number.isFinite(p)) return 0;
  const f = p > 1 ? p / 100 : p;
  return Math.max(0, Math.min(1, f));
}

/** Overall fraction and a Turkish status line for the toolbar. */
export function describeJob(s: JobState): { fraction: number | null; text: string; tone: "busy" | "ok" | "error" } {
  switch (s.phase) {
    case "idle":
      return { fraction: null, text: "", tone: "busy" };
    case "uploading":
      return { fraction: s.fraction, text: `Yükleniyor · %${Math.round(s.fraction * 100)}`, tone: "busy" };
    case "processing": {
      const job = s.job;
      const f = job ? progressFraction(job.progress) : 0;
      let text = !job || job.status === "queued" ? "Sırada" : stageLabel(job.stage);
      if (job && job.status === "running") text += ` · %${Math.round(f * 100)}`;
      if (s.transientError) text += " · yeniden deneniyor…";
      return { fraction: f, text, tone: "busy" };
    }
    case "done":
      return { fraction: 1, text: "Tamamlandı, analiz açıldı", tone: "ok" };
    case "error":
      return { fraction: null, text: s.message, tone: "error" };
  }
}

export function useAnalysisJob(onDone: (runId: string) => void) {
  const [state, setState] = useState<JobState>({ phase: "idle" });
  const uploadRef = useRef<UploadHandle | null>(null);
  const onDoneRef = useRef(onDone);

  useEffect(() => {
    onDoneRef.current = onDone;
  }, [onDone]);

  // Abort an in-flight upload on unmount.
  useEffect(() => () => uploadRef.current?.abort(), []);

  const jobId = state.phase === "processing" ? state.jobId : null;
  const jobFile = state.phase === "processing" ? state.file : "";

  // Poll the job every second until done / error.
  useEffect(() => {
    if (!jobId) return;
    const ctrl = new AbortController();
    let timer = 0;
    let failures = 0;
    let stopped = false;
    const tick = async () => {
      try {
        const job = await api.job(jobId, ctrl.signal);
        if (stopped) return;
        failures = 0;
        if (job.status === "done") {
          if (job.run_id) {
            setState({ phase: "done", file: jobFile, job });
            onDoneRef.current(job.run_id);
          } else {
            setState({ phase: "error", file: jobFile, message: "İş bitti ama bir analiz kimliği döndürmedi." });
          }
          return;
        }
        if (job.status === "error") {
          setState({ phase: "error", file: jobFile, message: job.error ?? `Analiz “${stageLabel(job.stage)}” aşamasında durdu.` });
          return;
        }
        setState({ phase: "processing", file: jobFile, jobId, job, transientError: null });
      } catch (err) {
        if (stopped || isAbort(err)) return;
        failures += 1;
        if (failures >= MAX_POLL_FAILURES) {
          setState({ phase: "error", file: jobFile, message: `İşle bağlantı kesildi: ${errorText(err)}` });
          return;
        }
        setState((prev) => (prev.phase === "processing" ? { ...prev, transientError: errorText(err) } : prev));
      }
      if (!stopped) timer = window.setTimeout(() => void tick(), 1000);
    };
    void tick();
    return () => {
      stopped = true;
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, [jobId, jobFile]);

  const start = useCallback((file: File) => {
    uploadRef.current?.abort();
    setState({ phase: "uploading", file: file.name, fraction: 0 });
    const handle = uploadVideo(file, (fraction) =>
      setState((prev) => (prev.phase === "uploading" ? { ...prev, fraction } : prev)),
    );
    uploadRef.current = handle;
    handle.promise.then(
      (id) => {
        uploadRef.current = null;
        setState({ phase: "processing", file: file.name, jobId: id, job: null, transientError: null });
      },
      (err: unknown) => {
        uploadRef.current = null;
        setState({ phase: "error", file: file.name, message: errorText(err) });
      },
    );
  }, []);

  const cancel = useCallback(() => uploadRef.current?.abort(), []);
  const dismiss = useCallback(() => setState({ phase: "idle" }), []);

  return { state, start, cancel, dismiss };
}
