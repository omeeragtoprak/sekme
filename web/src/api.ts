// Typed fetch helpers for the CourtVision FastAPI backend (same origin, under /api).
import type { Frames, Health, Job, LabelRecord, LabelSummary, MediaName, Report, Review, ReviewIn, RunSummary } from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Extract a readable message from a FastAPI error body ({detail: string | {msg}[]}) or plain text. */
function errorMessageFromBody(body: unknown, fallback: string): string {
  if (isRecord(body)) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const msgs = detail
        .map((d) => (isRecord(d) && typeof d.msg === "string" ? d.msg : null))
        .filter((m): m is string => m !== null);
      if (msgs.length > 0) return msgs.join("; ");
    }
    if (typeof body.error === "string") return body.error;
  }
  if (typeof body === "string") {
    const t = body.trim();
    // Proxies / static servers answer with HTML pages; never dump markup into the UI.
    if (t.length > 0 && !t.startsWith("<")) return t.slice(0, 300);
  }
  return fallback;
}

async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (text.length === 0) return null;
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed;
  } catch {
    return text;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: { Accept: "application/json", ...(init?.headers ?? {}) } });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(`Ağ hatası — CourtVision sunucusuna ulaşılamıyor (${path})`, 0);
  }
  const body = await readBody(res);
  if (!res.ok) {
    throw new ApiError(`${res.status} — ${errorMessageFromBody(body, path)}`, res.status);
  }
  if (body === null || typeof body === "string") {
    throw new ApiError(`${path} beklenmeyen bir yanıt döndürdü (JSON değil)`, res.status);
  }
  return body as T;
}

const enc = encodeURIComponent;

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/api/health", { signal }),
  runs: (signal?: AbortSignal) => request<RunSummary[]>("/api/runs", { signal }),
  report: (runId: string, signal?: AbortSignal) => request<Report>(`/api/runs/${enc(runId)}/report`, { signal }),
  frames: (runId: string, signal?: AbortSignal) => request<Frames>(`/api/runs/${enc(runId)}/frames`, { signal }),
  reviews: (runId: string, signal?: AbortSignal) => request<Review[]>(`/api/runs/${enc(runId)}/reviews`, { signal }),
  postReview: (runId: string, body: ReviewIn, signal?: AbortSignal) =>
    request<Review>(`/api/runs/${enc(runId)}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }),
  job: (jobId: string, signal?: AbortSignal) => request<Job>(`/api/jobs/${enc(jobId)}`, { signal }),
  labels: (runId: string, signal?: AbortSignal) => request<LabelRecord[]>(`/api/runs/${enc(runId)}/labels`, { signal }),
  labelSummary: (runId: string, signal?: AbortSignal) =>
    request<LabelSummary>(`/api/runs/${enc(runId)}/labels/summary`, { signal }),
  postLabel: (
    runId: string,
    body: { flight_id: number; frame: number; px: [number, number]; labeler: string; note: string },
  ) =>
    request<LabelRecord>(`/api/runs/${enc(runId)}/labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export function mediaUrl(runId: string, name: MediaName): string {
  return `/api/runs/${enc(runId)}/media/${name}`;
}

export type UploadHandle = { promise: Promise<string>; abort: () => void };

/**
 * POST /api/jobs (multipart, field "file") -> 202 {"job_id"}.
 * Uses XMLHttpRequest (not fetch) so that upload progress can be reported.
 */
export function uploadVideo(file: File, onProgress: (fraction: number) => void): UploadHandle {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<string>((resolve, reject) => {
    xhr.open("POST", "/api/jobs");
    xhr.responseType = "text";
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiError("Video yüklenirken ağ hatası oluştu", 0));
    xhr.onabort = () => reject(new ApiError("Yükleme durduruldu", 0));
    xhr.onload = () => {
      let body: unknown = xhr.responseText;
      try {
        body = JSON.parse(xhr.responseText) as unknown;
      } catch {
        /* keep text body */
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new ApiError(`${xhr.status} — ${errorMessageFromBody(body, "yükleme reddedildi")}`, xhr.status));
        return;
      }
      if (isRecord(body) && typeof body.job_id === "string") resolve(body.job_id);
      else reject(new ApiError("Yükleme kabul edildi ama yanıtta iş kimliği yok", xhr.status));
    };
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}

export function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

export function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}
