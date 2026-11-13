import { useCallback, useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import { mediaUrl } from "../api";
import {
  bounceFrameRaw,
  COURT_LINES,
  ellipsePoints,
  fmtClock,
  fmtNum,
  frameAt,
  PALETTE,
  projectH,
  verdictColor,
  withAlpha,
  type Vec2,
} from "../geometry";
import { MEDIA_LABEL, playerName } from "../i18n";
import { FrameBackIcon, FrameForwardIcon, PauseIcon, PlayIcon, Spinner } from "../icons";
import type { Async, Flight, Frames, MediaName, PlayerLabel, Report, Side } from "../types";

export type VideoStageHandle = {
  /** Seek so that Math.round(mediaTime * fps) === frame. */
  seekToFrame: (frame: number, pause?: boolean) => void;
  /** Call before switching the video source so playback resumes at the same moment. */
  rememberPosition: () => void;
};

export type Layers = { court: boolean; ball: boolean; players: boolean; bounces: boolean };

export const LAYER_LABELS: { key: keyof Layers; label: string }[] = [
  { key: "court", label: "Kort çizgileri" },
  { key: "ball", label: "Top ve izi" },
  { key: "players", label: "Oyuncular" },
  { key: "bounces", label: "Sekmeler (%95 elips)" },
];

const DEFAULT_LAYERS: Layers = { court: true, ball: true, players: true, bounces: true };
const LAYERS_KEY = "courtvision.overlayLayers";

export function loadLayers(): Layers {
  try {
    const raw = window.localStorage.getItem(LAYERS_KEY);
    if (!raw) return DEFAULT_LAYERS;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return DEFAULT_LAYERS;
    const rec = parsed as Record<string, unknown>;
    const pick = (k: keyof Layers) => (typeof rec[k] === "boolean" ? (rec[k] as boolean) : DEFAULT_LAYERS[k]);
    return { court: pick("court"), ball: pick("ball"), players: pick("players"), bounces: pick("bounces") };
  } catch {
    return DEFAULT_LAYERS;
  }
}

export function saveLayers(l: Layers) {
  try {
    window.localStorage.setItem(LAYERS_KEY, JSON.stringify(l));
  } catch {
    /* storage unavailable (private mode / quota) — ignore */
  }
}

type Props = {
  ref?: Ref<VideoStageHandle>;
  runId: string;
  media: MediaName[];
  view: MediaName;
  layers: Layers;
  report: Report;
  frames: Async<Frames>;
  currentFrame: number;
  selectedFlightId: number | null;
  onFrame: (frame: number) => void;
  /** Ground-truth labelling: clicks on the video report the frame and the pixel. */
  labelMode?: boolean;
  onLabel?: (frame: number, px: [number, number]) => void;
  onCancelLabel?: () => void;
};

const LOUPE = 150; // CSS px
const LOUPE_ZOOM = 4;

const RATES = [1, 0.5, 0.25] as const;
const TRAIL = 10;

const MEDIA_ERRORS: Record<number, string> = {
  1: "oynatma durduruldu",
  2: "video yüklenirken ağ hatası oluştu",
  3: "video çözülemedi",
  4: "video yok ya da biçimi desteklenmiyor",
};

type DrawInputs = {
  frames: Frames | null;
  flights: Flight[];
  labels: Record<Side, PlayerLabel>;
  layers: Layers;
  selectedFlightId: number | null;
  enabled: boolean;
};

const SIDES: Side[] = ["near", "far"];
const SIDE_COLOR: Record<Side, string> = { near: PALETTE.near, far: PALETTE.far };

function drawOverlay(canvas: HTMLCanvasElement, frame: number, inp: DrawInputs) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const fr = inp.frames;
  if (!inp.enabled || !fr) return;
  // Sizes below are in CSS pixels; s converts them to canvas (= video) pixels so the overlay stays
  // legible whatever the video resolution and on-screen size.
  const cssH = canvas.clientHeight > 0 ? canvas.clientHeight : h;
  const s = Math.max(0.25, h / cssH);
  const H = frame >= 0 && frame < fr.H.length ? fr.H[frame] : null;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Court lines (projected with this frame's homography) -------------------------------------
  if (inp.layers.court && H) {
    ctx.save();
    ctx.strokeStyle = withAlpha(PALETTE.courtLines, 0.9);
    ctx.lineWidth = 1.5 * s;
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 2 * s;
    ctx.beginPath();
    for (const seg of COURT_LINES) {
      const a = projectH(H, seg.a[0], seg.a[1]);
      const b = projectH(H, seg.b[0], seg.b[1]);
      if (!a || !b) continue;
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Bounce markers: 95% ellipse of call.cov (48 world points projected with H) ---------------
  if (inp.layers.bounces && H) {
    for (const f of inp.flights) {
      const bf = bounceFrameRaw(f); // draw when bounce.frame <= current frame
      if (bf === null || bf > frame) continue;
      const mean: Vec2 | null = f.call ? [f.call.x, f.call.y] : f.bounce ? [f.bounce.x, f.bounce.y] : null;
      const cov = f.call ? f.call.cov : f.bounce ? f.bounce.cov : null;
      if (!mean || !cov) continue;
      const pts = ellipsePoints(mean, cov).map((p) => projectH(H, p[0], p[1]));
      const c = projectH(H, mean[0], mean[1]);
      if (!c || pts.some((p) => p === null)) continue;
      const color = f.call ? verdictColor(f.call.verdict) : PALETTE.muted;
      const selected = f.id === inp.selectedFlightId;
      ctx.save();
      ctx.beginPath();
      pts.forEach((p, i) => {
        if (!p) return;
        if (i === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
      });
      ctx.closePath();
      ctx.fillStyle = withAlpha(color, 0.3);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = (selected ? 2 : 1.25) * s;
      ctx.stroke();
      const k = (selected ? 7 : 5) * s;
      ctx.beginPath();
      ctx.moveTo(c[0] - k, c[1]);
      ctx.lineTo(c[0] + k, c[1]);
      ctx.moveTo(c[0], c[1] - k);
      ctx.lineTo(c[0], c[1] + k);
      ctx.lineWidth = 1.25 * s;
      ctx.stroke();
      if (selected) {
        ctx.beginPath();
        ctx.arc(c[0], c[1], 16 * s, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.5 * s;
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  // Players: boxes with a small dark label "Oyuncu 1 · 12,3 km/sa" ----------------------------
  if (inp.layers.players) {
    ctx.save();
    ctx.font = `600 ${Math.round(11.5 * s)}px -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    for (const side of SIDES) {
      const pf = fr.players[side];
      const box = pf && frame < pf.box.length ? pf.box[frame] : null;
      if (!box) continue;
      const [x1, y1, x2, y2] = box;
      const color = SIDE_COLOR[side];
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5 * s;
      ctx.beginPath();
      ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 4 * s);
      ctx.stroke();
      const v = frame < pf.speed_kmh.length ? pf.speed_kmh[frame] : null;
      const text = `${playerName(inp.labels[side])} · ${v === null || v === undefined ? "—" : fmtNum(v, 1)} km/sa`;
      const tw = ctx.measureText(text).width;
      const ph = 19 * s;
      const pw = tw + 22 * s;
      const px = Math.max(0, Math.min(w - pw, x1));
      const py = y1 - ph - 4 * s >= 0 ? y1 - ph - 4 * s : y2 + 4 * s;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.beginPath();
      ctx.roundRect(px, py, pw, ph, ph / 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px + 9 * s, py + ph / 2, 3 * s, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = PALETTE.text;
      ctx.fillText(text, px + 16 * s, py + ph / 2 + 0.5);
    }
    ctx.restore();
  }

  // Ball: dot with a 10-frame trail (interpolated positions drawn hollow) -------------------
  if (inp.layers.ball) {
    ctx.save();
    const start = Math.max(0, frame - TRAIL + 1);
    let prev: Vec2 | null = null;
    for (let i = start; i <= frame && i < fr.ball.length; i++) {
      const obs = fr.ball[i];
      if (!obs || obs[2] === 0) {
        prev = null;
        continue;
      }
      const age = frame - i; // 0 = current
      const alpha = 1 - age / TRAIL;
      const p: Vec2 = [obs[0], obs[1]];
      if (prev) {
        ctx.beginPath();
        ctx.moveTo(prev[0], prev[1]);
        ctx.lineTo(p[0], p[1]);
        ctx.strokeStyle = withAlpha(PALETTE.ball, 0.55 * alpha);
        ctx.lineWidth = Math.max(1, 2.5 * s * alpha);
        ctx.stroke();
      }
      prev = p;
    }
    const cur = frame < fr.ball.length ? fr.ball[frame] : null;
    if (cur && cur[2] !== 0) {
      ctx.beginPath();
      ctx.arc(cur[0], cur[1], 4.5 * s, 0, Math.PI * 2);
      if (cur[2] === 2) {
        ctx.setLineDash([2.5 * s, 2.5 * s]);
        ctx.strokeStyle = PALETTE.ball;
        ctx.lineWidth = 1.5 * s;
        ctx.stroke();
      } else {
        ctx.fillStyle = PALETTE.ball;
        ctx.shadowColor = "rgba(0,0,0,0.7)";
        ctx.shadowBlur = 3 * s;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = "rgba(0,0,0,0.8)";
        ctx.lineWidth = 1 * s;
        ctx.stroke();
      }
    }
    ctx.restore();
  }
}

function supportsRVFC(): boolean {
  return typeof HTMLVideoElement !== "undefined" && "requestVideoFrameCallback" in HTMLVideoElement.prototype;
}

function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable;
}

export default function VideoStage({
  ref,
  runId,
  media,
  view,
  layers,
  report,
  frames,
  currentFrame,
  selectedFlightId,
  onFrame,
  labelMode = false,
  onLabel,
  onCancelLabel,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const loupeRef = useRef<HTMLCanvasElement>(null);
  const [loupe, setLoupe] = useState<{ x: number; y: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tab: MediaName | null = media.includes(view) ? view : (media[0] ?? null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState<number>(1);
  const [mediaStatus, setMediaStatus] = useState<"loading" | "ready" | "buffering" | "error">("loading");
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number }>({
    w: report.video.width > 0 ? report.video.width : 1920,
    h: report.video.height > 0 ? report.video.height : 1080,
  });

  const framesData = frames.status === "ready" ? frames.data : null;
  const fps = report.video.fps > 0 ? report.video.fps : framesData && framesData.fps > 0 ? framesData.fps : 30;
  const nFrames = report.video.n_frames > 0 ? report.video.n_frames : (framesData?.n ?? 1);
  const duration = report.video.duration_s > 0 ? report.video.duration_s : nFrames / fps;

  // Latest values for imperative callbacks (rVFC / keyboard) ---------------------------------
  const fpsRef = useRef(fps);
  const nRef = useRef(nFrames);
  const frameRef = useRef(currentFrame);
  const onFrameRef = useRef(onFrame);
  const pendingRestore = useRef<{ time: number; play: boolean } | null>(null);
  const inputsRef = useRef<DrawInputs>({
    frames: null,
    flights: [],
    labels: { near: "P1", far: "P2" },
    layers,
    selectedFlightId,
    enabled: false,
  });

  useEffect(() => {
    fpsRef.current = fps;
    nRef.current = nFrames;
    onFrameRef.current = onFrame;
  }, [fps, nFrames, onFrame]);

  const redraw = useCallback((frame: number) => {
    const c = canvasRef.current;
    if (c) drawOverlay(c, frame, inputsRef.current);
  }, []);

  const publishFrame = useCallback(
    (frame: number) => {
      redraw(frame);
      if (frame !== frameRef.current) {
        frameRef.current = frame;
        onFrameRef.current(frame);
      }
    },
    [redraw],
  );

  // Keep draw inputs current and redraw whenever they change.
  useEffect(() => {
    inputsRef.current = {
      frames: framesData,
      flights: report.flights,
      labels: { near: report.players.near.label, far: report.players.far.label },
      layers,
      selectedFlightId,
      // While a bounce is being marked, nothing the system measured is drawn: the label must not lean
      // on the prediction it is meant to test.
      enabled: tab === "source" && !labelMode,
    };
    redraw(frameRef.current);
  }, [framesData, report, layers, selectedFlightId, tab, dims, redraw, labelMode]);

  // A new source starts loading: show the loading state until it can play.
  useEffect(() => {
    setMediaStatus("loading");
    setMediaError(null);
    setPlaying(false);
  }, [tab]);

  // The overlay's stroke sizes depend on the on-screen size: redraw when the stage is resized.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => redraw(frameRef.current));
    ro.observe(c);
    return () => ro.disconnect();
  }, [redraw]);

  // Sync the overlay with the decoded video frame ---------------------------------------------
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let disposed = false;
    let rvfcId: number | null = null;
    let raf = 0;
    const fromCurrentTime = () => {
      if (!disposed) publishFrame(frameAt(v.currentTime, fpsRef.current, nRef.current));
    };
    const cleanups: (() => void)[] = [];
    const listen = (type: keyof HTMLMediaElementEventMap, fn: () => void) => {
      v.addEventListener(type, fn);
      cleanups.push(() => v.removeEventListener(type, fn));
    };

    if (supportsRVFC()) {
      const cb: VideoFrameRequestCallback = (_now, meta) => {
        if (disposed) return;
        // While a seek is pending the presented frame is stale; the "seeked" handler and the next
        // callback report the new frame. Ignoring it keeps frame stepping anchored to the seek target.
        if (!v.seeking) publishFrame(frameAt(meta.mediaTime, fpsRef.current, nRef.current));
        rvfcId = v.requestVideoFrameCallback(cb);
      };
      rvfcId = v.requestVideoFrameCallback(cb);
    } else {
      // Fallback: timeupdate + rAF while playing.
      const loop = () => {
        raf = 0;
        if (disposed) return;
        fromCurrentTime();
        if (!v.paused && !v.ended) raf = requestAnimationFrame(loop);
      };
      listen("timeupdate", fromCurrentTime);
      listen("play", () => {
        if (!raf) raf = requestAnimationFrame(loop);
      });
    }
    listen("seeked", fromCurrentTime);
    listen("loadeddata", fromCurrentTime);

    return () => {
      disposed = true;
      if (rvfcId !== null && typeof v.cancelVideoFrameCallback === "function") v.cancelVideoFrameCallback(rvfcId);
      if (raf) cancelAnimationFrame(raf);
      cleanups.forEach((fn) => fn());
    };
  }, [tab, publishFrame]);

  // Imperative API ------------------------------------------------------------------------------
  const seekToFrame = useCallback(
    (frame: number, pause = false) => {
      const v = videoRef.current;
      const n = nRef.current;
      const f = Math.max(0, Math.min(n - 1, Math.round(frame)));
      if (v) {
        if (pause && !v.paused) v.pause();
        // Land inside the frame's display interval so that round(mediaTime * fps) === f.
        v.currentTime = (f + 0.25) / fpsRef.current;
      }
      publishFrame(f);
    },
    [publishFrame],
  );

  const rememberPosition = useCallback(() => {
    const v = videoRef.current;
    if (v) pendingRestore.current = { time: v.currentTime, play: !v.paused && !v.ended };
  }, []);

  useImperativeHandle(ref, () => ({ seekToFrame, rememberPosition }), [seekToFrame, rememberPosition]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused || v.ended) {
      if (v.ended || frameRef.current >= nRef.current - 1) v.currentTime = 0;
      v.play().catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setMediaError(`Oynatılamadı: ${err instanceof Error ? err.message : String(err)}`);
      });
    } else {
      v.pause();
    }
  }, []);

  const step = useCallback((delta: number) => seekToFrame(frameRef.current + delta, true), [seekToFrame]);

  // Keyboard: ← / → step one frame (⇧ ten), space toggles play/pause, Esc leaves labelling.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      if (isTypingTarget(e.target)) return;
      if (e.key === "Escape" && labelMode) {
        e.preventDefault();
        onCancelLabel?.();
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        step(e.shiftKey ? -10 : -1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        step(e.shiftKey ? 10 : 1);
      } else if (e.key === " " || e.code === "Space") {
        if (e.target instanceof HTMLButtonElement) return; // the focused button handles space itself
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, togglePlay, labelMode, onCancelLabel]);

  /** Video pixel under a pointer event on the stage. */
  const videoPx = (e: { clientX: number; clientY: number; currentTarget: Element }): [number, number, number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    return [(x * dims.w) / r.width, (y * dims.h) / r.height, x, y];
  };

  const drawLoupe = (u: number, v: number) => {
    const c = loupeRef.current;
    const vid = videoRef.current;
    const ctx = c?.getContext("2d");
    if (!c || !vid || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== LOUPE * dpr) {
      c.width = LOUPE * dpr;
      c.height = LOUPE * dpr;
    }
    const half = LOUPE / LOUPE_ZOOM / 2;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(vid, u - half, v - half, 2 * half, 2 * half, 0, 0, c.width, c.height);
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.lineWidth = dpr;
    const m = c.width / 2;
    ctx.beginPath();
    ctx.moveTo(m, m - 18 * dpr);
    ctx.lineTo(m, m - 5 * dpr);
    ctx.moveTo(m, m + 5 * dpr);
    ctx.lineTo(m, m + 18 * dpr);
    ctx.moveTo(m - 18 * dpr, m);
    ctx.lineTo(m - 5 * dpr, m);
    ctx.moveTo(m + 5 * dpr, m);
    ctx.lineTo(m + 18 * dpr, m);
    ctx.stroke();
  };

  const cycleRate = () => {
    const next = RATES[(RATES.indexOf(rate as (typeof RATES)[number]) + 1) % RATES.length];
    setRate(next);
    const v = videoRef.current;
    if (v) {
      v.defaultPlaybackRate = next;
      v.playbackRate = next;
    }
  };

  const retryMedia = () => {
    const v = videoRef.current;
    if (!v) return;
    pendingRestore.current = { time: v.currentTime, play: false };
    setMediaStatus("loading");
    setMediaError(null);
    v.load();
  };

  const H = framesData && currentFrame < framesData.H.length ? framesData.H[currentFrame] : null;
  const overlayActive = tab === "source";
  // A rendered view has the system's marks burned in: clicks are taken on the raw video only.
  const canLabel = labelMode && tab === "source";
  const ar = dims.w / dims.h;
  const t = currentFrame / fps;

  if (!tab) {
    return (
      <div className="stage-wrap">
        <div className="stage" style={{ aspectRatio: `${ar}` }}>
          <div className="stage-cover">
            <p className="empty-title" style={{ color: "#fff" }}>
              Bu analiz için video yok
            </p>
            <p className="empty-text" style={{ color: "rgb(255 255 255 / 0.7)" }}>
              Sunucu “{runId}” için kaynak, klasik ya da pro video bildirmiyor.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="stage-wrap" style={{ ["--ar" as string]: ar.toFixed(4) }}>
      <div
        className={`stage${playing ? " is-playing" : ""}${overlayActive ? "" : " stage--burned"}`}
        style={{ aspectRatio: `${dims.w} / ${dims.h}` }}
      >
        <video
          ref={videoRef}
          className="stage-video"
          src={mediaUrl(runId, tab)}
          preload="auto"
          muted
          playsInline
          onClick={togglePlay}
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            if (v.videoWidth > 0 && v.videoHeight > 0) setDims({ w: v.videoWidth, h: v.videoHeight });
            v.defaultPlaybackRate = rate;
            v.playbackRate = rate;
            const p = pendingRestore.current;
            pendingRestore.current = null;
            if (p) {
              v.currentTime = p.time;
              if (p.play) void v.play().catch(() => undefined);
            }
          }}
          onLoadedData={() => setMediaStatus("ready")}
          onCanPlay={() => setMediaStatus("ready")}
          onPlaying={() => setMediaStatus("ready")}
          onWaiting={() => setMediaStatus("buffering")}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onError={(e) => {
            const err = e.currentTarget.error;
            setMediaStatus("error");
            setMediaError(
              `${MEDIA_LABEL[tab]} videosu yüklenemedi: ${err ? (MEDIA_ERRORS[err.code] ?? err.message) : "bilinmeyen hata"}.`,
            );
          }}
        />
        <canvas
          ref={canvasRef}
          className="stage-canvas"
          width={dims.w}
          height={dims.h}
          aria-hidden="true"
          style={{ visibility: overlayActive ? "visible" : "hidden" }}
        />

        {canLabel && (
          <div
            className="label-layer"
            onPointerMove={(e) => {
              const [u, v, x, y] = videoPx(e);
              setLoupe({ x, y });
              drawLoupe(u, v);
            }}
            onPointerLeave={() => setLoupe(null)}
            onClick={(e) => {
              const [u, v] = videoPx(e);
              onLabel?.(frameRef.current, [Math.round(u * 10) / 10, Math.round(v * 10) / 10]);
            }}
          >
            <canvas
              ref={loupeRef}
              className="loupe"
              aria-hidden="true"
              style={{
                visibility: loupe ? "visible" : "hidden",
                left: loupe ? Math.max(4, loupe.x - LOUPE / 2) : 0,
                top: loupe ? (loupe.y > LOUPE + 30 ? loupe.y - LOUPE - 24 : loupe.y + 24) : 0,
              }}
            />
          </div>
        )}

        <div className="stage-badges">
          {labelMode && (
            <span className="badge badge--warn glass-clear">
              <span className="badge-dot" aria-hidden="true" />
              {canLabel
                ? "Topun yere değdiği noktaya tıklayın · ← → kare · Esc vazgeç · sistem katmanları gizli"
                : "Sekme yalnız Kaynak videoda işaretlenir (işlenmiş videoda sistemin işaretleri var) · Esc vazgeç"}
            </span>
          )}
          {overlayActive ? (
            <span className="badge glass-clear">
              <span className="badge-dot" aria-hidden="true" />
              Canlı katman
            </span>
          ) : (
            <span className="badge glass-clear">{MEDIA_LABEL[tab]} · videoya işlenmiş katman</span>
          )}
          {overlayActive && frames.status === "loading" && (
            <span className="badge glass-clear">
              <Spinner size={12} /> Katman verisi yükleniyor
            </span>
          )}
          {overlayActive && frames.status === "error" && (
            <span className="badge badge--warn glass-clear" title={frames.error}>
              <span className="badge-dot" aria-hidden="true" />
              Katman verisi alınamadı
            </span>
          )}
          {overlayActive && framesData && !H && (
            <span className="badge badge--warn glass-clear">
              <span className="badge-dot" aria-hidden="true" />
              Bu kare kalibre değil · kort ve sekme gösterilmiyor
            </span>
          )}
        </div>

        <div className="player-bar glass-clear" role="group" aria-label="Oynatma">
          <button type="button" className="pb-btn" onClick={() => step(-1)} aria-label="Önceki kare" title="Önceki kare (←, ⇧ ile 10 kare)">
            <FrameBackIcon size={17} />
          </button>
          <button
            type="button"
            className="pb-btn pb-btn--play"
            onClick={togglePlay}
            aria-label={playing ? "Duraklat" : "Oynat"}
            title={playing ? "Duraklat (boşluk)" : "Oynat (boşluk)"}
          >
            {playing ? <PauseIcon size={20} /> : <PlayIcon size={20} />}
          </button>
          <button type="button" className="pb-btn" onClick={() => step(1)} aria-label="Sonraki kare" title="Sonraki kare (→, ⇧ ile 10 kare)">
            <FrameForwardIcon size={17} />
          </button>
          <span className="pb-sep" aria-hidden="true" />
          <span className="pb-time" aria-live="off">
            {fmtClock(t)} <span>/ {fmtClock(duration)} · kare {currentFrame}</span>
          </span>
          <button type="button" className="pb-btn pb-rate" onClick={cycleRate} title="Oynatma hızı" aria-label={`Oynatma hızı ${rate}×`}>
            {String(rate).replace(".", ",")}×
          </button>
        </div>

        {(mediaStatus === "loading" || mediaStatus === "buffering") && !mediaError && (
          <div className="stage-cover stage-cover--soft" aria-live="polite">
            <Spinner size={22} />
          </div>
        )}
        {mediaError && (
          <div className="stage-cover" role="alert">
            <p className="empty-title" style={{ color: "#fff" }}>
              Video açılamadı
            </p>
            <p className="empty-text" style={{ color: "rgb(255 255 255 / 0.75)" }}>
              {mediaError}
            </p>
            <button type="button" className="btn" style={{ color: "#fff", background: "rgb(255 255 255 / 0.18)" }} onClick={retryMedia}>
              Yeniden Dene
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
