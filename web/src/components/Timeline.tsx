import {
  memo,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { bounceFrameOf, bounceFrameRaw, fmtClock, fmtProb, verdictColor } from "../geometry";
import { playerName, SIDE_LABEL, verdictLabel } from "../i18n";
import type { Report } from "../types";

type Props = {
  report: Report;
  calibValid: boolean[] | null;
  currentFrame: number;
  selectedFlightId: number | null;
  onSeek: (frame: number) => void;
  onSelectFlight: (id: number) => void;
};

const ZOOMS = [1, 2, 4, 8] as const;
const TICK_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];

/** "m:ss" ticks (with a decimal only for sub-second steps). */
function fmtTick(s: number, step: number): string {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  const sec = step < 1 ? r.toFixed(1).replace(".", ",").padStart(4, "0") : String(Math.round(r)).padStart(2, "0");
  return `${m}:${sec}`;
}

function TimelineImpl({ report, calibValid, currentFrame, selectedFlightId, onSeek, onSelectFlight }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const scrubbing = useRef(false);
  const [zoom, setZoom] = useState<number>(1);
  const fps = report.video.fps > 0 ? report.video.fps : 30;
  const nFrames = Math.max(1, report.video.n_frames);
  const duration = report.video.duration_s > 0 ? report.video.duration_s : nFrames / fps;
  const pct = (frame: number) => Math.max(0, Math.min(100, (frame / fps / duration) * 100));
  // Many shots on a long clip: shrink markers so neighbours do not overlap (zoom in for detail).
  const dense = report.flights.length / zoom > 30;

  const { ticks, step } = useMemo(() => {
    const span = duration / zoom;
    const st = TICK_STEPS.find((s) => span / s <= 10) ?? TICK_STEPS[TICK_STEPS.length - 1];
    const out: number[] = [];
    for (let s = 0; s <= duration + 1e-6; s += st) out.push(Math.round(s * 10) / 10);
    return { ticks: out, step: st };
  }, [duration, zoom]);

  // Runs of uncalibrated frames (honesty: show where the overlay / calls cannot be trusted).
  const uncalibrated = useMemo(() => {
    if (!calibValid) return [];
    const runs: [number, number][] = [];
    let start = -1;
    calibValid.forEach((ok, i) => {
      if (!ok && start < 0) start = i;
      if (ok && start >= 0) {
        runs.push([start, i - 1]);
        start = -1;
      }
    });
    if (start >= 0) runs.push([start, calibValid.length - 1]);
    return runs;
  }, [calibValid]);

  const cuts = useMemo(() => report.calibration.shot_cuts.filter((f) => f > 0), [report.calibration.shot_cuts]);

  // Keep the playhead visible when zoomed in (unless the user is scrubbing).
  useEffect(() => {
    const sc = scrollRef.current;
    if (!sc || zoom === 1 || scrubbing.current) return;
    const x = (pct(currentFrame) / 100) * sc.scrollWidth;
    const margin = sc.clientWidth * 0.12;
    if (x < sc.scrollLeft + margin || x > sc.scrollLeft + sc.clientWidth - margin) {
      sc.scrollLeft = Math.max(0, x - sc.clientWidth * 0.35);
    }
  }, [currentFrame, zoom]); // pct() only depends on fps/duration, which are fixed for a run

  const frameFromClientX = (clientX: number): number | null => {
    const el = trackRef.current;
    if (!el) return null;
    const r = el.getBoundingClientRect();
    if (r.width <= 0) return null;
    const x = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    return Math.max(0, Math.min(nFrames - 1, Math.round(x * duration * fps)));
  };

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    scrubbing.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    const f = frameFromClientX(e.clientX);
    if (f !== null) onSeek(f);
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!scrubbing.current) return;
    const f = frameFromClientX(e.clientX);
    if (f !== null && f !== currentFrame) onSeek(f);
  };
  const endScrub = (e: ReactPointerEvent<HTMLDivElement>) => {
    scrubbing.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const stopPointer = (e: ReactPointerEvent) => e.stopPropagation();

  return (
    <div className={`timeline ${dense ? "timeline--dense" : ""}`}>
      <div className="tl-lanes" aria-hidden="true">
        <span>Uzak</span>
        <span>Yakın</span>
      </div>
      <div className={`tl-scroll ${zoom === 1 ? "is-fit" : ""}`} ref={scrollRef}>
        <div className="tl-inner" style={{ width: `${zoom * 100}%` }}>
          <div
            ref={trackRef}
            className="tl-track"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endScrub}
            onPointerCancel={endScrub}
            onKeyDown={(e) => {
              if (e.key === "Home") {
                e.preventDefault();
                onSeek(0);
              } else if (e.key === "End") {
                e.preventDefault();
                onSeek(nFrames - 1);
              }
            }}
            role="slider"
            tabIndex={0}
            aria-label="Zaman çizelgesi"
            aria-valuemin={0}
            aria-valuemax={nFrames - 1}
            aria-valuenow={currentFrame}
            aria-valuetext={`kare ${currentFrame}, ${fmtClock(currentFrame / fps)}`}
          >
            <div className="tl-mid" />
            {uncalibrated.map(([a, b]) => (
              <div
                key={`u${a}`}
                className="tl-uncal"
                style={{ left: `${pct(a)}%`, width: `${Math.max(0.15, pct(b + 1) - pct(a))}%` }}
                title={`${a}–${b}. kareler kalibre değil (kort katmanı ve karar yok)`}
              />
            ))}
            {cuts.map((f) => (
              <div key={`c${f}`} className="tl-cut" style={{ left: `${pct(f)}%` }} title={`${f}. karede kamera geçişi`} />
            ))}
            {report.flights.map((f, i) => (
              <div
                key={`f${f.id}`}
                className={`tl-flight ${i % 2 ? "is-odd" : ""} ${f.id === selectedFlightId ? "is-selected" : ""}`}
                style={{ left: `${pct(f.start_frame)}%`, width: `${Math.max(0, pct(f.end_frame) - pct(f.start_frame))}%` }}
              />
            ))}
            {report.hits.map((h, i) => {
              const who = playerName(report.players[h.side].label);
              return (
                <button
                  key={`h${i}`}
                  type="button"
                  className={`tl-hit tl-hit--${h.side}`}
                  style={{ left: `${pct(h.frame)}%` }}
                  onPointerDown={stopPointer}
                  onClick={() => onSeek(h.frame)}
                  aria-label={`${who} vuruşu, ${fmtClock(h.t)}, kare ${h.frame}`}
                  title={`${h.serve ? "Servis" : "Vuruş"} · ${who} (${SIDE_LABEL[h.side]}) · kare ${h.frame}${h.next_to_player ? "" : " · oyuncunun yanında değil"}`}
                />
              );
            })}
            {report.flights.map((f) => {
              const bf = bounceFrameOf(f);
              const raw = bounceFrameRaw(f);
              if (bf === null || raw === null) return null;
              const label = f.call ? `${verdictLabel(f.call.verdict)} · ${fmtProb(f.call.p_in)}` : "karar yok";
              return (
                <button
                  key={`b${f.id}`}
                  type="button"
                  className={`tl-bounce ${f.call ? "" : "tl-bounce--nocall"} ${f.id === selectedFlightId ? "is-selected" : ""}`}
                  style={{ left: `${pct(raw)}%`, "--c": verdictColor(f.call?.verdict) } as CSSProperties}
                  onPointerDown={stopPointer}
                  onClick={() => {
                    onSelectFlight(f.id);
                    onSeek(bf);
                  }}
                  aria-label={`Vuruş ${f.id + 1} sekmesi: ${label}, kare ${bf}`}
                  title={`Vuruş ${f.id + 1} sekmesi · ${label} · kare ${bf}`}
                />
              );
            })}
            <div className="tl-playhead" style={{ left: `${pct(currentFrame)}%` }} />
          </div>
          <div className="tl-ticks num" aria-hidden="true">
            {ticks.map((s) => {
              const at = Math.min(100, (s / duration) * 100);
              // Keep edge labels inside the track.
              const shift = at < 1.5 ? "0%" : at > 97 ? "-100%" : "-50%";
              return (
                <span key={s} style={{ left: `${at}%`, transform: `translateX(${shift})` }}>
                  {fmtTick(s, step)}
                </span>
              );
            })}
          </div>
        </div>
      </div>
      <div className="tl-foot">
        <div className="tl-legend" aria-hidden="true">
          <span>
            <i className="lg-tick" /> vuruş
          </span>
          <span>
            <i className="lg-dot" style={{ "--c": verdictColor("IN") } as CSSProperties} /> içeride
          </span>
          <span>
            <i className="lg-dot" style={{ "--c": verdictColor("OUT") } as CSSProperties} /> dışarıda
          </span>
          <span>
            <i className="lg-dot" style={{ "--c": verdictColor("REVIEW") } as CSSProperties} /> inceleme
          </span>
          {uncalibrated.length > 0 && (
            <span>
              <i className="lg-hatch" /> kalibre değil
            </span>
          )}
          {cuts.length > 0 && (
            <span>
              <i className="lg-cut" /> kamera geçişi
            </span>
          )}
        </div>
        <div className="tl-right">
          <span className="tl-hint">← → kare · ⇧ 10 kare · boşluk oynat</span>
          <div className="segmented segmented--sm" role="group" aria-label="Yakınlaştırma">
            {ZOOMS.map((z) => (
              <button key={z} type="button" className="num" aria-pressed={zoom === z} onClick={() => setZoom(z)}>
                {z === 1 ? "Sığdır" : `${z}×`}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

const Timeline = memo(TimelineImpl);
export default Timeline;
