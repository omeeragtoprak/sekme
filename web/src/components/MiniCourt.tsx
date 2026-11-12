import { memo, useMemo, type CSSProperties, type KeyboardEvent } from "react";
import {
  bounceFrameRaw,
  COURT,
  COURT_LINES,
  ellipsePoints,
  fmtNum,
  fmtProb,
  fmtSignedCm,
  PALETTE,
  verdictColor,
  type Segment,
  type Vec2,
} from "../geometry";
import { lineLabel, playerName, SIDE_LABEL, verdictLabel } from "../i18n";
import type { Cov2, Flight, Frames, Report, Side } from "../types";

type Props = {
  report: Report;
  frames: Frames | null;
  currentFrame: number;
  selectedFlightId: number | null;
  onSelectFlight: (id: number) => void;
};

const SIDE_COLOR: Record<Side, string> = { near: PALETTE.near, far: PALETTE.far };
const SIDES: Side[] = ["near", "far"];

/** SVG user unit = 1 cm (keeps SVG text at sane font sizes). */
const U = 100;
/** World (x, y) metres -> SVG (x, -y) centimetres: far baseline (y > 0) at the top. */
const sx = (x: number) => x * U;
const sy = (y: number) => -y * U;

function polyPoints(pts: Vec2[]): string {
  return pts.map((p) => `${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`).join(" ");
}

type BounceMark = { flight: Flight; mean: Vec2; cov: Cov2; frame: number; color: string; hasCall: boolean };

function bounceMarks(flights: Flight[]): BounceMark[] {
  const out: BounceMark[] = [];
  for (const f of flights) {
    const frame = bounceFrameRaw(f);
    if (frame === null) continue;
    if (f.call) out.push({ flight: f, mean: [f.call.x, f.call.y], cov: f.call.cov, frame, color: verdictColor(f.call.verdict), hasCall: true });
    else if (f.bounce) out.push({ flight: f, mean: [f.bounce.x, f.bounce.y], cov: f.bounce.cov, frame, color: PALETTE.muted, hasCall: false });
  }
  return out;
}

/** Mapping of a call's line identifier (e.g. "left singles sideline") to its segment, for highlighting only. */
function lineByName(name: string): Segment | null {
  const n = name.toLowerCase();
  const far = n.includes("far");
  const near = n.includes("near");
  const left = n.includes("left");
  const right = n.includes("right");
  const { baselineY: B, doublesX: D, singlesX: S, serviceY: SV } = COURT;
  if (n.includes("baseline") && (far || near)) return { a: [-D, far ? B : -B], b: [D, far ? B : -B], kind: "baseline" };
  if (n.includes("sideline") && (left || right)) {
    const x = (n.includes("doubles") ? D : S) * (left ? -1 : 1);
    return { a: [x, -B], b: [x, B], kind: "sideline" };
  }
  if (n.includes("centre") || n.includes("center")) return { a: [0, -SV], b: [0, SV], kind: "centre" };
  if (n.includes("service") && (far || near)) return { a: [-S, far ? SV : -SV], b: [S, far ? SV : -SV], kind: "service" };
  return null;
}

function niceScale(span: number): number {
  const target = span / 4;
  const steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2];
  let best = steps[0];
  for (const s of steps) if (s <= target) best = s;
  return best;
}

function MiniCourtImpl({ report, frames, currentFrame, selectedFlightId, onSelectFlight }: Props) {
  const marks = useMemo(() => bounceMarks(report.flights), [report.flights]);
  const ellipses = useMemo(
    () => new Map(marks.map((m) => [m.flight.id, polyPoints(ellipsePoints(m.mean, m.cov))])),
    [marks],
  );
  const selected = report.flights.find((f) => f.id === selectedFlightId) ?? null;
  const selMark = marks.find((m) => m.flight.id === selectedFlightId) ?? null;

  const groundTrack = useMemo(() => {
    if (!selected?.trajectory) return null;
    return selected.trajectory.map((s) => `${sx(s[1]).toFixed(1)},${sy(s[2]).toFixed(1)}`).join(" ");
  }, [selected]);

  const players = SIDES.map((side) => {
    const pf = frames?.players[side];
    const w = pf && currentFrame < pf.world.length ? pf.world[currentFrame] : null;
    const v = pf && currentFrame < pf.speed_kmh.length ? pf.speed_kmh[currentFrame] : null;
    return { side, label: report.players[side].label, pos: w, speed: v };
  });

  const onKeySelect = (e: KeyboardEvent, id: number) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelectFlight(id);
    }
  };

  return (
    <>
      <section className="block">
        <h3 className="group-title">Kuşbakışı</h3>
        <div className="group">
          <div className="court-figure">
            <svg
              className="court-svg"
              viewBox="-860 -1660 1720 3320"
              role="img"
              aria-label="Kuşbakışı kort: oyuncular ve sekmeler; uzak dip çizgi üstte"
            >
              <rect x={-860} y={-1660} width={1720} height={3320} style={{ fill: "var(--court-surround)" }} />
              <rect
                x={sx(-COURT.doublesX)}
                y={sy(COURT.baselineY)}
                width={2 * COURT.doublesX * U}
                height={2 * COURT.baselineY * U}
                style={{ fill: "var(--court-surface)" }}
              />
              <g stroke="#f7f9fc" strokeOpacity={0.92} strokeWidth={1.1} vectorEffect="non-scaling-stroke">
                {COURT_LINES.map((s, i) => (
                  <line key={i} x1={sx(s.a[0])} y1={sy(s.a[1])} x2={sx(s.b[0])} y2={sy(s.b[1])} vectorEffect="non-scaling-stroke" />
                ))}
              </g>
              <line
                x1={sx(-COURT.netHalfWidth)}
                y1={0}
                x2={sx(COURT.netHalfWidth)}
                y2={0}
                stroke="#ffffff"
                strokeOpacity={0.75}
                strokeWidth={2}
                strokeDasharray="3 2"
                vectorEffect="non-scaling-stroke"
              />
              <text x={0} y={-1560} className="court-text" textAnchor="middle">
                Uzak · {playerName(report.players.far.label)}
              </text>
              <text x={0} y={1610} className="court-text" textAnchor="middle">
                Yakın · {playerName(report.players.near.label)}
              </text>

              {groundTrack && (
                <polyline
                  points={groundTrack}
                  fill="none"
                  stroke="#ffffff"
                  strokeOpacity={0.8}
                  strokeWidth={1.2}
                  strokeDasharray="4 3"
                  vectorEffect="non-scaling-stroke"
                />
              )}

              {marks.map((m) => {
                const isSel = m.flight.id === selectedFlightId;
                const future = m.frame > currentFrame;
                const label =
                  m.hasCall && m.flight.call ? `${verdictLabel(m.flight.call.verdict)}, ${fmtProb(m.flight.call.p_in)}` : "karar yok";
                return (
                  <g
                    key={m.flight.id}
                    className="court-bounce"
                    opacity={future && !isSel ? 0.45 : 1}
                    role="button"
                    tabIndex={0}
                    aria-label={`Vuruş ${m.flight.id + 1} sekmesi: ${label}`}
                    onClick={() => onSelectFlight(m.flight.id)}
                    onKeyDown={(e) => onKeySelect(e, m.flight.id)}
                  >
                    <title>{`Vuruş ${m.flight.id + 1} · ${label}`}</title>
                    <circle cx={sx(m.mean[0])} cy={sy(m.mean[1])} r={60} fill="transparent" />
                    <polygon
                      points={ellipses.get(m.flight.id) ?? ""}
                      fill={m.color}
                      fillOpacity={0.35}
                      stroke={m.color}
                      strokeWidth={1}
                      vectorEffect="non-scaling-stroke"
                    />
                    <circle
                      cx={sx(m.mean[0])}
                      cy={sy(m.mean[1])}
                      r={isSel ? 24 : 16}
                      fill={m.color}
                      stroke="#ffffff"
                      strokeWidth={isSel ? 1.5 : 0}
                      vectorEffect="non-scaling-stroke"
                    />
                    {isSel && (
                      <circle
                        cx={sx(m.mean[0])}
                        cy={sy(m.mean[1])}
                        r={58}
                        fill="none"
                        stroke="#ffffff"
                        strokeWidth={1.6}
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                  </g>
                );
              })}

              {players.map((pl) =>
                pl.pos ? (
                  <g key={pl.side}>
                    <circle
                      cx={sx(pl.pos[0])}
                      cy={sy(pl.pos[1])}
                      r={52}
                      fill={SIDE_COLOR[pl.side]}
                      stroke="#ffffff"
                      strokeWidth={1.5}
                      vectorEffect="non-scaling-stroke"
                    />
                    <text x={sx(pl.pos[0])} y={sy(pl.pos[1]) + 24} textAnchor="middle" className="court-player-label" fill="#ffffff">
                      {pl.label.replace("P", "")}
                    </text>
                  </g>
                ) : null,
              )}
            </svg>
          </div>
        </div>
        <p className="group-foot">Sekmeler %95 güven elipsiyle çizilir; videoda henüz gelinmemiş sekmeler soluk görünür.</p>
      </section>

      <section className="block">
        <h3 className="group-title">Bu karede oyuncular</h3>
        <div className="group">
          {players.map((pl) => (
            <div className="row-stack" key={pl.side}>
              <div className="row-line">
                <span>
                  <i className="legend-swatch" style={{ "--c": SIDE_COLOR[pl.side] } as CSSProperties} aria-hidden="true" />
                  {playerName(pl.label)} · {SIDE_LABEL[pl.side]}
                </span>
                <span className="row-sub num">
                  {pl.pos && pl.speed !== null && pl.speed !== undefined ? `${fmtNum(pl.speed, 1)} km/sa` : ""}
                </span>
              </div>
              <p className="row-sub num">
                {pl.pos ? `x ${fmtNum(pl.pos[0], 1)} m · y ${fmtNum(pl.pos[1], 1)} m` : "Bu karede izlenmiyor"}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section className="block">
        <h3 className="group-title">Seçili sekme</h3>
        <div className="group">
          <CloseUp mark={selMark} />
          {selected?.bounce && (
            <div className="row">
              <span className="row-label">Konum</span>
              <span className="row-value num">
                x {fmtNum(selected.bounce.x, 2)} · y {fmtNum(selected.bounce.y, 2)} m · σ {fmtNum(selected.bounce.sd_x_cm, 1)} ×{" "}
                {fmtNum(selected.bounce.sd_y_cm, 1)} cm
              </span>
            </div>
          )}
        </div>
      </section>
    </>
  );
}

function CloseUp({ mark }: { mark: BounceMark | null }) {
  if (!mark) {
    return (
      <div className="row-stack">
        <p className="row-sub">Sekmesi olan bir vuruş seçin; %95 elipsi çizgilerle birlikte gerçek ölçekte burada görünür.</p>
      </div>
    );
  }
  const call = mark.flight.call;
  const [cx, cy] = mark.mean;
  const margin = call ? Math.abs(call.margin_m) : 0.5;
  const halfM = Math.max(0.45, Math.min(2.2, margin + 0.4));
  const half = halfM * U; // SVG units (cm)
  const vb = `${(sx(cx) - half).toFixed(2)} ${(sy(cy) - half).toFixed(2)} ${(2 * half).toFixed(2)} ${(2 * half).toFixed(2)}`;
  const hl = call ? lineByName(call.line) : null;
  const bar = niceScale(2 * halfM); // metres
  const barX = sx(cx) - half + 0.08 * half;
  const barY = sy(cy) + half - 0.1 * half;
  const k = half * 0.06;
  return (
    <figure className="closeup">
      <svg viewBox={vb} className="closeup-svg" role="img" aria-label="Seçili sekmenin gerçek ölçekte yakın görünümü">
        <rect x={sx(cx) - half} y={sy(cy) - half} width={2 * half} height={2 * half} style={{ fill: "var(--court-surface)" }} />
        <g stroke="#f7f9fc" strokeOpacity={0.9} strokeWidth={1.2} vectorEffect="non-scaling-stroke">
          {COURT_LINES.map((s, i) => (
            <line key={i} x1={sx(s.a[0])} y1={sy(s.a[1])} x2={sx(s.b[0])} y2={sy(s.b[1])} vectorEffect="non-scaling-stroke" />
          ))}
        </g>
        {hl && (
          <line
            x1={sx(hl.a[0])}
            y1={sy(hl.a[1])}
            x2={sx(hl.b[0])}
            y2={sy(hl.b[1])}
            stroke={PALETTE.ball}
            strokeWidth={2.4}
            vectorEffect="non-scaling-stroke"
          />
        )}
        <polygon
          points={polyPoints(ellipsePoints(mark.mean, mark.cov))}
          fill={mark.color}
          fillOpacity={0.3}
          stroke={mark.color}
          strokeWidth={1.4}
          vectorEffect="non-scaling-stroke"
        />
        <g stroke={mark.color} strokeWidth={1.4} vectorEffect="non-scaling-stroke">
          <line x1={sx(cx) - k} y1={sy(cy)} x2={sx(cx) + k} y2={sy(cy)} vectorEffect="non-scaling-stroke" />
          <line x1={sx(cx)} y1={sy(cy) - k} x2={sx(cx)} y2={sy(cy) + k} vectorEffect="non-scaling-stroke" />
        </g>
        <line x1={barX} y1={barY} x2={barX + bar * U} y2={barY} stroke="#ffffff" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        <text x={barX} y={barY - 0.04 * half} className="closeup-bar" style={{ fontSize: `${(0.09 * half).toFixed(2)}px` }}>
          {bar >= 1 ? `${String(bar).replace(".", ",")} m` : `${Math.round(bar * 100)} cm`}
        </text>
      </svg>
      <figcaption>
        {call ? (
          <>
            <span className="num">{fmtSignedCm(call.margin_m)}</span> · {lineLabel(call.line)}
            {hl ? " (sarı çizgi)" : ""} · çizgiler orta hatlarıyla, elips %95
          </>
        ) : (
          "Kararsız sekme · %95 elips; çizgiye mesafe hesaplanmadı"
        )}
      </figcaption>
    </figure>
  );
}

const MiniCourt = memo(MiniCourtImpl);
export default MiniCourt;
