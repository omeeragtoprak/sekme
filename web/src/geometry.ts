// World-frame court geometry, homography projection, ellipse sampling and small display helpers.
//
// World frame (metres): right-handed, origin on the ground at the centre of the net.
//   x across the court (+ = right as seen from the camera)
//   y along the court  (+ = towards the FAR baseline)
//   z up
import type { Cov2, Flight, Verdict } from "./types";

export type Vec2 = [number, number];
export type Segment = { a: Vec2; b: Vec2; kind: "baseline" | "sideline" | "service" | "centre" };

export const COURT = {
  baselineY: 11.835,
  doublesX: 5.46,
  singlesX: 4.09,
  serviceY: 6.375,
  netHalfWidth: 6.399,
  netHeightCentre: 0.914,
  netHeightPost: 1.07,
} as const;

/** Court line centres to draw (SPEC "World frame and court geometry"). */
export const COURT_LINES: readonly Segment[] = (() => {
  const { baselineY: B, doublesX: D, singlesX: S, serviceY: SV } = COURT;
  const segs: Segment[] = [
    { a: [-D, B], b: [D, B], kind: "baseline" },
    { a: [-D, -B], b: [D, -B], kind: "baseline" },
    { a: [-D, -B], b: [-D, B], kind: "sideline" },
    { a: [D, -B], b: [D, B], kind: "sideline" },
    { a: [-S, -B], b: [-S, B], kind: "sideline" },
    { a: [S, -B], b: [S, B], kind: "sideline" },
    { a: [-S, SV], b: [S, SV], kind: "service" },
    { a: [-S, -SV], b: [S, -SV], kind: "service" },
    { a: [0, -SV], b: [0, SV], kind: "centre" },
  ];
  return segs;
})();

/** Net height (m) at world x, linear sag from the posts (1.07 m) to the centre strap (0.914 m). */
export function netHeightAt(x: number): number {
  const r = Math.min(1, Math.abs(x) / COURT.netHalfWidth);
  return COURT.netHeightCentre + (COURT.netHeightPost - COURT.netHeightCentre) * r;
}

/**
 * Project a world ground point (x, y, 0) into the image with a 3x3 row-major homography.
 * [u, v, w] = H · [x, y, 1]; pixel = (u/w, v/w). Returns null if H is invalid or w ≈ 0 / negative
 * (point behind the camera).
 */
export function projectH(H: readonly number[] | null | undefined, x: number, y: number): Vec2 | null {
  if (!H || H.length < 9) return null;
  const u = H[0] * x + H[1] * y + H[2];
  const v = H[3] * x + H[4] * y + H[5];
  const w = H[6] * x + H[7] * y + H[8];
  if (!Number.isFinite(w) || w <= 1e-9) return null;
  const px = u / w;
  const py = v / w;
  if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
  return [px, py];
}

/** k for a 95% bivariate-normal confidence ellipse: sqrt(chi2_2(0.95)) = 2.4477. */
export const K95 = 2.4477;

export type EllipseAxes = { a: number; b: number; angle: number }; // semi-axes (m) and rotation (rad) of the k-sigma ellipse

/** Eigen-decomposition of a symmetric 2x2 covariance -> k-sigma semi-axes and orientation. */
export function ellipseAxes(cov: Cov2, k: number = K95): EllipseAxes {
  const sxx = cov[0][0];
  const syy = cov[1][1];
  const sxy = 0.5 * (cov[0][1] + cov[1][0]);
  const mean = 0.5 * (sxx + syy);
  const diff = 0.5 * (sxx - syy);
  const root = Math.sqrt(diff * diff + sxy * sxy);
  const l1 = Math.max(0, mean + root);
  const l2 = Math.max(0, mean - root);
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return { a: k * Math.sqrt(l1), b: k * Math.sqrt(l2), angle };
}

/** Sample n points on the k-sigma ellipse of `cov` centred at `mean`, in world coordinates. */
export function ellipsePoints(mean: Vec2, cov: Cov2, k: number = K95, n = 48): Vec2[] {
  const { a, b, angle } = ellipseAxes(cov, k);
  const ca = Math.cos(angle);
  const sa = Math.sin(angle);
  const pts: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    const ex = a * Math.cos(th);
    const ey = b * Math.sin(th);
    pts.push([mean[0] + ca * ex - sa * ey, mean[1] + sa * ex + ca * ey]);
  }
  return pts;
}

/** World (x, y, z) -> three.js (x, z, -y); three.js is y-up. */
export function worldToThree(x: number, y: number, z: number): [number, number, number] {
  return [x, z, -y];
}

/** Video frame index for a media time: Math.round(mediaTime * fps), clamped to [0, n-1]. */
export function frameAt(mediaTime: number, fps: number, nFrames: number): number {
  const f = Math.round(mediaTime * fps);
  return Math.max(0, Math.min(Math.max(0, nFrames - 1), f));
}

/** Exact (possibly fractional) bounce frame: fitted bounce.frame if present, else the detected bounce_frame. */
export function bounceFrameRaw(f: Flight): number | null {
  if (f.bounce) return f.bounce.frame;
  if (f.bounce_frame !== null) return f.bounce_frame;
  return null;
}

/**
 * Integer video frame to seek to for a bounce: the first frame at or after the bounce instant, so that the
 * overlay rule "bounce.frame <= current frame" already shows the bounce marker there.
 */
export function bounceFrameOf(f: Flight): number | null {
  const raw = bounceFrameRaw(f);
  return raw === null ? null : Math.ceil(raw - 1e-6);
}

/** Instantaneous speeds (km/h) along a sampled trajectory, central differences. */
export function trajectorySpeedsKmh(traj: readonly (readonly [number, number, number, number])[]): number[] {
  const n = traj.length;
  const out: number[] = new Array<number>(n).fill(0);
  if (n < 2) return out;
  for (let i = 0; i < n; i++) {
    const i0 = Math.max(0, i - 1);
    const i1 = Math.min(n - 1, i + 1);
    const p0 = traj[i0];
    const p1 = traj[i1];
    const dt = p1[0] - p0[0];
    if (dt <= 0) continue;
    const d = Math.hypot(p1[1] - p0[1], p1[2] - p0[2], p1[3] - p0[3]);
    out[i] = (d / dt) * 3.6;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Display helpers. Canvas and WebGL need colour literals: these are Apple's system colours (dark
// variants, since overlays sit on video). The CSS variables in styles.css carry the light/dark pairs.

export const PALETTE = {
  accent: "#0091ff", // systemBlue
  court: "#3b6c9e",
  courtLines: "#64d2ff", // overlay court model, distinct from the painted white lines
  ball: "#ffd60a", // systemYellow
  near: "#0a84ff",
  far: "#bf5af2", // systemPurple
  in: "#30d158", // systemGreen
  out: "#ff4245", // systemRed
  review: "#ff9230", // systemOrange
  muted: "#8e8e93", // systemGray
  text: "#ffffff",
} as const;

export function verdictColor(v: Verdict | null | undefined): string {
  if (v === "IN") return PALETTE.in;
  if (v === "OUT") return PALETTE.out;
  if (v === "REVIEW") return PALETTE.review;
  return PALETTE.muted;
}

export function verdictKind(v: Verdict | null | undefined): "in" | "out" | "review" | "none" {
  return v === "IN" ? "in" : v === "OUT" ? "out" : v === "REVIEW" ? "review" : "none";
}

/** "#rrggbb" + alpha -> "rgba(r,g,b,a)". */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export const DASH = "—";
const MINUS = "\u2212";

const numFormats = new Map<number, Intl.NumberFormat>();

/** Turkish number formatting (decimal comma, typographic minus). */
export function fmtNum(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  let f = numFormats.get(digits);
  if (!f) {
    f = new Intl.NumberFormat("tr-TR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
    numFormats.set(digits, f);
  }
  return f.format(v).replace("-", MINUS);
}

/** Percentage in Turkish order ("%87,1"). */
export function fmtPct(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return DASH;
  return `%${fmtNum(fraction * 100, digits)}`;
}

/** Probability without overstating certainty: never prints %100 or %0. */
export function fmtProb(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return DASH;
  if (p >= 0.9995) return ">%99,9";
  if (p <= 0.0005) return "<%0,1";
  return fmtPct(p);
}

export function fmtSignedCm(m: number): string {
  const cm = Math.round(m * 100);
  if (cm === 0) return "0 cm";
  return `${cm > 0 ? "+" : MINUS}${Math.abs(cm)} cm`;
}

export function fmtSigmaCm(m: number): string {
  const cm = m * 100;
  if (cm < 0.5) return "<1 cm";
  return `${Math.round(cm)} cm`;
}

/** Seconds -> "mm:ss,ss" (two decimals keep consecutive 30 fps frames distinct). */
export function fmtClock(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, "0")}:${rest.toFixed(2).padStart(5, "0").replace(".", ",")}`;
}

const DATE_FMT = new Intl.DateTimeFormat("tr-TR", {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return DATE_FMT.format(d);
}
