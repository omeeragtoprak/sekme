import { memo, useCallback, useEffect, useRef, useState } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  CatmullRomCurve3,
  CircleGeometry,
  Color,
  CylinderGeometry,
  DirectionalLight,
  DoubleSide,
  Group,
  HemisphereLight,
  Line,
  LineBasicMaterial,
  LineLoop,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  PlaneGeometry,
  RepeatWrapping,
  Scene,
  SphereGeometry,
  TubeGeometry,
  Vector3,
  WebGLRenderer,
  type Material,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  COURT,
  COURT_LINES,
  ellipsePoints,
  fmtNum,
  fmtProb,
  fmtSigmaCm,
  fmtSignedCm,
  netHeightAt,
  PALETTE,
  trajectorySpeedsKmh,
  verdictColor,
  worldToThree,
} from "../geometry";
import { verdictLabel } from "../i18n";
import { PauseIcon, PlayIcon, Spinner } from "../icons";
import type { Flight, TrajectorySample } from "../types";

type Props = { flight: Flight | null; shotLabel: string | null };

type Preset = "broadcast" | "side" | "top";
/** Camera positions in WORLD coordinates (x, y, z); all look at the origin. */
const PRESETS: { key: Preset; label: string; world: [number, number, number] }[] = [
  { key: "broadcast", label: "Yayın", world: [0, -30, 12] },
  { key: "side", label: "Yan", world: [16, 0, 4] },
  { key: "top", label: "Üst", world: [0, -0.01, 52] },
];

const RATES = [0.25, 0.5, 1] as const;

/** Scene background follows the page theme (light / dark), read from the --scene CSS variable. */
function sceneColor(): Color {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--scene").trim();
  try {
    return new Color(v || "#17181b");
  } catch {
    return new Color("#17181b");
  }
}
const BALL_RADIUS = 0.11; // drawn ~3.3x the real 3.3 cm radius so it is visible from the broadcast view
const TUBE_RADIUS = 0.045;

/** Speed ramp (km/h -> sRGB), deliberately distinct from the IN/OUT/REVIEW verdict colours. */
const SPEED_STOPS: { v: number; c: string }[] = [
  { v: 40, c: "#3a7bff" },
  { v: 80, c: "#2ad1ff" },
  { v: 120, c: "#f2f6fa" },
  { v: 160, c: "#ff4fd8" },
];
const SPEED_GRADIENT = `linear-gradient(90deg, ${SPEED_STOPS.map((s) => s.c).join(", ")})`;

function speedColor(kmh: number, out: Color): Color {
  const stops = SPEED_STOPS;
  if (kmh <= stops[0].v) return out.set(stops[0].c);
  for (let i = 1; i < stops.length; i++) {
    if (kmh <= stops[i].v) {
      const a = stops[i - 1];
      const b = stops[i];
      const t = (kmh - a.v) / (b.v - a.v);
      return out.set(a.c).lerp(new Color(b.c), t);
    }
  }
  return out.set(stops[stops.length - 1].c);
}

const v3 = (x: number, y: number, z: number) => {
  const [a, b, c] = worldToThree(x, y, z);
  return new Vector3(a, b, c);
};

function disposeObject(root: Object3D) {
  root.traverse((o) => {
    if (o instanceof Mesh || o instanceof Line) {
      (o.geometry as BufferGeometry).dispose();
      const mats: Material[] = Array.isArray(o.material) ? (o.material as Material[]) : [o.material as Material];
      for (const m of mats) {
        if ((m instanceof MeshBasicMaterial || m instanceof MeshStandardMaterial || m instanceof MeshLambertMaterial) && m.map) {
          m.map.dispose();
        }
        m.dispose();
      }
    }
  });
}

/** Flat quads for the court lines (line centres from geometry.ts), slightly above the ground. */
function buildLines(): Mesh {
  const pos: number[] = [];
  const idx: number[] = [];
  for (const seg of COURT_LINES) {
    const w = seg.kind === "baseline" ? 0.1 : 0.05;
    const dx = seg.b[0] - seg.a[0];
    const dy = seg.b[1] - seg.a[1];
    const len = Math.hypot(dx, dy);
    if (len === 0) continue;
    const nx = (-dy / len) * (w / 2);
    const ny = (dx / len) * (w / 2);
    const base = pos.length / 3;
    const corners: [number, number][] = [
      [seg.a[0] + nx, seg.a[1] + ny],
      [seg.a[0] - nx, seg.a[1] - ny],
      [seg.b[0] - nx, seg.b[1] - ny],
      [seg.b[0] + nx, seg.b[1] + ny],
    ];
    for (const [x, y] of corners) pos.push(...worldToThree(x, y, 0.004));
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(new Float32Array(pos), 3));
  g.setIndex(idx);
  return new Mesh(g, new MeshBasicMaterial({ color: 0xf4f7fb, side: DoubleSide }));
}

function makeNetTexture(): CanvasTexture | null {
  const c = document.createElement("canvas");
  c.width = 32;
  c.height = 32;
  const g = c.getContext("2d");
  if (!g) return null;
  g.clearRect(0, 0, 32, 32);
  g.strokeStyle = "rgba(235,240,246,1)";
  g.lineWidth = 2;
  g.strokeRect(1, 1, 30, 30);
  const tex = new CanvasTexture(c);
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  tex.repeat.set(96, 7);
  tex.anisotropy = 4;
  return tex;
}

function buildCourt(): Group {
  const g = new Group();
  // Surround and court surface
  const apron = new Mesh(new PlaneGeometry(22, 40), new MeshBasicMaterial({ color: 0x2f5a84 }));
  apron.rotation.x = -Math.PI / 2;
  apron.position.y = -0.02;
  g.add(apron);
  const court = new Mesh(
    new PlaneGeometry(2 * COURT.doublesX + 3.2, 2 * COURT.baselineY + 6.4),
    new MeshBasicMaterial({ color: new Color(PALETTE.court) }),
  );
  court.rotation.x = -Math.PI / 2;
  g.add(court);
  g.add(buildLines());

  // Net: sagging mesh between the posts (1.07 m) and the centre strap (0.914 m)
  const N = 32;
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i <= N; i++) {
    const x = -COURT.netHalfWidth + (2 * COURT.netHalfWidth * i) / N;
    const h = netHeightAt(x);
    pos.push(...worldToThree(x, 0, 0), ...worldToThree(x, 0, h));
    uv.push(i / N, 0, i / N, h / COURT.netHeightPost);
    if (i < N) {
      const a = 2 * i;
      idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const netGeom = new BufferGeometry();
  netGeom.setAttribute("position", new BufferAttribute(new Float32Array(pos), 3));
  netGeom.setAttribute("uv", new BufferAttribute(new Float32Array(uv), 2));
  netGeom.setIndex(idx);
  const tex = makeNetTexture();
  const net = new Mesh(
    netGeom,
    new MeshBasicMaterial({
      color: 0xdfe6ee,
      map: tex,
      transparent: true,
      opacity: tex ? 0.55 : 0.22,
      side: DoubleSide,
      depthWrite: false,
    }),
  );
  g.add(net);
  // Net tape
  const tapePts: Vector3[] = [];
  for (let i = 0; i <= N; i++) {
    const x = -COURT.netHalfWidth + (2 * COURT.netHalfWidth * i) / N;
    tapePts.push(v3(x, 0, netHeightAt(x)));
  }
  g.add(new Line(new BufferGeometry().setFromPoints(tapePts), new LineBasicMaterial({ color: 0xffffff })));
  // Posts
  for (const sgn of [-1, 1]) {
    const post = new Mesh(
      new CylinderGeometry(0.045, 0.045, COURT.netHeightPost, 16),
      new MeshStandardMaterial({ color: 0x3a4250, roughness: 0.6, metalness: 0.3 }),
    );
    post.position.copy(v3(sgn * COURT.netHalfWidth, 0, COURT.netHeightPost / 2));
    g.add(post);
  }
  return g;
}

type FlightBuild = { group: Group; traj: TrajectorySample[] | null; speeds: number[] };

function buildFlight(flight: Flight | null): FlightBuild {
  const group = new Group();
  const traj = flight?.trajectory && flight.trajectory.length >= 2 ? flight.trajectory : null;
  let speeds: number[] = [];
  if (traj) {
    speeds = trajectorySpeedsKmh(traj);
    const pts = traj.map((s) => v3(s[1], s[2], s[3]));
    const curve = new CatmullRomCurve3(pts, false, "centripetal");
    const tubular = Math.max(64, traj.length * 4);
    const radial = 8;
    const geom = new TubeGeometry(curve, tubular, TUBE_RADIUS, radial, false);
    // Colour by speed: map each ring (arc-length parameter) to the sample speeds by cumulative length.
    const cum: number[] = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
    const total = cum[cum.length - 1] || 1;
    const colors = new Float32Array(geom.getAttribute("position").count * 3);
    const col = new Color();
    let k = 0;
    for (let i = 0; i <= tubular; i++) {
      const s = (i / tubular) * total;
      while (k < cum.length - 2 && cum[k + 1] < s) k++;
      const span = cum[k + 1] - cum[k];
      const t = span > 0 ? Math.min(1, Math.max(0, (s - cum[k]) / span)) : 0;
      speedColor(speeds[k] + (speeds[k + 1] - speeds[k]) * t, col);
      for (let j = 0; j <= radial; j++) {
        const o = (i * (radial + 1) + j) * 3;
        if (o + 2 < colors.length) {
          colors[o] = col.r;
          colors[o + 1] = col.g;
          colors[o + 2] = col.b;
        }
      }
    }
    geom.setAttribute("color", new BufferAttribute(colors, 3));
    group.add(new Mesh(geom, new MeshLambertMaterial({ vertexColors: true })));

    // Ground track (shadow line) of the trajectory
    const track = traj.filter((s) => s[3] >= -0.05).map((s) => v3(s[1], s[2], 0.007));
    if (track.length >= 2) {
      group.add(
        new Line(
          new BufferGeometry().setFromPoints(track),
          new LineBasicMaterial({ color: 0x0b0e12, transparent: true, opacity: 0.45 }),
        ),
      );
    }
  }

  // Bounce: 95% ellipse on the ground + a pin, in the verdict colour
  const mean = flight?.call ? ([flight.call.x, flight.call.y] as [number, number]) : flight?.bounce ? ([flight.bounce.x, flight.bounce.y] as [number, number]) : null;
  const cov = flight?.call?.cov ?? flight?.bounce?.cov ?? null;
  if (mean && cov) {
    const color = new Color(flight?.call ? verdictColor(flight.call.verdict) : PALETTE.muted);
    const ring = ellipsePoints(mean, cov).map((p) => v3(p[0], p[1], 0.009));
    const center = v3(mean[0], mean[1], 0.009);
    const fanPos: number[] = [center.x, center.y, center.z];
    for (const p of ring) fanPos.push(p.x, p.y, p.z);
    const fanIdx: number[] = [];
    for (let i = 0; i < ring.length; i++) fanIdx.push(0, 1 + i, 1 + ((i + 1) % ring.length));
    const fan = new BufferGeometry();
    fan.setAttribute("position", new BufferAttribute(new Float32Array(fanPos), 3));
    fan.setIndex(fanIdx);
    group.add(
      new Mesh(fan, new MeshBasicMaterial({ color, transparent: true, opacity: 0.55, side: DoubleSide, depthWrite: false })),
    );
    group.add(new LineLoop(new BufferGeometry().setFromPoints(ring), new LineBasicMaterial({ color })));
    group.add(
      new Line(
        new BufferGeometry().setFromPoints([v3(mean[0], mean[1], 0.01), v3(mean[0], mean[1], 0.7)]),
        new LineBasicMaterial({ color }),
      ),
    );
    const cap = new Mesh(new SphereGeometry(0.035, 12, 8), new MeshBasicMaterial({ color }));
    cap.position.copy(v3(mean[0], mean[1], 0.7));
    group.add(cap);
  }
  return { group, traj, speeds };
}

type Ctx = {
  renderer: WebGLRenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  controls: OrbitControls;
  ball: Mesh;
  shadow: Mesh;
  dirty: boolean;
  anim: { traj: TrajectorySample[] | null; speeds: number[]; t: number; T: number; playing: boolean; rate: number };
  tween: { fromPos: Vector3; toPos: Vector3; fromTarget: Vector3; start: number; dur: number } | null;
};

function sampleAt(traj: TrajectorySample[], speeds: number[], tRel: number): { p: Vector3; z: number; v: number; t: number } {
  const t = traj[0][0] + tRel;
  let i = 0;
  while (i < traj.length - 2 && traj[i + 1][0] < t) i++;
  const a = traj[i];
  const b = traj[i + 1];
  const span = b[0] - a[0];
  const u = span > 0 ? Math.min(1, Math.max(0, (t - a[0]) / span)) : 0;
  const x = a[1] + (b[1] - a[1]) * u;
  const y = a[2] + (b[2] - a[2]) * u;
  const z = a[3] + (b[3] - a[3]) * u;
  return { p: v3(x, y, z), z, v: speeds[i] + (speeds[i + 1] - speeds[i]) * u, t };
}

function Replay3DImpl({ flight, shotLabel }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<Ctx | null>(null);
  const sliderRef = useRef<HTMLInputElement>(null);
  const readTRef = useRef<HTMLSpanElement>(null);
  const readVRef = useRef<HTMLSpanElement>(null);
  const readZRef = useRef<HTMLSpanElement>(null);
  const [glError, setGlError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [preset, setPreset] = useState<Preset>("broadcast");
  const [rate, setRate] = useState<number>(0.5);

  /** Move the ball to the current animation time and update the DOM readouts (no React re-render). */
  const syncBall = useCallback((ctx: Ctx) => {
    const { traj, speeds, t, T } = ctx.anim;
    if (!traj) {
      ctx.ball.visible = false;
      ctx.shadow.visible = false;
      if (readTRef.current) readTRef.current.textContent = "—";
      if (readVRef.current) readVRef.current.textContent = "—";
      if (readZRef.current) readZRef.current.textContent = "—";
      return;
    }
    const s = sampleAt(traj, speeds, t);
    ctx.ball.visible = true;
    ctx.shadow.visible = s.z > -0.05;
    ctx.ball.position.copy(s.p);
    ctx.shadow.position.set(s.p.x, 0.006, s.p.z);
    const u = T > 0 ? t / T : 0;
    if (sliderRef.current) {
      sliderRef.current.value = String(u);
      sliderRef.current.style.setProperty("--p", `${(u * 100).toFixed(1)}%`);
    }
    if (readTRef.current) readTRef.current.textContent = `${fmtNum(s.t, 2)} sn`;
    if (readVRef.current) readVRef.current.textContent = `${fmtNum(s.v, 0)} km/sa`;
    if (readZRef.current) readZRef.current.textContent = `${fmtNum(s.z, 2)} m`;
    ctx.dirty = true;
  }, []);

  // Mount: renderer, scene, camera, controls, render loop, resize observer.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    } catch (err) {
      setGlError(err instanceof Error ? err.message : "Bu tarayıcıda WebGL kullanılamıyor.");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(sceneColor(), 1);
    renderer.domElement.className = "replay-gl";
    host.appendChild(renderer.domElement);

    const scene = new Scene();
    scene.add(new HemisphereLight(0xdfe9ff, 0x1a2230, 1.6));
    const sun = new DirectionalLight(0xffffff, 1.4);
    sun.position.set(-8, 20, 10);
    scene.add(sun);
    scene.add(buildCourt());

    const ball = new Mesh(
      new SphereGeometry(BALL_RADIUS, 24, 16),
      new MeshStandardMaterial({ color: 0xe4ff3c, emissive: 0x4a5a00, roughness: 0.55 }),
    );
    ball.visible = false;
    scene.add(ball);
    const shadow = new Mesh(
      new CircleGeometry(BALL_RADIUS * 1.1, 24),
      new MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.visible = false;
    scene.add(shadow);

    const camera = new PerspectiveCamera(30, 16 / 10, 0.5, 400);
    camera.position.copy(v3(...PRESETS[0].world));
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.minDistance = 2;
    controls.maxDistance = 90;
    controls.maxPolarAngle = Math.PI / 2 - 0.03;
    controls.enableZoom = false; // enabled after the user clicks into the view, so page scrolling is never hijacked
    controls.target.set(0, 0, 0);
    controls.update();

    const ctx: Ctx = {
      renderer,
      scene,
      camera,
      controls,
      ball,
      shadow,
      dirty: true,
      anim: { traj: null, speeds: [], t: 0, T: 0, playing: false, rate: 0.5 },
      tween: null,
    };
    ctxRef.current = ctx;

    const onChange = () => {
      ctx.dirty = true;
    };
    controls.addEventListener("change", onChange);
    const el = renderer.domElement;
    const enableZoom = () => {
      controls.enableZoom = true;
    };
    const disableZoom = () => {
      controls.enableZoom = false;
    };
    el.addEventListener("pointerdown", enableZoom);
    el.addEventListener("pointerleave", disableZoom);

    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (!r || r.width < 2 || r.height < 2) return;
      renderer.setSize(r.width, r.height, false);
      camera.aspect = r.width / r.height;
      camera.updateProjectionMatrix();
      ctx.dirty = true;
    });
    ro.observe(host);

    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    const onScheme = () => {
      renderer.setClearColor(sceneColor(), 1);
      ctx.dirty = true;
    };
    scheme.addEventListener("change", onScheme);

    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (ctx.tween) {
        const tw = ctx.tween;
        const u = Math.min(1, (now - tw.start) / tw.dur);
        const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        camera.position.lerpVectors(tw.fromPos, tw.toPos, e);
        controls.target.lerpVectors(tw.fromTarget, new Vector3(0, 0, 0), e);
        camera.lookAt(controls.target);
        if (u >= 1) ctx.tween = null;
        ctx.dirty = true;
      }
      const a = ctx.anim;
      if (a.playing && a.traj) {
        a.t += dt * a.rate;
        if (a.t >= a.T) {
          a.t = a.T;
          a.playing = false;
          setPlaying(false);
        }
        syncBall(ctx);
      }
      if (controls.update()) ctx.dirty = true;
      if (ctx.dirty) {
        renderer.render(scene, camera);
        ctx.dirty = false;
      }
    };
    raf = requestAnimationFrame(loop);
    setReady(true);

    return () => {
      cancelAnimationFrame(raf);
      scheme.removeEventListener("change", onScheme);
      ro.disconnect();
      controls.removeEventListener("change", onChange);
      el.removeEventListener("pointerdown", enableZoom);
      el.removeEventListener("pointerleave", disableZoom);
      controls.dispose();
      disposeObject(scene);
      scene.clear();
      renderer.dispose();
      renderer.forceContextLoss();
      el.remove();
      ctxRef.current = null;
      setReady(false);
    };
  }, [syncBall]);

  // Rebuild the flight-specific objects whenever the selected flight changes.
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ready || !ctx) return;
    const { group, traj, speeds } = buildFlight(flight);
    ctx.scene.add(group);
    ctx.anim = {
      traj,
      speeds,
      t: 0,
      T: traj ? Math.max(0, traj[traj.length - 1][0] - traj[0][0]) : 0,
      playing: false,
      rate: ctx.anim.rate,
    };
    setPlaying(false);
    syncBall(ctx);
    ctx.dirty = true;
    return () => {
      ctx.scene.remove(group);
      disposeObject(group);
      ctx.dirty = true;
    };
  }, [flight, ready, syncBall]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (ctx) ctx.anim.rate = rate;
  }, [rate, ready]);

  const togglePlay = () => {
    const ctx = ctxRef.current;
    if (!ctx || !ctx.anim.traj) return;
    const a = ctx.anim;
    if (a.playing) {
      a.playing = false;
      setPlaying(false);
    } else {
      if (a.t >= a.T) a.t = 0;
      a.playing = true;
      setPlaying(true);
    }
    ctx.dirty = true;
  };

  const goPreset = (p: Preset) => {
    setPreset(p);
    const ctx = ctxRef.current;
    const def = PRESETS.find((x) => x.key === p);
    if (!ctx || !def) return;
    ctx.tween = {
      fromPos: ctx.camera.position.clone(),
      toPos: v3(...def.world),
      fromTarget: ctx.controls.target.clone(),
      start: performance.now(),
      dur: 650,
    };
  };

  const onScrub = (value: number) => {
    const ctx = ctxRef.current;
    if (!ctx || !ctx.anim.traj) return;
    ctx.anim.playing = false;
    setPlaying(false);
    ctx.anim.t = Math.max(0, Math.min(1, value)) * ctx.anim.T;
    syncBall(ctx);
  };

  const hasTraj = !!flight?.trajectory && flight.trajectory.length >= 2;
  const call = flight?.call ?? null;

  return (
    <section className="section" aria-labelledby="replay-title">
      <div className="section-head">
        <div>
          <h2 id="replay-title" className="section-title">
            3B Tekrar
          </h2>
          <p className="section-meta">{shotLabel ?? "Vuruş seçilmedi"}</p>
        </div>
        <div className="segmented segmented--sm" role="group" aria-label="Kamera açısı">
          {PRESETS.map((p) => (
            <button key={p.key} type="button" aria-pressed={preset === p.key} onClick={() => goPreset(p.key)}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="replay">
        <div ref={hostRef} className="replay-host" aria-label="Seçili vuruşun etkileşimli 3B görünümü" role="img" />
        {!ready && !glError && (
          <div className="replay-cover">
            <Spinner />
          </div>
        )}
        {glError && (
          <div className="replay-cover" role="alert">
            <p className="empty-title">3B görünüm açılamadı</p>
            <p className="empty-text">{glError}</p>
          </div>
        )}
        {!glError && !flight && (
          <div className="replay-cover">
            <p className="empty-title">Vuruş seçilmedi</p>
            <p className="empty-text">Uydurulan 3B yörüngeyi izlemek için listeden ya da zaman çizelgesinden bir vuruş seçin.</p>
          </div>
        )}
        {!glError && flight && !hasTraj && (
          <div className="replay-cover">
            <p className="empty-title">Bu vuruş için 3B uydurma yok</p>
            <p className="empty-text">{flight.notes[0] ?? "Analiz bu uçuş için bir yörünge üretmedi."}</p>
          </div>
        )}
        {hasTraj && (
          <div className="replay-legend glass-clear" aria-hidden="true">
            <span className="num">40</span>
            <span className="replay-ramp" style={{ background: SPEED_GRADIENT }} />
            <span className="num">160 km/sa</span>
          </div>
        )}
      </div>

      <div className="replay-bar">
        <button
          type="button"
          className="round-btn"
          onClick={togglePlay}
          disabled={!hasTraj || !!glError}
          aria-label={playing ? "Tekrarı duraklat" : "Tekrarı oynat"}
        >
          {playing ? <PauseIcon size={15} /> : <PlayIcon size={15} />}
        </button>
        <input
          ref={sliderRef}
          className="slider"
          type="range"
          min={0}
          max={1}
          step={0.001}
          defaultValue={0}
          disabled={!hasTraj || !!glError}
          aria-label="Tekrar konumu"
          onChange={(e) => onScrub(Number(e.target.value))}
        />
        <div className="segmented segmented--sm" role="group" aria-label="Tekrar hızı">
          {RATES.map((r) => (
            <button key={r} type="button" className="num" aria-pressed={rate === r} onClick={() => setRate(r)}>
              {String(r).replace(".", ",")}×
            </button>
          ))}
        </div>
      </div>
      <dl className="facts-inline num">
        <div>
          <dt>Zaman</dt>
          <dd>
            <span ref={readTRef}>—</span>
          </dd>
        </div>
        <div>
          <dt>Hız</dt>
          <dd>
            <span ref={readVRef}>—</span>
          </dd>
        </div>
        <div>
          <dt>Yükseklik</dt>
          <dd>
            <span ref={readZRef}>—</span>
          </dd>
        </div>
        {call && (
          <div>
            <dt>Karar</dt>
            <dd>
              {verdictLabel(call.verdict)} · {fmtProb(call.p_in)} · {fmtSignedCm(call.margin_m)} ± {fmtSigmaCm(call.sigma_m)}
            </dd>
          </div>
        )}
      </dl>
      <p className="foot-note">
        {flight?.fit ? `Yörünge ${flight.fit.n_obs} top gözlemine uyduruldu · hata ${fmtNum(flight.fit.rms_px, 2)} px. ` : ""}
        Top görünür olsun diye büyütülerek çizilir · sürükleyerek döndürün · tıklayıp kaydırarak yakınlaştırın.
      </p>
    </section>
  );
}

const Replay3D = memo(Replay3DImpl);
export default Replay3D;
