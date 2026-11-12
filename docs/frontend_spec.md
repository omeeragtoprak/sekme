# CourtVision Review Room — frontend specification

Single-page app in `web/` (React 19 + TypeScript 5.9 + Vite 8 + three.js 0.186, already installed;
do **not** add dependencies). Backend: FastAPI on the same origin under `/api` (the dev server proxies
`/api` to `http://127.0.0.1:8765`). The production build goes to `web/dist` and FastAPI serves it at `/`.

The product is **decision support for tennis line calls**, modelled on football VAR review rooms:
the system proposes, a human confirms. It must never look more certain than it is: every call shows
P(in), the 1-sigma margin and the line; uncertain calls are `REVIEW` (amber) and invite a human decision.

## API contract

```
GET  /api/health                         -> {"status":"ok","version":"3.0.0a1"}
GET  /api/runs                           -> RunSummary[]
GET  /api/runs/{run_id}/report           -> Report (below)
GET  /api/runs/{run_id}/frames           -> Frames (below)
GET  /api/runs/{run_id}/media/{name}     -> video/mp4 with HTTP Range support; name in "source" | "classic" | "pro"
GET  /api/runs/{run_id}/reviews          -> Review[]
POST /api/runs/{run_id}/reviews          -> 201 Review   body: ReviewIn
POST /api/jobs  (multipart, field "file") -> 202 {"job_id": string}
GET  /api/jobs/{job_id}                  -> Job
```

```ts
type RunSummary = { id: string; video: string; created: string; duration_s: number; n_flights: number;
  n_calls: number; calibrated_ratio: number; media: ("source"|"classic"|"pro")[] };

type ReviewIn = { flight_id: number; decision: "confirm" | "overturn_in" | "overturn_out" | "unsure";
  reviewer: string; note: string };           // reviewer 1..64 chars, note 0..500 chars
type Review = ReviewIn & { id: string; created: string;
  system_call: { verdict: "IN"|"OUT"|"REVIEW"; p_in: number; margin_m: number; sigma_m: number; line: string } };

type Job = { id: string; status: "queued"|"running"|"done"|"error"; stage: string; progress: number;
  run_id: string | null; error: string | null };

type Report = {
  schema: "courtvision.report/1"; version: string; created: string;
  video: { file: string; width: number; height: number; fps: number; n_frames: number; duration_s: number; sha256: string };
  models: Record<"court"|"ball"|"players", { file: string; source: string; license: string; sha256: string }>;
  calibration: { valid_frames: number; total_frames: number; shot_cuts: number[]; line_support_median: number|null;
    line_rmse_px_median: number|null; focal_px_median: number|null; camera_center_m: [number,number,number]|null; keyframes: number };
  ball: { detected_frames: number; interpolated_frames: number; missing_frames: number };
  players: Record<"near"|"far", { label: "P1"|"P2"; track_ids: number[]; coverage: number; distance_m: number; peak_kmh: number; mean_kmh: number }>;
  hits: { frame: number; t: number; side: "near"|"far"; next_to_player: boolean }[];
  flights: Flight[];
  timings_s: Record<string, number>;
  limitations: string[];
  outputs?: { classic: string; pro: string };
};
type Flight = {
  id: number; start_frame: number; end_frame: number; hitter: "near"|"far"; receiver: "near"|"far";
  kind: "rally"|"serve"; bounce_frame: number|null;
  hit_speed_kmh: number|null; bounce_speed_kmh: number|null; net_clearance_m: number|null; hit_height_m: number|null;
  notes: string[];
  bounce: null | { t: number; frame: number; x: number; y: number; cov: [[number,number],[number,number]]; sd_x_cm: number; sd_y_cm: number };
  call: null | { verdict: "IN"|"OUT"|"REVIEW"; p_in: number; margin_m: number; sigma_m: number; line: string; zone: string;
                 x: number; y: number; cov: [[number,number],[number,number]] };
  trajectory: null | [t: number, x: number, y: number, z: number][];   // 60 samples, metres, world frame
  fit: null | { rms_px: number; n_obs: number; spin_in: number; spin_out: number };
};
type Frames = {
  fps: number; n: number;
  calib_valid: boolean[];
  H: (number[] | null)[];          // per frame 3x3 row-major homography WORLD(x,y,1) -> IMAGE pixels, null if not calibrated
  ball: ([number, number, 0|1|2] | null)[];   // pixel x, y, source (1 detected, 2 interpolated)
  players: Record<"near"|"far", { box: ([number,number,number,number]|null)[]; world: ([number,number]|null)[];
                                   speed_kmh: (number|null)[]; dist_m: number[] }>;
};
```

## World frame and court geometry (metres)

Right-handed, origin on the ground at the centre of the net. `x` across the court (positive = right as
seen from the camera), `y` along the court (positive = towards the FAR baseline), `z` up.
Court lines (line centres) to draw: baselines at y = ±11.835, doubles sidelines x = ±5.46,
singles sidelines x = ±4.09, service lines y = ±6.375 between x = ±4.09, centre service line x = 0
between y = ±6.375. Net: y = 0, from x = -6.399 to 6.399, height 0.914 m at centre, 1.07 m at the posts.
Project a world ground point into the video with the frame's H: `[u,v,w] = H·[x,y,1]`, pixel = `(u/w, v/w)`.

## Screens and behaviour

Dark "broadcast control room" look: near-black background (#0b0e12), panels #141922 with 1px #222a36
borders, accent cyan #2ad1ff, verdict colours IN #2fd27a, OUT #ff4d5e, REVIEW #ffb020, text #e8edf3 /
muted #8a96a8. System font stack, tabular numbers for metrics. Must work from 1280 px wide upward
(no mobile layout needed, but nothing may overflow horizontally at 1280).

1. **Top bar**: product name "CourtVision · Review Room", run selector (from `/api/runs`), an "Analyse
   video" upload button (POST /api/jobs, then poll GET /api/jobs/{id} every 1 s showing stage + progress
   bar; when done, refresh runs and select the new run), and honesty chips: calibrated frames
   `valid/total`, line fit `line_rmse_px_median px`, ball detected `%`.

2. **Video stage** (left, ~65% width): `<video>` of the selected media (tabs: Source · Classic HUD · Pro HUD).
   On top, a `<canvas>` overlay drawn **client-side** only when "Source" is selected, synchronised with
   `requestVideoFrameCallback` (fallback: `timeupdate`). Current frame = `Math.round(mediaTime * fps)`.
   Toggleable layers: court lines (from H), ball (dot) with a 10-frame trail, player boxes with
   "P1 12.3 km/h" labels, bounce markers: for each flight whose bounce.frame <= current frame, draw the
   95% ellipse of `call.cov` (k = 2.4477 sigma, sample 48 points on the ellipse in world coordinates and
   project each with H) filled with the verdict colour at 30% alpha, plus a cross at the mean.
   Canvas must scale with the displayed video size (use the video's intrinsic width/height for
   coordinates and CSS scaling for display).
   Controls under the video: play/pause, frame step back/forward (also keys ← →, space = play/pause),
   playback rate 0.25/0.5/1, current time and frame number.

3. **Timeline** under the controls: full-width bar for the clip duration; hits as small diamonds
   (P1 below the axis, P2 above), bounces as circles coloured by verdict; clicking seeks to that frame.

4. **Right column**:
   a. **Line calls list**: one card per flight (flights with no call are shown collapsed with their
      `notes` so the user sees why there is no call). Card: "Shot n · P1 → far half", verdict pill,
      a horizontal P(in) meter (0-100%), `margin +47 cm ± 2 cm (1σ) · left singles sideline`, shot speed
      (`hit_speed_kmh`), net clearance, fit rms px. Click selects the flight (seeks video to its bounce frame
      and shows it in the 3D replay). Review actions: Confirm / Overturn → IN / Overturn → OUT / Unsure,
      reviewer name (remember in localStorage), optional note; POST, then show the review history for
      that flight (from GET reviews) with timestamp and the system call snapshot. Never hide the system's
      original call after an overturn — show both.
   b. **Mini court** (SVG, top-down, far baseline at the top): court lines, players' positions at the
      current frame, all bounces with their 95% ellipses (same maths as above), selected flight highlighted.
   c. **3D replay** (three.js, OrbitControls from `three/examples/jsm/controls/OrbitControls.js`):
      court plane (blue #2a5c8a) with white lines, net (semi-transparent mesh with posts),
      selected flight trajectory as a tube or line from `trajectory` (colour by speed), bounce ellipse on
      the ground, ball sphere animated along the trajectory with a play button, camera presets
      (Broadcast: from (0,-30,12) looking at origin; Side: from (16,0,4); Top). three.js is y-up:
      map world (x, y, z) -> three (x, z, -y).
   d. **Session facts**: players distance / peak / mean speed, calibration focal & camera centre,
      model cards (source + licence), timings, and the `limitations` list verbatim under a heading
      "What this system cannot do yet".

5. Empty and error states: no runs → explain how to run `courtvision analyze <video>` or use the upload
   button; API errors shown inline, never a blank screen.

## Code organisation

`src/main.tsx`, `src/App.tsx`, `src/api.ts` (typed fetch helpers using the types above),
`src/types.ts`, `src/geometry.ts` (court lines, homography projection, ellipse sampling),
`src/components/{TopBar,VideoStage,Timeline,CallList,MiniCourt,Replay3D,SessionFacts}.tsx`,
`src/styles.css` (CSS variables for the palette). Strict TypeScript, no `any`. `npm run build` must pass.
