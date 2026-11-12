// API contract types — mirrors web/SPEC.md ("API contract") exactly.

export type MediaName = "source" | "classic" | "pro";
export type Verdict = "IN" | "OUT" | "REVIEW";
export type Decision = "confirm" | "overturn_in" | "overturn_out" | "unsure";
export type Side = "near" | "far";
export type PlayerLabel = "P1" | "P2";

/** 2x2 covariance matrix in world metres². */
export type Cov2 = [[number, number], [number, number]];

/** One trajectory sample: [t (s), x, y, z] in metres, world frame. */
export type TrajectorySample = [t: number, x: number, y: number, z: number];

export type Health = { status: string; version: string };

export type RunSummary = {
  id: string;
  video: string;
  created: string;
  duration_s: number;
  n_flights: number;
  n_calls: number;
  calibrated_ratio: number;
  media: MediaName[];
};

export type ReviewIn = {
  flight_id: number;
  decision: Decision;
  reviewer: string; // 1..64 chars
  note: string; // 0..500 chars
};

export type SystemCallSnapshot = {
  verdict: Verdict;
  p_in: number;
  margin_m: number;
  sigma_m: number;
  line: string;
};

export type Review = ReviewIn & {
  id: string;
  created: string;
  system_call: SystemCallSnapshot;
};

export type LabelEval = {
  error_cm: number;
  mahalanobis2: number;
  inside_95: boolean;
  truth: "IN" | "OUT";
  truth_margin_cm: number;
  truth_line: string;
  system: Verdict;
  agrees: boolean | null;
};

export type LabelRecord = {
  id: string;
  created: string;
  flight_id: number;
  frame: number;
  px: [number, number];
  labeler: string;
  note: string;
  world_xy: [number, number];
  px_cm: number;
  eval: LabelEval;
};

export type LabelSummary =
  | { n: 0 }
  | {
      n: number;
      error_cm_median: number;
      error_cm_p90: number;
      inside_95_share: number;
      decisive: number;
      agreements: number;
      reviews: number;
      review_truth: { IN: number; OUT: number };
    };

export type JobStatus = "queued" | "running" | "done" | "error";

export type Job = {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  run_id: string | null;
  error: string | null;
};

export type ModelCard = { file: string; source: string; license: string; sha256: string };

export type PlayerSummary = {
  label: PlayerLabel;
  track_ids: number[];
  coverage: number;
  distance_m: number;
  peak_kmh: number;
  mean_kmh: number;
};

export type Hit = { frame: number; t: number; side: Side; next_to_player: boolean; serve?: boolean };

export type Bounce = {
  t: number;
  frame: number;
  x: number;
  y: number;
  cov: Cov2;
  sd_x_cm: number;
  sd_y_cm: number;
};

export type Call = {
  verdict: Verdict;
  p_in: number;
  margin_m: number;
  sigma_m: number;
  line: string;
  zone: string;
  x: number;
  y: number;
  cov: Cov2;
  /** Independent evidence that disagrees with the geometry (e.g. the receiver played an "OUT" ball). */
  conflict?: string | null;
};

export type Fit = { rms_px: number; n_obs: number; spin_in: number; spin_out: number };

export type Flight = {
  id: number;
  start_frame: number;
  end_frame: number;
  hitter: Side;
  receiver: Side;
  kind: "rally" | "serve";
  bounce_frame: number | null;
  hit_speed_kmh: number | null;
  bounce_speed_kmh: number | null;
  net_clearance_m: number | null;
  /** 1-sigma of net_clearance_m, propagated from the fit covariance. */
  net_clearance_sd_m?: number | null;
  hit_height_m: number | null;
  /** Contact frame refined by the 3D fit (fractional), null when the flight starts before the clip/segment. */
  contact_frame?: number | null;
  /** true: the receiver struck the ball after the bounce; false: the other player did; null: no further hit seen. */
  play_continued?: boolean | null;
  /** Why a fitted flight was not trusted (no call is produced). */
  rejected?: string | null;
  notes: string[];
  /** Stadium radar board speed matched to this serve (km/h), when the broadcast shows one. */
  board_speed_kmh?: number | null;
  /** Server named by the score graphic's serve marker (serves only). */
  server_name?: string | null;
  /** Players named from the score graphic and the ends-change rule, when the graphic is near in time. */
  hitter_name?: string | null;
  receiver_name?: string | null;
  /** Stroke type from the hitter's pose and the ball's side at contact. */
  stroke?: "serve" | "forehand" | "backhand" | "volley" | "overhead" | null;
  stroke_wing?: "forehand" | "backhand" | null;
  stroke_confident?: boolean;
  stroke_cue?: string;
  hand?: "right" | "left" | null;
  hand_source?: string | null;
  bounce: Bounce | null;
  call: Call | null;
  trajectory: TrajectorySample[] | null; // 60 samples, metres, world frame
  fit: Fit | null;
};

export type ServeSpeedPair = {
  flight_id: number;
  t: number;
  ours_kmh: number;
  board: number;
  board_kmh: number;
  lag_s: [number, number];
  error_pct: number;
};

export type Broadcast =
  | { ocr: null; reason: string }
  | {
      ocr: string;
      frames_read: number;
      step_frames: number;
      names: [string, string] | null;
      score_readings: number;
      score_changes: {
        t: [number, number];
        before: [number[], string | null][];
        after: [number[], string | null][];
        winner: string | null;
        n_points: number | null;
      }[];
      speed_unit: string | null;
      speed_updates: { t: [number, number]; value: number; previous: number | null }[];
      serve_speed: {
        pairs: ServeSpeedPair[];
        n: number;
        mean_error_pct: number | null;
        mean_abs_error_pct: number | null;
        unmatched_updates: { t: [number, number]; board: number; candidates: number; reason?: "cut" | "missed" | "ambiguous" }[];
      };
      server_sides: {
        serves: { flight_id: number; server: string; side: Side; end_changes: number; start_side: Side }[];
        consistent: boolean | null;
        start_end: Record<string, Side> | null;
      };
    };

export type Report = {
  schema: "courtvision.report/1";
  version: string;
  created: string;
  video: {
    file: string;
    width: number;
    height: number;
    fps: number;
    n_frames: number;
    duration_s: number;
    sha256: string;
  };
  models: Record<"court" | "ball" | "players", ModelCard> & { pose?: ModelCard };
  strokes?: { model: string; poses: number; requested: number; hands: Record<string, "right" | "left"> };
  calibration: {
    valid_frames: number;
    total_frames: number;
    shot_cuts: number[];
    line_support_median: number | null;
    line_rmse_px_median: number | null;
    focal_px_median: number | null;
    camera_center_m: [number, number, number] | null;
    keyframes: number;
  };
  ball: { detected_frames: number; interpolated_frames: number; missing_frames: number };
  players: Record<Side, PlayerSummary>;
  hits: Hit[];
  flights: Flight[];
  timings_s: Record<string, number>;
  limitations: string[];
  broadcast?: Broadcast;
  outputs?: { classic: string; pro: string };
};

/** Ball pixel observation: x, y, source (0 none, 1 detected, 2 interpolated). */
export type BallObs = [number, number, 0 | 1 | 2];

/** Player box in image pixels: x1, y1, x2, y2. */
export type PixelBox = [number, number, number, number];

export type PlayerFrames = {
  box: (PixelBox | null)[];
  world: ([number, number] | null)[];
  speed_kmh: (number | null)[];
  dist_m: number[];
};

export type Frames = {
  fps: number;
  n: number;
  calib_valid: boolean[];
  /** Per frame 3x3 row-major homography WORLD(x,y,1) -> IMAGE pixels; null if not calibrated. */
  H: (number[] | null)[];
  ball: (BallObs | null)[];
  players: Record<Side, PlayerFrames>;
};

/** Generic async resource state used across the app (never render a blank screen). */
export type Async<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: string };
