# Serve speed: pinning the fit to the measured contact height, on unused broadcast footage

Plan written on 2026-11-14, **before any test footage was downloaded or seen** (amended the same day
after an independent review, still before any download; see the end). Nothing below has been run on
test data. The download needs the user's explicit yes (file, source, size).

## Why (development data only)

- A serve's fitted speed is compared with the stadium radar board (`results/radar.txt`). On the US Open
  2025 final, 150–704 s (`docs/EVAL_SERVE_CONTACT.md`; development data since that test), the serves the
  toss-bridge candidate finds give 18 fitted serves with a board read by eye: mean |error| 10.9 %, with
  slow second serves overestimated and others underestimated.
- The fitted speed is almost exactly 1.19 × the straight-line average speed from contact to bounce on
  every serve, so the error is in the geometry the fit settles on. On the 15 near-side serves it tracks
  how far the fitted contact height drifts from the contact height measured on the camera ray above the
  server: correlation 0.96, about +26 % speed per metre of drift (drifts of −0.6 to +0.9 m). The fit is
  meant to be held at that height, but only softly: a ±0.3 m band with a 0.1 m hinge, which dozens of
  pixel residuals outweigh.
- **Candidate** (`sekme analyze --pin-serve-height`, `flights.SERVE_PIN_TOL_M = 0.05`,
  `SERVE_PIN_SCALE_M = 0.01`; off by default): the band is ±0.05 m and 1 cm outside it counts as one
  sigma, so the fitted contact stays at the measured height. Development results, ball v4:

| Development check | Current code | Pinned |
|---|---|---|
| 2025 final 150–704 s with the toss bridge, 18 serves, board read by eye | 10.9 % | 4.7 % |
| the same, 15 near-side serves | 10.8 % | 3.9 % |
| 2025 final 150–704 s, default detector, 5 serves, board read by eye | 9.6 % | 6.0 % |
| 5 radar clips (`results/radar.txt`), 5 automatic pairs | 6.0 % (SD 6.4) | 3.9 % (SD 3.2) |
| call verdicts on six clips (366 calls) | — | none changed; bounces move ≤ 3.6 cm |

  (Mean |error| of the fitted serve speed against the board.) A first try with only the hinge stiffened
  (band still ±0.3 m) gave 8.2 % on the 18 serves: the fit sat on the band's edge.
- **Risk:** the pin is only as good as the measured height (the image contact point on the camera ray
  above the server's feet). A simulated serve pinned 0.45 m too low comes out 5 % slow
  (`tests/test_physics_linecall.py`). The three far-side serves of the development clip changed little
  (+4.8 → +4.8 %, +12.6 → +7.5 %, +15.6 → +13.8 %).

## Data

- **Footage, fixed now: six matches**, each by the official US Open YouTube channel's highlight video of
  that match (the upload titled as its highlights; the longest if there are several), 1080p where
  offered, else 720p, downloaded with `.venv/bin/yt-dlp` after the user's yes:
  1. 2024 women's final, Sabalenka–Pegula;
  2. 2024 men's semi-final, Sinner–Draper;
  3. 2024 men's semi-final, Fritz–Tiafoe;
  4. 2025 women's final, Sabalenka–Anisimova;
  5. 2025 men's semi-final, Alcaraz–Djokovic;
  6. 2025 men's semi-final, Sinner–Auger-Aliassime.

  None of them is footage the project has used (2023 men's final, 2024 men's final Sinner–Fritz, 2025
  men's final, 2025 QF Djokovic–Fritz). Each file's video id and SHA-256 are written here right after
  the download, before any labelling or analysis. Kept local, research only.
- **Labels, blind**, as in `docs/EVAL_SERVE_CONTACT.md`: a separate labeller (a fresh agent session) with
  no access to any system output for these videos marks every serve shown in real time from the main
  camera: contact bracket, server side, and the stadium board's serve speed read by eye (the first new
  value after the contact; none if it does not change or cannot be read). The labeller's start and end
  times and the first analysis's start time are recorded; labels are committed before any scoring.

## Runs

Ball v4. Perception once per video (one GPU job at a time), then, from that one cache: the default
analysis, `--fps-scaled-thresholds` (the second candidate, below), the same plus `--pin-serve-height`, and,
reported only, the toss bridge. *(Amended before any download, after an independent review: the pin was
first to be judged on the default detector, which at 60 fps finds about a quarter of the serves; it is now
judged on the runs with scaled thresholds, with and without the pin, so it has serves to be judged on.
Its effect, the fit's contact height, does not depend on which serves are found.)* Both rules and the code
commit are recorded here before the first scoring, and both candidates are scored in one pass.

## Measurement

- Serves are matched to labels **in the unpinned run** as in `scripts/eval_serve_contact.py` (same side,
  contact estimate within 30 frames of the bracket, one-to-one, closest first); the pinned run's serve
  is the flight starting at the same hit.
- **Paired set:** matched serves with a board reading and a fitted, not rejected, speed in the unpinned
  run. If the pinned run rejects one of them or gives it no speed, that serve counts as **worse** in
  criterion 2 and is listed (and is left out of criterion 1's means, which are over the serves both
  runs measure). Error = 100 × (fitted − board) / board.

## Rule (decided once, on the six videos)

- **Inconclusive** if the paired set has fewer than 15 serves: the candidate stays off by default, its
  code stays, and these videos count as used for it (a later test adds new videos only).
- Otherwise the candidate becomes the default only if all hold:
  1. the paired **mean |error| falls by at least 1.5 points**;
  2. of the paired serves whose |error| changes by 0.5 point or more, **at least two thirds get closer**
     to the board (changes under 0.5 point are ties and are not counted);
  3. **no call verdict changes** (IN, OUT or REVIEW) on any serve flight of the six videos; changes on
     other flights should not occur (the pin touches only the serve anchor) and are reported.

  If any fails, it is rejected: its code is removed and the result stays here. Reported, not used: mean
  signed error, near and far side separately, by board speed (< 100 mph, ≥ 115 mph), the share within
  ±5 %, the automatic board pairs, the change in the bounce's standard deviations and in P(in), and the
  same with the toss bridge.

## Limits

- Six highlight videos of one tournament and one broadcaster; the board itself is a measurement with
  its own error (radar, rounded to 1 mph); highlight edits cut some serves' board changes.
- The measured contact height has its own error: in a simulated 190 km/h serve at 60 fps, measuring on
  the frame after the contact gives 3.14 m for a 2.95 m contact (+0.2 m), and the contact is assumed
  above the feet although it is usually in front of them. ±0.05 m is tighter than that error; the pin trades it for the fit's larger drift.
  A pinned fit also reports a smaller bounce covariance, so P(in) can be overconfident when the measured
  height is wrong; rule 3 and the reported σ change are there for that.
- The plausibility gate checks the measured contact height, not the fitted one (in both runs).
- If adopted, the pin also applies to serves the toss bridge would add; those are only reported here.

## Amendments before any download (2026-11-14, after an independent review)

The six matches named (the first version said "the 2024 and 2025 men's semi-finals" and "the highlight
video", which left choices open); video ids and hashes to be recorded at download; an explicit
inconclusive outcome; the paired set taken from the unpinned run, a pinned rejection counting as worse;
the majority criterion raised to two thirds with ties defined; every verdict change on a serve flight,
REVIEW included, fails the rule; the limits on the measured height and the bounce covariance added; a
test that the flag sets the serve anchor's band (`tests/test_events_gates.py`).

## Second candidate on the same videos: frame counts scaled at 60 fps (rule written 2026-11-14, before any download)

**Change** (`sekme analyze --fps-scaled-thresholds`; off by default): three frame counts that were set on
30 fps video are scaled by s = round(fps / 30), the factor the ball network's input stride already uses
(s = 1 at 25–30 fps, so nothing changes there; s = 2 at 50–60 fps): the court-fixed filter's window and
count (`drop_court_fixed`, ±15 and 10 frames → ±30 and 20), the toss run's gap (2 → 4 frames) and the toss
speed limit per frame (÷ 2).

**Why** (development data only, `docs/EVAL_SPOT.md`, end of candidate 1): on the US Open 2025 final,
150–704 s, the default pipeline finds 6 of 23 labelled serves at 60 fps and 18 at 30 fps (every second
frame). The ball network does detect the ball at the toss apex at 60 fps; the court-fixed filter removes
it, because a ball hanging for 0.17 s fills 10 frames. With the change: 20 of 23 at 60 fps (19 within 2
frames). On the four 60 fps development clips it also keeps 3–5 % more ball detections and changes some
calls (one IN → REVIEW, two OUT counts, a few bounces moved), which nothing local can check directly.
Indirectly: the same 554 s at 30 fps (every second frame, where these counts are what they were set for)
gives 121 calls; the default 60 fps run finds 88 of those bounces (73 %), the scaled one 103 (85 %), with
the same verdict on 88 / 88 and 102 / 103, no decisive IN/OUT flip in either, positions closer (90th
percentile 13.0 → 11.5 cm) and fewer calls the 30 fps run does not make (35 → 29). Development evidence
that the scaling makes 60 fps behave like 30 fps; not part of the rule.

**Rule**, on the six videos of this plan with the labels made for it (videos below 50 fps are left out:
s = round(fps / 30) is 1 there and the change does nothing; if fewer than three remain, this footage
cannot decide), default analysis against `--fps-scaled-thresholds`:

1. pooled **serve recall** (labelled main-camera serves found as serve flights, matched as in
   `scripts/eval_serve_contact.py`) rises by **at least 10 points** (a floor: the development effect is
   much larger, so criteria 2–4 are the real test);
2. the share of matched serves within 2 frames of their bracket falls by **no more than 5 points**;
3. **no decisive call flips**: no bounce found by both runs (within 2 frames) is called IN by one and OUT
   by the other, both decisive (REVIEW in either run is not a flip). Two confident opposite calls on one
   bounce mean one is a confident error, the error the project's rules exist to stop;
4. **false serves**: serve flights not matched to a label, leaving out those followed by a labelled serve
   of the same side within 30 s (a fault the labels skip), rise by **no more than 2** over the six videos.

Otherwise it is rejected and its code removed. Reported, not used: ball detections, calls added or lost,
REVIEW changes, bounce shifts, and the speed pairs. Still counted in frames after the change: the serve
contact search windows (`b − 2 … b + 3`, `refine_contact`, `contact_in_gap`) and the 3-detection minimum
of a toss run; at 60 fps they cover half the time.
