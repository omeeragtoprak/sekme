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

Ball v4 and the default serve detector (no toss bridge). Perception once per video (one GPU job at a
time), then two analyses from the cached perception: without and with `--pin-serve-height`. Nothing
else differs. With the toss bridge as well: reported, not used.

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
