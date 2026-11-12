# SEKME — honest tennis line-call decision support

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="promo/assets/sekme-logo.svg">
  <img src="promo/assets/sekme-logo-light.svg" alt="SEKME" width="360">
</picture>

From a single broadcast camera: calibrated court, ball and player tracking, a physics-based 3D
flight fit for every shot, and **probabilistic** line calls (P(in) with a 1-sigma margin) that a
human confirms in a VAR-style *Review Room*. The system never claims millimetres it cannot measure;
anything not decisive at 95 % goes to human review.

> *Sekme* is Turkish for the bounce of a ball. Python package: `courtvision`.
>
> Status: alpha (3.0.0a1). Works end-to-end on real broadcast footage on an Apple M4 laptop.
> Numbers below are measured, and the known gaps are listed — see *Limitations*.

## Quick start

```bash
uv venv .venv --system-site-packages && uv pip install --python .venv/bin/python -e ".[dev]"
# model weights (see docs/LICENSES.md): models/court_kps_yastrebksv.pt, models/wasb_tennis_best.pth.tar, models/yolo26s.pt
PYTHONPATH=src .venv/bin/python -m courtvision.cli analyze data/videos/<clip>.mp4 --out runs/<id>
cd web && npm install && npm run build && cd ..
PYTHONPATH=src .venv/bin/python -m courtvision.cli serve --runs runs     # http://127.0.0.1:8765
PYTHONPATH=src .venv/bin/python -m courtvision.cli benchmark --fps 60   # simulated accuracy for a camera setup
.venv/bin/python -m pytest                                              # 46 tests (1 needs the sample clip)
```

`--reuse-perception` re-runs analysis/rendering from the cached detections (seconds instead of minutes).

## What it does

| Stage | How | Honesty mechanism |
|---|---|---|
| Court calibration | keypoint heatmap net on keyframes/cuts → homography → **sub-pixel refit on the painted line ridges** (FWHM centres) → pinhole camera (focal + pose) | frames whose fit is not backed by the lines are *not calibrated* and never measured |
| Shot cuts | HSV histogram correlation | flights never cross a cut or an uncalibrated span |
| Ball | WASB (HRNet, BMVC 2023), 3-frame sliding window, constant-acceleration association that keeps gating across short misses, outlier removal, short-gap fill | candidates that stay on one spot of the court (the model fires on the baseline centre marks) are dropped; interpolated points are flagged and not used in fits |
| Players | YOLO26 + ByteTrack, near/far selection on the court plane, speed (11 m/s) and size gates, re-seeding after gaps | Hampel-filtered kinematics, peak reported as p98 |
| Events | persistent direction reversals = hits; two-piece changepoint = bounce, restricted to the receiving half | — |
| Strokes | YOLO26-pose (17 keypoints) on the hitter at each contact: hand from the raised wrist at serves (voted per named player), forehand/backhand from the ball's side in the player's own frame, volley (no incoming bounce, near the net), overhead (wrist and ball above the head) | 16 / 16 forehand/backhand checked by eye on the US Open clip; ambiguous contacts stay undetermined |
| Serves | toss = slow, near-vertical ball above a player on the baseline after a dead ball; contact = changepoint between toss and struck flight; contact height measured on the camera ray above the server and used to pin the fit | the service box is used only when the toss and the contact height (> 2.1 m) agree; with one cue the call is decisive only if reading it as a rally shot gives the same verdict |
| 3D flight | bounce-anchored fit: gravity + drag + per-phase Magnus spin; contact times are free parameters; players' court positions anchor the contacts; robust stage then Gaussian stage | plausibility gates (speed, height, residual, observation count, bounce sigma) → no call |
| Broadcast check | Apple Vision OCR (on-device) of the score graphic and the stadium radar board every 0.5 s: names, server marker, score changes, serve speeds; ITF Rule 10 end changes name the near/far players | used only to check the pipeline, never as an input to a call; values count only when read twice in a row |
| Ground truth labels | reviewer clicks the ball's ground contact at a chosen frame (4x loupe); the pixel goes through the inverse homography and is judged with the system's own rule in the same zone; append-only `labels.jsonl` | each label reports its own resolution (cm per pixel); summary: error median/p90, share inside the 95 % ellipse, verdict agreement |
| Line call | Monte Carlo P(in) over the bounce covariance against the legal zone (outer line edges + footprint) | IN/OUT only at ≥ 95 %; OUT contradicted by continued play is downgraded to REVIEW |
| Review Room | FastAPI + React/three.js, Turkish UI after Apple's HIG for macOS 27 / iOS 27 (toolbar, smart-filter sidebar, inspector, light/dark, phone layout): frame-accurate player with client-side overlays, timeline, mini-court with 95 % ellipses, 3D replay, human confirm/overturn | append-only audit log with the system call and model hashes; probabilities never shown as %100 / %0 |

## Measured results

**Simulation** (`courtvision benchmark`; broadcast camera 28 m back, 11 m high, 1080p, 1.5 px detection
noise, topspin/slice the fitter only approximates, 50 % of shots aimed within 15–20 cm of a line):

| | 30 fps | 60 fps |
|---|---|---|
| median / p90 bounce error | 4.7 / 10.5 cm | 2.7 / 8.2 cm |
| truth inside reported 95 % ellipse | 96.7 % | 95.0 % |
| decisive (IN/OUT) calls | 73 % | 93 % |
| wrong decisive calls | 0 / 60 | 0 / 60 |

**Real broadcast footage** (no ground-truth labels yet — checked against play outcomes and by eye):

| | Vienna clip (1080p, 7 s) | US Open 2023 final highlights (720p, first 90 s) |
|---|---|---|
| calibrated frames | 214 / 214 | 2530 / 2698 (the rest are close-ups, correctly excluded) |
| line-fit residual (median) | 0.75 px | 0.34 px |
| camera recovered | 49 m back, 13.2 m up, on axis | 33 m back, 10.0 m up, on axis |
| ball detected | 96 % of frames | 87 % of frames (court-mark false positives removed) |
| shots with a call | 5 / 6 | 53 / 66 (6 fits rejected by gates, 7 without a clear bounce) |
| serves | none in the clip (rally only), none reported | 3 / 3 visible serves found, 0 false; independently confirmed by the stadium radar board (6 board updates: 3 matched, 3 classified as serves edited out at cuts); all IN in the correct service box |
| calls | 5 IN | 51 IN, 1 OUT (verified by eye: ball lands 1.2 m beyond the baseline, point ends), 1 REVIEW |
| serve speed vs stadium radar | — | 5 official US Open highlight clips (2023 final ×2, 2024 final, 2025 final, 2025 QF). Pairs kept only when the board change is timed to ≤ 4 s (it follows the serve by 2.5–3.5 s): **6 serves, mean −0.5 % (SE 2.5 %), mean |error| 4.9 %, SD 6.1 %** — no detectable bias, ~6 % per-serve scatter. 17 board changes were not attributable (timing unknown: edited highlights), 11 followed a cut serve. Fixes on the way: serve contact at the toss/struck-flight meeting point, boards read with the unit glued on ("116MРL"), no serve right after a cut when the ball was in play, rejected fits' speeds no longer compared |
| server marker vs detected side | — | consistent for both serves with a visible score graphic |
| median contact speed | — | 116 km/h |
| processing (M4, 16 GB) | ~50 s | perception 383 s + analysis 43 s + broadcast OCR 22 s + render 48 s |

Every IN call on both clips is consistent with play continuing. One REVIEW (P(in) 13 %, margin −16 ± 14 cm)
was followed by the receiver playing the ball, which suggests a ~20 cm far-baseline bias in that shot.

## Limitations (also shipped in every report)

- One camera at 25–30 fps: decimetre-level uncertainty along the camera axis; ITF Bronze/Silver (±25 mm bands) is out of reach. Use 60+ fps or two synchronised cameras for tighter calls.
- Independent benchmark on human-labelled broadcast bounces (TrackNet, games unseen by the ball detector, `docs/EVAL_TRACKNET.md`): 77 % of first bounces after a stroke (73 % of all bounces) found, 93 % precision, median 11 cm (p90 33 cm) from the labelled contact, 108/108 decisive calls consistent. The reference cannot resolve errors below ~10 cm, so centimetre-level line-call accuracy is still unmeasured; for that, label your own footage with the tool below. The labelling tool now exists (Review Room → shot → *Sekme Noktasını İşaretle*); the numbers appear in Analiz → *Etiketli doğruluk* once shots are labelled.
- Serve detection is validated on only 3 real serves (all found, no false positives on 90 s); that is far too few to state a detection rate. Serves struck before a camera cut cannot be recovered.
- Net clearance carries ~0.1–0.25 m 1-sigma (median 14 cm on the US Open clip) and is reported with it; a slightly negative value within 2 sigma is not evidence of a net cord.
- Lens distortion is not modelled (~±0.7 px pattern at the image edges, 3–10 cm on the ground at bounce points). It is carried as uncertainty instead: every call adds a systematic calibration term (max(court-line RMSE, 0.5 px), propagated through the homography) to the fit covariance. On both US Open clips this changed no verdict; median 1σ across the line went 2.6 → 2.9 cm (2023) and 1.6 → 1.7 cm (2025).
- Ball footprint is a fixed 2 cm half-length; scoring/point outcome inference is not implemented.
- Analysis runs ~3.6× slower than real time on an M4 (90 s of 720p30: 323 s end to end, was 438 s): fp16 ball and person networks, court calibration in a worker thread, identical calls. fp16 breaks the court network (14 → ~1 keypoints found), so it stays fp32; no CoreML/ONNX export yet.
- Court-keypoint weights have no published licence (research only); YOLO26 is AGPL-3.0 — see `docs/LICENSES.md`.

## Language

Everything a person reads is Turkish: the Review Room, report notes and limitations, burned-in HUD
videos and API error messages (`src/courtvision/i18n.py`, `web/src/i18n.ts`). Data identifiers stay
English (verdict codes `IN/OUT/REVIEW`, line names such as `far baseline`) so reports, the review log
and tests stay stable.

## Layout

```
src/courtvision/  calib/ geometry/ perception/ physics/ events/ officiating/ analytics/ render/ api/ eval/ sim/ io/
web/              Review Room (React 19, Vite 8, three.js 0.186)
tests/            unit, regression (real-footage failures), API security, end-to-end
docs/             PLAN_v3_TR.md (plan), frontend_spec.md, LICENSES.md, EVAL_TRACKNET.md (benchmark)
```

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE). Model weights and videos are not part of this
repository: the player/pose detectors are Ultralytics YOLO26 (AGPL-3.0), the ball detector is WASB
(MIT), and the court-keypoint weights have no published licence (research use only). Details and
download sources: [docs/LICENSES.md](docs/LICENSES.md). Broadcast footage used during development is
not redistributed.
