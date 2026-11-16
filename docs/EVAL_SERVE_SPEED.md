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

  Found by that rule on 2026-11-14 from metadata only (`scripts/six_videos_fetch.py list`; nothing
  downloaded), all "Extended Highlights" at 60 fps: 1. `6pbIaygok2E` (736 s), 2. `JP95GKqjpZk` (735 s),
  3. `-Kw7rzkJROw` (730 s), 4. `clVU6bPiwbY` (730 s), 5. `91QiqjXzBdc` (624 s), 6. `vSqQT6A2a0o` (673 s);
  at most ~3.5 GB of 1080p video in all. With ~4 GB free they are handled three at a time (download,
  label, analyse, keep the caches and reports, delete the video).

  Downloaded on 2026-11-16 (`six_videos_fetch.py download`, after the user's yes; `data/six/manifest.json`),
  before any labelling or analysis. yt-dlp's `bv*[height<=1080][ext=mp4]` picked AV1 at 1920x1080, 59.94 fps
  (format 399), so the files are smaller than the sizes listed above; OpenCV decodes them:
  - `m2025_sf_alcaraz_djokovic` (`91QiqjXzBdc`, 210 613 471 bytes), SHA-256
    `b2e9e600f4eaee6164a10c096ec61dabbe2e32fc11f18afa9b18088e4b5ec97f`;
  - `m2025_sf_sinner_auger_aliassime` (`vSqQT6A2a0o`, 234 405 859 bytes), SHA-256
    `ae18a45a0c068e5bafc75bec0ccf5cd32c416ed1679a408e4e0c3ee728d706e4`;
  - `w2025_final_sabalenka_anisimova` (`clVU6bPiwbY`, 231 110 720 bytes), SHA-256
    `e991e7ebfe092879057f21590a8543e16fc1cbf9295019c2c8e3f8d599311a11`;
  - `w2024_final_sabalenka_pegula` (`6pbIaygok2E`, 350 683 038 bytes; H.264, the same format rule), SHA-256
    `7098b25607a5b026b65ea188430c119839531f689f55eb1aa4a3940b2bc6c606`;
  - `m2024_sf_sinner_draper` (`JP95GKqjpZk`, 314 033 517 bytes; AV1), SHA-256
    `ad3090c06cc342081aeedcd7a067a527604f084facdfee6d1e1f91d87a6cfcc3`;
  - `m2024_sf_fritz_tiafoe` (`-Kw7rzkJROw`, 342 517 872 bytes; H.264), SHA-256
    `a537908cf2f8f336827602862012da6969e74161b2a6d7745ac7c5168b124247`.

  All six are local, each labelled before any analysis of it. The analyses wait for the E2E-Spot decision on
  `--far-stroke-breaks` (`docs/EVAL_SPOT.md`), which changes the analysis code either way; the labels do not.
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

## Third candidate on the same videos: the player size gate from the camera (rule written 2026-11-14, before any download)

**Change** (`sekme analyze --player-size-from-camera`; off by default): the player selection's size gate
compares a box with the height a standing 1.85 m player has at the detected feet in that frame's calibrated
camera, band 0.5–1.6, instead of 0.6–1.6 × one track's median height over the whole clip.

**Why** (development data, `docs/EVAL_PLAYERS.md`): on long highlight videos that median can come from a
tighter shot and reject the near player in the wide one (US Open 2025 final, 150–704 s: near player tracked
on 37 % of calibrated frames). A 0.6 band was tested on development data by its own rule and not adopted
(a crouching far player at 0.54–0.59 of the expected height was lost on one clip). The 0.5 band was chosen
after that, on the same data, so those clips cannot judge it: with it, no player's coverage falls on any
of the six development clips, the near player reaches 94–97 % on the four 60 fps ones, TrackNet games
1–7 give 221 / 329 first bounces at 89.4 % precision with 229 / 229 decisive calls consistent (baseline
220 / 329, 89.1 %, 228 / 228), and serve contacts within 2 frames go 6 → 8. *(Measured with the TrackNet scorer as it was then; the corrected scorer of 2026-11-14 gives the same baseline as 220 / 323 first bounces, see `docs/EVAL_TRACKNET.md`, "Two scorer corrections". Compare like with like.)*

**Labels, blind** (by the same separate labeller, after the serve labels; frames numbered by decoding each
file in order with OpenCV, as the pipeline does; files named as in `scripts/six_videos_fetch.py`):

- *random sample*: the 80 frames per video listed by `scripts/eval_six_videos.py sample` (drawn from the file
  name and frame count alone; the lists are committed before labelling);
- *disagreement sample*: after the runs, up to 60 calibrated frames per video where the default and the
  candidate pick different players (`scripts/eval_six_videos.py disagreements`), so that clustered failures,
  like the one that sank the 0.6 band, cannot hide between random frames. The labeller sees only the frames,
  never which run picked what.

For each frame: whether it is the main broadcast camera and, if so, each player's feet in the image (x, y),
or "not in frame". Committed before any scoring.

**Rule** (default analysis against `--player-size-from-camera`). A player is *found* when the tracked feet
lie within 0.3 × the tracked box's height of the labelled feet (at least 15 px at 1080 lines), so a ball
kid beside the far player is not the player; a selection is *wrong* when it is tracked away from the
labelled feet or while the player is labelled out of frame. On the random sample:

1. near-player coverage (found / labelled in frame) rises by **at least 5 points**;
2. far-player coverage falls by **no more than 1 point**;
3. wrong selections (share of player-frames) rise by **no more than 1 point**;

and on the disagreement sample, for each player with at least 10 frames where exactly one run is right
(right: found, or not tracking a player labelled out of frame):

4. the candidate is the one that is right in **at least two thirds** of them.

Inconclusive if fewer than 100 near-player frames are labelled in frame in the random sample (the
candidate stays off; these videos count as used for it). Otherwise rejected unless all hold; reported, not
used: the calls, serves and speeds of the `players` run, and the frames outside the main camera.

**On the development evidence above:** with a band chosen on those clips, "no coverage falls on them" is
true by construction (0.5 sits just under the lowest failing ratio, 0.54) and is not evidence; only the
near-player gain and the TrackNet games are informative, and only the six videos decide.

## Scoring code (committed before any download)

`scripts/eval_six_videos.py run` analyses each video once on the GPU and the variants from that cache;
`score` refuses to score a partial set and applies the rules through `courtvision.eval.six_videos`
(tests: `tests/test_six_videos.py`, including that its matching is the serve-contact test's). A dry run
on the development clip (US Open 2025 final, 150–704 s, seeded with its existing cache; nothing kept)
went through: the 60 fps rule answered "cannot decide" (one video), and the pin's criteria were computed
on 17 paired serves (mean |error| −5.3 points, 13 closer, 1 further, no serve verdict changed), which
shows the rule can be met on development data and decides nothing.

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
Indirectly: each of the four 60 fps development clips was also made into a 30 fps video (every second
frame, where these counts are what they were set for). Of the 270 calls the 30 fps runs make, the default
60 fps runs find 224 bounces (83 %), the scaled ones 243 (90 %; at least as many on every clip), with the
same verdict on 224 / 224 and 242 / 243, no decisive IN/OUT flip in either, and fewer calls the 30 fps
runs do not make (62 → 50). On the 554 s clip positions also come closer (90th percentile 13.0 → 11.5 cm).
Development evidence that the scaling makes 60 fps behave like 30 fps; not part of the rule.

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

## Scoring code corrected before any download (2026-11-14, after an independent review of the code)

A review of the scoring code against this plan (`src/courtvision/eval/six_videos.py`,
`scripts/eval_six_videos.py`), made before anything was downloaded, found places where the code did not do
what the plan says. Corrected, none of the rules changed:

- **"At least" and "no more than" hold at their boundary.** The percentages of counts were subtracted in
  floating point, so 4 → 7 found serves of 30 gave a rise of 9.999… points and failed "at least 10"; they
  are now compared exactly (fractions of counts), as are the two-thirds criteria.
- **A partial set is refused**, as written: `score` needs the serve labels of exactly the six named videos,
  and every run of each.
- **Every run is checked to be what it claims:** made with exactly its variant's flags (the report records
  them), from the downloaded file (its SHA-256 against the download manifest), with the same models in all
  runs and ball v4 (`models/sekme_ball.pt`). Runs are made only from committed analysis code, and the commit
  is taken before the run starts (a run during which the code changes is discarded). The scorer itself is
  not part of the analysis, so a change to it does not make the runs stale; `score` refuses uncommitted
  changes to either, and an existing result file (the test is scored once).
- **Frame lists:** an existing list is never overwritten; at scoring, the presence list must be the one drawn
  for the video's frame count as the analysis decoded it, and the disagreement list the one drawn from the
  runs. Labels must have every field (a missing "main camera" or player field is an error, not a default).
- **Decisive flips:** every pair of bounces within 2 frames counts, not only a one-to-one pairing, which
  could pair a bounce with the agreeing one of two neighbours and hide the opposite call.

Clarified, as the code already did it: *wrong selections (share of player-frames)* counts two player-frames
per labelled main-camera frame (near and far), whether or not the player is in view, since tracking a player
labelled out of view is a wrong selection too.

The items "reported, not used" were not produced; they now are (`reported_not_used` in the result file),
including a run with the toss bridge and the pin (`bridge_pin`), which "the same with the toss bridge" needs.
Variant runs drop their copy of the perception cache once made (it stays in the default run), for the disk.

A second review of these corrections found no blocker and one major gap, also closed before any download:
the code identity of a run covered all of `src/`, so any commit to the Review Room between two batches of
videos (deleted after each batch) would have made the earlier runs unusable. It now covers only the files
`sekme analyze` can load (`cli.py` and the import closure of what its analyze command imports, read from the
source); the one function the pipeline took from the Review Room's module (linking the source video into the
run folder) moved to `io/video.py`. Also: the frame lists count frames with the pipeline's own reader (the
first version used a faster count that could differ at a damaged last frame); a failing `git` counts as
uncommitted code; the label files must be committed (in a tracked folder, e.g. `results/six_labels/`) before
scoring; the ball model file must be there; the models must be the same for all six videos. A third check
found no blocker.

Not changed, noted: s = round(fps / 30) is already 2 from 45 fps, so "s = 1 below 50 fps" above is only
true below 45 fps; the rule keeps videos of 50 fps and more, and all six are 60 fps.

## Scoring code corrected before any download (2026-11-16, after a second review of the code)

A second review read `src/courtvision/eval/six_videos.py` against every number and condition of this plan
(tolerances, frame-rate scaling, edges, pairing order, ties, rounding, how each verdict follows from its
criteria) and pinned them in `tests/test_scorers_edges.py`. Nothing had been downloaded, drawn or labelled.
Everything matched except these two points, now corrected; no rule changed:

- **The serves the pinned run loses are listed.** Measurement says such a serve "counts as **worse** in
  criterion 2 and is listed"; the count was right, but no serve was named. The result now carries
  `lost_by_pinned`: the video (its place in the order of the six names), the serve's start frame, and what
  the pinned run did (no flight, no speed, or its rejection).
- **The disagreement sample uses the whole tolerance a player is found within.** Two runs count as picking
  different players when their feet are further apart than 0.3 × the smaller box's height **but at least
  15 px at 1080 lines** (scaled with the height), the floor of the "found" rule. Without it, two runs tracking
  the same small far player a few pixels apart counted as a disagreement and took places in the 60-frame
  sample that criterion 4 then cannot use.

Reading of criterion 2 of the second candidate, recorded now (the code already does this): if the default
run matches no serve, its share within 2 frames does not exist and cannot fall; the criterion then holds when
the scaled run matches any serve, and fails when neither does.

## Fourth candidate on the same videos: player names from consistent score readings (rule written 2026-11-16, before any download)

**Change** (`sekme analyze --score-consistency`; off by default): the player names on shots (start ends from the
marked servers' serves, then the end changes counted from the score graphic) are computed only from readings
that are consistent over time. A reading is dropped when its last set is finished while a point is under way
(the column of the set being played was not read), and, in time order, when it shows fewer sets than a reading
before it (a match's set count never falls). The report's score, serve sides and speeds still come from every
reading; only the names change (`eval/broadcast_check.py` `consistent_scores`).

**Why** (development data only, the US Open 2025 final, which is spent as a test set for names): the raw OCR boxes
show the score graphic's current-set cell is often not read at all, for instance "2 6 1 / 6 3 6" with "AD" or
"15-30" being played, the fourth set's column missing for about 80 s. The end-change count from such a reading
is off by that set's games, so the names come out on the wrong ends. On the per-flight reference of the
identity test (279 flights): default 205 right, 72 wrong, 2 unnamed; with the change 181 right, 27 wrong, 71
unnamed (`devir/scratch/names_dev.py`; the other variants tried on the same data: the set-count check alone 197 /
43 / 39, the missing-column check alone 184 / 41 / 54, the rejected single-reading check 207 / 62 / 10). The
change was chosen after looking at those errors and among four variants on the same clip, so these numbers are
optimistic; only the six videos decide. A name left out is shown as "—", which misleads less than a wrong one.

**Labels, blind** (by a separate labeller, after the runs): `scripts/eval_six_videos.py names` lists up to 40
flights per video, drawn from the video's name and the default run's flight ids alone (`names_sample`; the lists are
committed before labelling). For each listed flight the labeller sees the video frames from the flight's start
(the score graphic included) and gives the near player's surname as written in `six_videos_fetch.MATCHES`, or null
when it cannot be told. The players are told apart by appearance across the video, and appearance is tied to a
name at serves where the graphic marks the server; the labeller never sees the system's names. Committed before
any scoring (`DIR/<name>_names.json`: `[{"flight", "near"}]`).

**Rule** (default analysis against `--score-consistency`, both from the same perception cache). On the labelled
flights (null left out), the near player's name in each run is right, wrong or missing.

- **Inconclusive** if the default run names fewer than 10 of them wrongly: the candidate stays off, its code
  stays, and these videos count as used for it.
- Otherwise the candidate becomes the default only if all hold: (1) its wrong names are **at most half** of the
  default's; (2) its right names are **at least 85 %** of the default's; (3) **nothing but names changes**: every
  flight's id, frames, hitter, kind, call and bounce are the same in both runs (anything else is a fault).

  If any fails, it is rejected: its code is removed and the result stays here. Reported, not used: missing names,
  each video's counts, and stroke wing changes (the hitting hand is voted per name).

Scoring code: `names_sample`, `check_name_labels` and `names_rule` in `src/courtvision/eval/six_videos.py`; the run
`consistency` and the `names` command in `scripts/eval_six_videos.py`; pinned by `tests/test_score_consistency.py`
and the dry run (`tests/test_six_videos_dry_run.py`).

### Amendments to the fourth candidate before any download (2026-11-16, after an independent review)

An adversarial review of the rule and its scoring code, made before anything was downloaded or labelled, found
six points; all are changed, the thresholds are not:

- **Names in any case.** The labels give the surnames of `six_videos_fetch.MATCHES` in any case ("sinner",
  "Sinner"); they are compared in upper case. As first written, a lower-case label would have stopped the scoring
  after the labels were committed.
- **The tie between appearance and names is recorded and checked.** Each video's label file is
  `{"ties": [{"frame", "server"}], "flights": [{"flight", "near"}]}`: the labeller lists at least 3 serves where the
  graphic marks the server and the server is seen serving, all agreeing with the appearance the labeller uses; with
  fewer, or with any disagreeing, every flight of that video is null. One wrong tie would flip every label of a video
  in both arms.
- **Where a name came from is reported.** The tie uses the graphic's serving marker, the same signal the system's
  serve-anchored names use, so flights named from a marked serve agree with the labels almost by construction in
  both arms; the independent part is the flights named from the graphic's end changes. Each flight now records
  `names_from` ("serve", "graphic" or none), and the result gives the counts split by the default run's source.
  The criteria stay pooled over all labelled flights, as written.
- **What is reported is produced.** The result gives each video's counts and, over all flights, how many differ
  between the two runs in `server_name`, `hand`, `hand_source`, `stroke` and `stroke_wing` (the hand is voted per
  name, and a serve's server name is recomputed from the readings kept, so these may change; they are reported, not
  judged). Criterion (3) is unchanged: flights' ids, frames, hitter, kind, call and bounce.
- **Scoring is not spent without the labels.** `score` refuses when the player or name labels are missing, unless
  `--skip players` / `--skip names` gives that test up explicitly (the result records it, and the skipped candidate's
  videos count as used); before, it wrote the result without them, and the scored-once rule then barred them for good.
- **The text "sides stay as read"** refers to the report's `broadcast` section (score changes, serve sides, speeds),
  which is unchanged; a flight's `server_name` is among the names.

### Second change to the fourth candidate before any download (2026-11-16): the left-out column read from its pixels

Looking at the frames behind the remaining wrong names showed why the current set's column is lost: the graphic
shows it ("2 6 1 1 / 6 3 6 0"), but Apple Vision does not read its "0"s (nor, often, a "1" next to them), at any
crop size (`docs/ARCHITECTURE.md`, Known issues). So the candidate now reads that cell from its pixels before
giving up on the reading: where the last set read is finished while a point is played, the cell is looked for in
the gap between the last set column and the points box (the two rows' points boxes are aligned, so a blank one
uses the other's position; a gap too narrow for a cell holds none), and classified in both rows
(`broadcast/scoreboard.py` `cell_digit`, done while the graphic is read and kept as the reading's `next_set`):
"1" a narrow bar; "0" a closed ring, both sides inked down its whole body and the middle empty; anything else,
a light cell (a points box), two glyphs in the window or a fragment, is unknown. With "0" or "1" in both rows the
column is restored, otherwise the reading is dropped as before. The reading itself, and so every default output,
is unchanged.

**Review before any download.** An adversarial review of the first form of this change (a column step measured
from the set columns, a simpler ring test) found that light points boxes with dark text ("AD", "40") read as "0",
that a step from two columns could be 40 % short or 64 % long, and that other digits sometimes read as "0". The
form above answers each (no step at all; light cells, two-glyph windows and open shapes refused), and an old cache
without `next_set` is read again when the candidate is on. Measured on the six broadcast clips' stored text boxes
(`devir/scratch/ocr_confusion_dev.py`), treating each last set column Vision did read as if it had been left out:
of 331 cells holding 2–6, none read as "0" or "1" (before the review: 45); of 89 holding "1", 83 read as "1", none
as "0". Vision reads no "0" there, so the "0"s were checked by eye: six recovered readings drawn at random all
match the graphic.

A second look by the same reviewer found the points boxes refused everywhere (none of 1 420 read as "0" on the six
clips), no cell holding 2–7 read as "0" or "1" (about 930), and the recoveries on the other clips matching their
frames (the 2024 final's 10 as 1-0, the quarter-final's 3 as 0-1). Left as accepted: a thin, full-height light
mark in a wide gap (a separator) would read as "1"; the US Open graphic of all six videos has none. A real "0"'s
recall cannot be measured on Vision's own readings (it reads none), only by eye.

Development, same 279 reference flights, the graphic read again with this code: right 220, wrong 18, unnamed 41
(the first version: 181 / 27 / 71; the default 205 / 72 / 2); 52 of the 68 readings with a lost column got it back
(39 as 0-0, 12 as 1-0, one 1-1). Still development numbers on a clip used to design the change. The labels, the
rule and its thresholds are as above, unchanged.
