# Event detection on independent broadcast matches (E2E-Spot tennis labels)

Plan written on 2026-11-13, **before any video was downloaded or any system output on these matches was
seen.** Nothing below has been run yet.

## Why

The TrackNet games 8–10 have been used for many accept/reject decisions (see EVAL_TRACKNET.md), and the
loss analysis there shows that most missed first bounces come from hit/bounce segmentation, not from
bounce timing or position. Testing a segmentation change honestly needs labelled matches the project has
never touched.

## Data

- **Labels:** E2E-Spot tennis (Hong et al., *Spotting Temporally Precise, Fine-Grained Events in Video*,
  ECCV 2022), `data/tennis/{train,val,test}.json` of github.com/jhong93/spot (BSD-3-Clause). Per point
  clip, the frame of every bounce, swing and serve, tagged near/far court. Timing only, no positions.
  Split by match: 9 test matches (1622 clips: 9312 bounces, 8282 swings, 1579 serves), 19 other matches
  for train/val.
- **Videos:** full-match broadcasts on the US Open and Wimbledon YouTube channels (ids in the repo's
  `videos.csv`). Kept local for research, not redistributed, like the other broadcast footage here.
- **Resolution: 360p.** On 2026-11-13 the download tool offered only a 640×360 stream for these videos.
  The ball network sees 512×288 either way, but 360p streams are heavily compressed; results are for
  360p and say so. None of these matches overlaps TrackNet or the project's own footage.

## Matches (chosen by download size, the only criterion: local disk space is limited)

| Use | Match | Clips | ≈ MB |
|---|---|---|---|
| dev | usopen_2019_womens_round1_williams_sharapova | 97 | 125 |
| dev | usopen_2021_womens_semifinal_raducanu_sakkari | 140 | 157 |
| dev | wimbledon_2019_womens_round1_gauff_williams | 142 | 236 |
| test | wimbledon_2019_womens_final_halep_williams | 64 | 147 |
| test | usopen_2019_womens_osaka_gauff | 70 | 174 |
| test | usopen_2019_womens_final_andreescu_williams | 91 | 285 |
| test | wimbledon_2018_womens_quarterfinal_williams_giorgi | 78 | 300 |

Every clip of each listed match is used (dev: train and val clips together). All seven are women's
matches, because the men's best-of-five videos are the long ones; that limits how far the numbers
generalise.

## Measurement (`scripts/eval_spot.py`, `courtvision.eval.spot`)

- **Clips** are written by decoding each full match once, in order, with OpenCV, the way the labels
  number frames (the repo's `frames_as_jpg.py`); no seeking, so the clip's first frame is the labelled
  start frame by construction (`tests/test_spot_cut.py`: B-frames, a 1.5 s start offset, overlapping
  clips). The run stops if the video's frame rate differs from the labels' (a different upload) or the
  video ends before the last clip. Each clip is analysed (`sekme analyze --no-render --no-broadcast`,
  default models) and then deleted. Residual risk: if the labellers' own OpenCV seek drifted, labels
  and frames would disagree by a constant; the median stroke timing gap is reported to show it.
- **System events:** strokes are the hit detector's hits (`report.hits`). Serves are counted two ways:
  `serve`, the flights the pipeline called a serve (contact frame; this is what sets the service box,
  and it needs a fitted flight), and `serve_toss`, hits the toss detector flagged as serves. Bounces
  are flights with a measured bounce, from the 3D fit or the image fallback (the ones that get a call);
  a bounce's side is the half it landed on (court y), not the receiver.
- **Matching:** one-to-one, closest gaps first, within **0.1 s** for bounces and strokes (the TrackNet
  bounce tolerance) and **0.5 s** for serves (the Review Room's serve labels). A labelled serve is also
  a stroke. Recall at 0.05 and 0.2 s is shown for bounces and strokes, descriptively.
- **Reported:** per kind recall, precision, F1, side agreement of matched pairs and the median timing
  gap; for bounces also the recall of *first* bounces after a stroke; per match and pooled; failed
  analyses; the share of calibrated frames; labelled events per clip per match (so a skew from the
  size-based selection is visible); code commit, model hashes and the sample file's hash.
- **Uncertainty:** with three or four matches per set, the honest spread is the per-match values, not
  a bootstrap over so few clusters.
- **Failed analyses** count as clips with no system events, and are listed. Scoring refuses to run
  until every clip has either a report or a failure record.

## Rules

1. **Baseline:** the current default system is run once on the dev and the test matches and reported.
   The baseline decides nothing.
2. **Changes:** event-logic changes are developed on the dev matches only. Before its single run on
   the test matches, each candidate gets its own rule written here, of this form: pooled first-bounce
   recall rises by at least a stated margin **and** rises in every test match; pooled bounce precision
   falls by no more than a stated margin; pooled serve recall (flight serves) does not fall by more
   than a stated margin. (Labels carry no positions, so decisive-call checks stay with TrackNet and the
   Review Room labels.)
3. **At most two candidates** are ever run on these test matches; after that they count as used, like
   TrackNet games 8–10, and a further change needs other matches.
4. A test result that rejects a candidate is final for it; its code is removed and the result stays
   documented.
