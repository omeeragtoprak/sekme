# Results

Saved outputs of the evaluation scripts, so that every number in the README and in `docs/` can be traced.
Each file names the script that produced it. Datasets and model weights are not in the repository; the
scripts reproduce these files from them.

| File | What | Produced by |
|---|---|---|
| `ball_network_test.txt` | ball networks on TrackNet games 8–10, frame level | `scripts/eval_ball.py` |
| `court_network_test.txt` | court networks on the TennisCourtDetector validation split, with leak-free subsets | `scripts/train_court.py --test` |
| `tracknet_benchmark.txt` | end-to-end bounce benchmark, games 8–10, three model sets | `scripts/eval_tracknet.py` |
| `uncertainty_calibration.txt` | whether the reported bounce uncertainty is honest | `scripts/calibration_check.py` |
| `players_human_labelled.txt` | player recall on hand-labelled frames (Tennis Player Actions, CC BY 4.0) | `scripts/eval_mendeley.py --players` |
| `pose_human_labelled.txt` | pose PCK on hand-labelled frames; stroke-type agreement on US Open 2023 | `scripts/eval_mendeley.py --pose`, pipeline runs |
| `radar.txt` | fitted serve speed vs the stadium radar board, current defaults, 5 clips | `sekme analyze` reports |
| `position_bias.txt` | TrackNet label vs ball image centre per game (no network; constant + speed term), network offsets, bounce depth bias vs raw and centred references, bootstrap intervals | `scripts/label_offset.py`, `scripts/detection_offset.py`, `scripts/eval_ball.py`, `scripts/eval_tracknet.py --centred-reference`, `scripts/bootstrap_bounce.py` |
| `label_noise.txt` | per-frame noise of the TrackNet label, the image centroid and two ball networks (three-cornered hat) | `scripts/label_noise.py` |
