# Player selection: a size gate from the camera instead of one track's median height

Rule written on 2026-11-14, **before the change was run on any of the data below.** Development data only:
no unused labelled footage exists for player positions, so this is a development decision, stated as such.

## Problem (found on the US Open 2025 final, 150–704 s)

`select_players` keeps a detection as the near (far) player only if its box height is 0.6–1.6 × `ref_h`,
the median height of the single most present track in that half over the whole clip. On a 9-minute
highlight video the most present near-side track came from a tighter shot (345 px), while in the usual
wide shot the near player is 167–199 px tall: he was rejected, and the near player was tracked on only
**35 %** of calibrated frames (far 93 %). Without the hitter's position the 3D fits lose their contact
anchor (72 near-side flights on that clip).

## Change

The size gate compares the box height with the height a 1.85 m player standing at the detected feet would
have in that frame's calibrated camera (same 0.6–1.6 band); `ref_h` remains the fallback where a frame has
no camera. Prototype on the same clip: near player tracked on 96 % of calibrated frames (far 95 %); where
both versions track a player the position is the same on 99–100 % of frames; nine newly tracked frames
checked by eye all show the right player.

## Rule

The change becomes the default only if all hold:

1. **TrackNet games 1–7** (development; the cached WASB + TCD detections re-run with the current code,
   `data/tracknet/runs_dev_base`: first bounces 220 / 329, precision 89.1 %, decisive calls 228 / 228):
   first-bounce recall does not fall, precision falls by no more than 0.5 point, and every decisive call
   agrees with the reference.
2. **The six broadcast clips** (`runs/<id>_v4`): neither player's tracked share of calibrated frames falls
   by more than 1 point on any clip.
3. **US Open 2025 final, 150–704 s** (`results/serve_contact_labels_2025final.json`, now development
   data): serve contacts within 2 frames (ball v4, default detector) do not fall below the current 6.

Otherwise it is not adopted and the result stays here. Reported, not used: call changes on the broadcast
clips, radar pairs, serves found.

## Result (2026-11-14): not adopted

| Check | Before | Camera gate | Rule |
|---|---|---|---|
| TrackNet games 1–7: first bounces, precision, decisive calls | 220 / 329, 89.1 %, 228 / 228 | 221 / 329, 89.4 %, 229 / 229 | holds |
| Tracked share of calibrated frames, worst fall (2023 final 90–179 s, far player) | 99.0 % | 97.9 % (−1.15 points) | **fails** (limit 1 point) |
| 2025 final 150–704 s: serve contacts within 2 frames | 6 | 8 | holds |

The gains elsewhere: near player 37.0 → 96.4 % (2025 final 150–704 s) and 76.8 → 95.1 % (2025 final 60–150 s),
other clips +0.1 to +1.5 points. The 24 far-player frames lost on the 2023 clip are two short runs (910–915,
1212–1229) where the far player's box is 0.54–0.59 of the height the camera expects for a standing 1.85 m
player: a crouch or lunge, just under the gate's 0.6. The old gate measured the size from this player's
own boxes and kept them.

By the rule the change is not the default and its code was removed (patch kept locally). What it leaves
open: the camera-based reference itself works (the near player on long highlight videos); the band's
lower end is the problem. A version with a wider lower bound would be a new candidate, and deciding it on
these same clips would be choosing the band on the data that judges it, so it needs footage not used here
(for instance the six videos named in `docs/EVAL_SERVE_SPEED.md`, with player presence checked by eye).

## Follow-up (2026-11-14, afternoon): a 0.5 band as a new candidate

A version with the band widened to 0.5–1.6 is a new candidate, `--player-size-from-camera` (off by default).
The 0.6 version's rejection stands. On the same development data, after the band was chosen: TrackNet games 1–7
221 / 329 first bounces, 89.4 % precision, 229 / 229 decisive calls consistent (as the 0.6 version);
coverage on the six broadcast clips falls nowhere (true by construction for a band set just under the
lowest failing ratio, 0.54), near player 37.0 → 96.9 % (2025 final 150–704 s), 76.8 → 96.6 % (60–150 s),
91.6 → 96.3 % (QF), 90.5 → 93.9 % (2024 final); serve contacts within 2 frames 6 → 8. Because the band was
chosen on these clips, they decide nothing: it is judged on unused footage with player positions
labelled by eye (third candidate in `docs/EVAL_SERVE_SPEED.md`).

**Speed note (same day, development):** detecting people on every second frame at 60 fps and holding the
detections in between makes the 90 s 1080p60 clip 16 % faster end to end (556 → 465 s; perception 483 →
395 s), with the same calls (none changed, bounces moved ≤ 4 cm). With the default size gate, though, the
near player was then tracked on 1 093 frames instead of 2 960: the changed track ids moved the one-track
height reference again. With the camera-based gate the same fast run tracks him on 3 706 frames, positions
within 2.4 cm (median) of the full run. The option was therefore not added (patch kept locally); it is worth
adding only if the camera-based gate becomes the default.
