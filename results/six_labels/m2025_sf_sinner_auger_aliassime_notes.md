# m2025_sf_sinner_auger_aliassime: labelling notes

Start 2026-11-16 19:34, end 2026-11-16 20:10 (local time).

Method: one in-order OpenCV decode (40311 frames, equals the container count) saving every 15th frame at 320x180 and the 80 listed presence frames at full resolution; contact sheets (every 30th frame) to find main-camera serves; then in-order full-resolution decodes with crops (3-4x for far servers) for +-8..+-12 frames around each serve; ball blob colour/shape measured per frame as an aid to the eye bracket; scoreboard speed board cropped every 60 frames from contact-30 to contact+510 (first changed value). Pre-serve sheets (8 s before each serve) were looked at to make sure no first-serve fault was missed in main-camera footage. No system output, run, doc or other video was opened.

Serves: 24 (19 near, 5 far), 22 radar readings (mph). Server names are not stored in the json (Sinner and Auger-Aliassime alternate ends).
Bracket rule: lo = last round ball on the toss path, hi = first streak/hidden-in-racket frame. Brackets of 3 frames (hi = lo+2): 22740 (ball hidden in racket blur at 22739), 17318 (ball lost over the logo letters 17315-17317), 28683 (ball hidden at 28682). Slightly uncertain by one frame: 20233 (ball a small smear at the racket), 17318, 28683, 30762.

Radar null: serve 11069 (board stays 125, camera cuts away at ~11205 before any new value) and serve 30762 (board shows 123 before and after, no change).
Skipped: serves/rallies shown from other cameras (low angle, net level, close-ups, replays, slow motion) and every point whose serve was not in the main camera. An earlier part of the work (sheets 0-20) was reviewed before a context reset; the candidate list was kept, and the pre-serve sheets and board chain were used as a cross-check.

Presence: 80 frames; main = true in 36, false in 44 (close-ups, low-angle/net-level/side cameras, crowd, ceremony). Feet = point between the feet on the ground, at 4.5x zoom with a pixel grid. Notes:
- 25070 (camera panned right) near player not in frame: null.
- 25285 and 25328: the near player (walking off court) is in frame at the right edge but his legs are hidden behind the stand/barrier; feet are an estimate (y about 1030).
- 17981 and 31063: near player's feet partly/fully behind the score bug; 17981 uses the visible shoe, 31063 is an estimate.
- 34844: far player (Sinner) is not on court in the frame, only a ball person: far null.
- Blurred, wide-stance or lunging poses (e.g. 1533 far, 15156 near, 34570 near) have a larger error (about 10 px).

## Second round
Start 07:58, end about 08:22 (2026-11-17). Method: one sequential OpenCV decode (CPU) saving the 60 listed frames at full resolution and 5 sampled frames per flight at half size; far feet read on 4x crops with a 10 px grid, near feet on 2.2x crops (shoe centres, midpoint of the two feet; for a raised foot the midpoint of both).
Telling the players apart: Sinner = navy polo and navy shorts, cap, tall, burgundy/white shoes; Auger = white shirt, black shorts, yellow shoes, neon racket.
Ties (server marker on the score graphic, server seen tossing on the main camera): 9206 Auger (far, white), 16502 Sinner (far, navy), 31383 Auger (far, white), 33072 Sinner (far, navy). All agree with the clothing rule.
Disagreement frames: main = elevated behind-baseline view with net and near baseline in view; marked not main: 22068 and 27256 (close-up, near baseline out of frame), 30527, 34958, 34964 (dissolves), 31925 (different low camera, near half cut off). Zoomed variants of the main camera (6804, 6983, 7031, 13101, 13243, 27092, 27222) counted as main.
Near feet null although main: feet below the frame or hidden (6804, 7031, 28184, 33241, 33346, 33436, 34981, 34997), near player only partly in frame (8625, 25256). Near feet estimated although occluded by the score graphic or camera platform: 8034, 8143, 8494, 8526, 16822, 31072, 31383, 33020, 33046, 33072 (left foot or both feet behind overlay; error maybe 20-40 px).
Flights: names from the mid frame, camera/cut checked on 5 frames. Null: 220 and 222 (camera on far half, only a white-shirted person outside the court at the bottom corner, unsure), 251 (near court empty). 289 and 199 end partly off-frame / in a dissolve but the name is clear.
