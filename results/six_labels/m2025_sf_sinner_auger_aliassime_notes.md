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
