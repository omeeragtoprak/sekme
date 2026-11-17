# Blind labels: m2024_sf_sinner_draper (Sinner vs Draper, 2024 US Open SF)

Work: 2026-11-16 20:32 to 21:30 (one labeller, by eye). Only the video, the template/helper scripts
and the presence frame list were opened.

## Method
- One sequential OpenCV decode (no seeking): 44039 frames (`n_frames_decoded_by_cv2`).
- Contact sheets every 15th frame at 320x180 (62 sheets) to find main-camera rallies and serves.
- Full-resolution crops every 1-4 frames around each candidate (3-4x zoom on far servers); bracket read there.
- Board: crops of the speed board every 60 frames from hi-30 to hi+510; first changed value after contact.

## Serves
25 serves (12 near, 13 far); radar read for 23, null for 2 (25632 and 33373: the view cut away to other
cameras before the board changed). Brackets are 1-2 frames wide except a few far serves with 3 frames
(motion blur / ball hidden in racket blur).
Servers: near = Draper at 1619, 9149, 10210, 20747, 25631, 30047, 32598, 40579, 41660; Sinner at 12284,
33372, 35700. Far = Draper at 3388, 14340, 15201, 17178; Sinner at 4283, 6203, 19129, 19978, 21956,
23769, 30908, 34578, 36414.

## Skipped serves (not on main camera)
Serves shown on low side camera, close-up, or replay: about 5400-5510, 8325-8415 (walking only), 11063,
15840-15885, 24840-24885 (contact on close-up, main camera shows only follow-through), 28080-28350,
37020-37065, 38565-38670 (close-up toss). 21322 near Draper: toss then abandoned (no racket contact),
not labelled.

## Uncertain
- Serves at 1619, 9149, 10210 use a slightly zoomed behind-baseline view (scoreboard bug visible, board
  readable); treated as main camera.
- Far Draper brackets 14340-14342 and 17178-17180 have heavy racket blur (3 frames wide).
- Radar: the board shows the previous serve until the new value appears; first change after contact was
  recorded in mph, sampled every 60 frames (so the change frame is only known to about 1 s).

## Presence (80 listed frames)
44 main-camera frames (including the low behind-baseline view at 10389 and 10621), 36 not main (close-ups,
crowd, other angles, graphics). Near player feet found in 41 of the 44; far player found in 44 of 44.
Near = null when the near player is in frame but his feet are cut off by the frame bottom
(14333, 24440, 36737). Coordinates read from a 100 px grid on full-res frames (about +-8 px at the far
baseline, +-15 px for near).

## Second round

Start 2026-11-17 07:05, end 07:23 (+03). Video decoded in order with OpenCV (39200 frames, 1080p); CPU only, one decoding process at a time; no `runs/`, `src/` or other results opened.

Task A (60 disagreement frames): all 60 are the wide main camera (several early frames, 841-1888, are a tighter zoom of the same behind-baseline camera; still main). Feet read from 5-8x crops with a pixel grid, point between the feet; when one foot is in the air the point is placed between the planted foot and the other foot. Near player null in 4 frames: 1124 (near player only partly in the right edge, feet below the frame), 14376 and 14408 (near player's legs cut by the bottom wall/frame, feet out of view), 16552 (near player not in the frame; only a ball person at the right). In 11 frames the near player's feet are hidden behind the score graphic (6141, 6147, 6154, 6188, 21867, 21886, 21908, 30828, 30842, 30853, 34494; head and torso visible): feet are an estimate, head y + about 160 px (calibrated on frames of the same shot where feet show), x from the torso; treat these as +-25 px. Far player feet: within about 3-5 px where shoes are visible (most frames); 9216, 9391, 17228, 25217 have one foot hidden or blurred (+-8 px).

Task B appearance rule (stable all match): Draper = magenta/purple shirt and shorts, red-white shoes; Sinner = grey-olive shirt, dark shorts. Ties (score graphic server marker next to the name, serve seen on the main camera; frame = estimated contact, +-3 frames): 12286 Sinner (near, grey), 17177 Draper (far, magenta), 24911 Draper (near), 25637 Draper (near), 32595 Draper (near, tiebreak), 33374 Sinner (near, set point). Also consistent but not listed as ties (frame near toss only): 150 Sinner serving from far, 3585 Draper far, 4584 Sinner far. All six ties agree with the clothing rule.

Flights: near name from the clothing rule at the flight's frames; 28 Draper, 12 Sinner, 0 null. Weakest: flight 295 (frames 37910-38000): near player (grey, diving) is almost out of the frame under the score graphic, named Sinner from the grey outfit and the magenta far player. Flight 137 starts on a dissolve (frame 15015) and is named from the later frames. Flights 17-19 and 0-4 are the tighter-zoom shots; the near player is clear there.
