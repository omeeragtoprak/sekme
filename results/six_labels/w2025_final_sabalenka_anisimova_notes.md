# Blind labelling notes: w2025_final_sabalenka_anisimova

Work start: 2026-11-16 19:34. Work end: 2026-11-16 20:24.
OpenCV decoded 43741 frames (about 59.94 fps), in order, no seeking.

## Method
- Pass 1: contact sheets, then a main-camera mask (court region compared with a reference main frame) to find main-camera segments.
- Near/far player strips every 15th frame to find tosses. Full-resolution crops every 3 frames, then every frame around contact,
  for lo/hi. Far serves were re-read at 3x zoom.
- lo = last frame with a round ball on the toss path; hi = first frame where the ball is a streak. Where the ball was hidden in
  racket blur or the transition looked ambiguous, the bracket is 2-3 frames wide (e.g. 950-952, 3088-3090, 12733-12735, 31216-31219).
- Radar: stadium IBM board crop sampled every 45 frames from contact-15 to contact+480. First new value after contact.
  The board lags contact by about 2-3.5 s. Pre-contact value often differs from the previous labelled serve because many points are
  shown only partly (cutaways, replays, other angles).
- Server names (note field): white outfit = Sabalenka, dark outfit with red shoes = Anisimova.

## Skipped / not labelled
- Close-up and low-angle camera views, replays and slow motion (e.g. 1595-2225, 2857-3030 close-up, 3840-4470, 5205-6060, 8000-8800
  and most of 10000-12400): not the main wide camera.
- Toss let go without a swing at about 14490 (ball dropped, then a real serve at 14571): not a serve. Racket raised at 16051-16081
  (receiver, not a serve).
- Serves that happen in cutaway footage (board values change between consecutive labelled serves several times) are not labelled.
- Camera pan/zoom stretches of the main camera (22095-22500, 23490-23720, 29480-29500, 32805-33345, 21440-21545, 30545-30875)
  had no toss visible; any serve there would have been missed.

## Uncertain
- 28630 (far, Anisimova): board stayed 109 for 480 frames after contact (same as previous serve), radar null.
- 21878 (far): board hidden by a camera pan; first new value (105) read at 22100 in the panned view.
- 26836: board hidden by pan around contact; 108 is the value already present, first new value 111 read at 27001.
- 12733-12735, 31216-31219: ball hidden near the racket, wider bracket.
- 39133: tens digit read as 7 at 39314-39328 ("79"), partly hidden by a ball kid afterwards.

## Presence (second task)
- 80 frames labelled once each. main=true for the high wide main camera (including slightly panned/zoomed shots: 22150, 29490,
  29494, 30747); main=false for close-ups, replays, low-angle behind-the-baseline shots, ceremony.
- Feet = midpoint between the feet (or grounded foot when one is in the air), from 1:1 near crops and 3x far crops with crosshairs,
  accuracy a few pixels (near) to about 5 px (far, motion-blurred ones).
- 15562 and 15565 are dissolves between two main shots; the clearer image was labelled.
- 29490 and 29494: heavily panned and blurred; only the near-side player is in frame (far null). main=true is a judgement call.
