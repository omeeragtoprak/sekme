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

## Second round

Blind labeller, second round. Start 2026-11-17 10:54, end 2026-11-17 11:08 (local). Video decoded in order with OpenCV on CPU, one process at a time; no seeking. OpenCV frame count 43741.

Method: one sequential decode saved the listed disagreement frames and 6 evenly spaced frames per flight (start..end); contact sheets and 3x zoom crops with a labelled pixel grid were read by eye. Feet = point between the feet on the ground, full-resolution pixels.

How the players are told apart (checked against the score graphic): Sabalenka = white dress with a black-trimmed top, blonde hair, pink shoes. Anisimova = dark grey-blue sleeveless dress, brown hair in a bun, red shoes. The serving marker (yellow ball next to a name) agreed with this in all ties.

Ties (graphic marks the server, main camera, frame = about the contact): 953 Sabalenka (near, 30-0 game 1), 4678 Anisimova (near, 15-15 game 2), 14570 Sabalenka (near), 26103 Anisimova (near), 26837 Sabalenka (near), 32985 Sabalenka (near, 6-5). No far-side serve was used (far server too small to tell reliably against the graphic at the same time).

Task A: 60 frames; 59 main camera, 1 not main (13057: dissolve between the main view and a Sabalenka close-up, labelled main=false). Uncertain feet: 9092, 21838, 35677 (near feet hidden behind the score graphic: estimated from the leg direction, error maybe 20-30 px in y); 21442 (near feet below a foreground bench strip at the bottom edge: estimated); 2637, 16898, 39002 (wide stance, one foot cut by the crop; midpoint taken from a wider view, a few px more error); 17473, 25145, 27600 (near feet at the very bottom of the frame). Far feet are mostly within about 5 px; frames where a far foot is in the air (16966, 37819, 38397) use the ground foot or a midpoint.

Task B flights: 40 flights; near named Sabalenka 25, Anisimova 14, null 1. Flight 19 is null (low behind-baseline and net-level close views, not the main camera). Uncertain: flight 303 (near player only partly visible at the bottom edge behind the graphic; white skirt seen, named Sabalenka); flight 257 contains a dissolve near frame 31471 (near player Anisimova throughout); flight 149 is a walk after the point (near Sabalenka, main camera).
