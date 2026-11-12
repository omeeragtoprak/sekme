"""Stroke type at each racket contact: serve, forehand, backhand, volley or overhead.

* Hand: at a serve contact the hitting wrist is the one raised above the head; votes are collected per
  player (by name when the broadcast names them). Without a serve the hand is assumed right (most
  professionals) and the result says so.
* Forehand / backhand: the side of the body the ball is on at contact, in the player's own frame. The
  near player has their back to the camera (their right is image right), the far player faces it
  (their right is image left). When the ball is almost in line with the hips the hitting wrist decides.
* Volley: the incoming ball did not bounce and the player is inside the service-line area.
* Overhead: a non-serve contact above the head - the hitting wrist raised *and* the ball above the head in
  the image. (A raised wrist alone is the follow-through of a groundstroke one frame later, and the 3D
  fit's rally contact height is too loosely constrained to use: two forehands came out at 2.4-2.6 m.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.perception.pose import L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, Pose

AMBIGUOUS_FRACTION = 0.15  # |ball offset| below this share of the player's box width: use the wrist
VOLLEY_MAX_Y_M = 7.0  # inside the service line (6.4 m) plus a step


@dataclass(frozen=True)
class Stroke:
    kind: str | None  # "serve" | "forehand" | "backhand" | "volley" | "overhead" | None (undetermined)
    wing: str | None  # "forehand" | "backhand" for groundstrokes and volleys
    confident: bool
    cue: str  # what decided it (shown to reviewers)


def facing(side: str) -> int:
    """+1 when the player's right is image right (near player, back to camera), -1 for the far player."""
    return 1 if side == "near" else -1


def serve_hand(pose: Pose | None) -> str | None:
    """The hitting hand at a serve contact: the wrist raised above the shoulders, clearly above the other."""
    if pose is None:
        return None
    lw, rw = pose.point(L_WRIST), pose.point(R_WRIST)
    sh = pose.mid(L_SHOULDER, R_SHOULDER)
    if lw is None or rw is None or sh is None:
        return None
    if min(lw[1], rw[1]) > sh[1]:
        return None  # neither arm is up: not the contact frame
    hip = pose.mid(L_HIP, R_HIP)
    torso = float(hip[1] - sh[1]) if hip is not None and hip[1] > sh[1] else 20.0
    if abs(lw[1] - rw[1]) < 0.3 * torso:
        return None  # both wrists at a similar height: no clear hitting arm
    return "right" if rw[1] < lw[1] else "left"


def vote_hand(votes: list[str]) -> str | None:
    if not votes:
        return None
    r = votes.count("right")
    return "right" if r * 2 > len(votes) else "left" if r * 2 < len(votes) else None


def classify(is_serve: bool, pose: Pose | None, ball_px: np.ndarray | None, box: np.ndarray | None, side: str,
             hand: str, incoming_bounced: bool | None, court_y: float | None) -> Stroke:
    if is_serve:
        return Stroke("serve", None, True, "servis tespiti")
    hs = 1 if hand == "right" else -1
    sgn = facing(side)
    width = float(box[2] - box[0]) if box is not None and np.all(np.isfinite(box)) else None
    centre = pose.mid(L_HIP, R_HIP) if pose is not None else None
    if centre is None and pose is not None:
        centre = pose.mid(L_SHOULDER, R_SHOULDER)
    if centre is None and box is not None and np.all(np.isfinite(box)):
        centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    # Wing: ball side first, hitting wrist when the ball is in line with the body.
    wing, confident, cue = None, False, "top ya da gövde görülmedi"
    if centre is not None and ball_px is not None and np.all(np.isfinite(ball_px)) and width:
        d = float(ball_px[0] - centre[0]) * sgn * hs / width  # > 0: ball on the hitting-hand side
        if abs(d) >= AMBIGUOUS_FRACTION:
            wing, confident, cue = ("forehand" if d > 0 else "backhand"), True, "topun gövdeye göre tarafı"
    if wing is None and pose is not None and centre is not None and width:
        wr = pose.point(R_WRIST if hand == "right" else L_WRIST)
        if wr is not None:
            d = float(wr[0] - centre[0]) * sgn * hs / width
            if abs(d) >= AMBIGUOUS_FRACTION:
                wing, confident, cue = ("forehand" if d > 0 else "backhand"), False, "vuran bileğin konumu"

    if pose is not None and ball_px is not None and np.all(np.isfinite(ball_px)):
        wr, nose = pose.point(R_WRIST if hand == "right" else L_WRIST), pose.point(NOSE)
        if wr is not None and nose is not None and wr[1] < nose[1] and ball_px[1] < nose[1]:
            return Stroke("overhead", None, True, "bilek ve top başın üstünde")
    if incoming_bounced is False and court_y is not None and abs(court_y) < VOLLEY_MAX_Y_M:
        return Stroke("volley", wing, confident, "gelen top sekmeden, fileye yakın vuruldu")
    return Stroke(wing, wing, confident, cue)
