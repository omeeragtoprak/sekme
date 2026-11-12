"""Broadcast-style overlays.

* ``classic``: the reference look (numbered yellow court keypoints, green
  player boxes with speeds, yellow ball marker, mini-court, bottom stats bar).
* ``pro``: classic plus ball trail, bounce markers with their 95% uncertainty
  ellipse projected onto the court, probabilistic line-call cards and a
  calibration health badge that says when the system cannot measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from courtvision.calib.camera import apply_h
from courtvision.geometry.court import CourtModel
from courtvision.i18n import VERDICT_UPPER, line_name, num, prob

# BGR colours
YELLOW = (0, 220, 255)
GREEN = (60, 220, 90)
WHITE = (245, 245, 245)
GREY = (190, 190, 190)
RED = (60, 60, 235)
AMBER = (0, 170, 255)
PANEL = (28, 24, 22)
VERDICT_COLOR = {"IN": GREEN, "OUT": RED, "REVIEW": AMBER}


def lean_in(call: dict) -> bool:
    """Autonomous reading of a non-decisive call: geometry, unless the receiver played the ball on."""
    return bool(call.get("play_continued")) or call["p_in"] >= 0.5

_FONT_CANDIDATES = [
    ("/System/Library/Fonts/Avenir Next Condensed.ttc", 2),  # Demi Bold
    ("/System/Library/Fonts/Helvetica.ttc", 1),
]


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path, index in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                continue
    return ImageFont.load_default(size)


@dataclass
class TextItem:
    xy: tuple[int, int]
    text: str
    size: int
    color: tuple[int, int, int] = WHITE  # BGR


def draw_texts(frame: np.ndarray, items: list[TextItem]) -> np.ndarray:
    """Render all text in one PIL pass (Hershey fonts look amateurish on broadcast overlays)."""
    if not items:
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for it in items:
        b, g, r = it.color
        d.text(it.xy, it.text, font=_font(it.size), fill=(r, g, b))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def panel(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.72,
          color: tuple[int, int, int] = PANEL) -> None:
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


class MiniCourt:
    """Top-right 2D court; world metres -> panel pixels (far baseline at the top)."""

    def __init__(self, frame_w: int, frame_h: int, court: CourtModel) -> None:
        self.h = int(frame_h * 0.42)
        self.w = int(self.h * 0.55)
        self.x0 = frame_w - self.w - int(frame_w * 0.02)
        self.y0 = int(frame_h * 0.05)
        self.court = court
        margin = 0.12
        self.sx = self.w * (1 - 2 * margin) / 10.97
        self.sy = self.h * (1 - 2 * margin) / 23.77
        self.cx = self.x0 + self.w / 2
        self.cy = self.y0 + self.h / 2

    def to_px(self, xy: np.ndarray) -> tuple[int, int]:
        return int(round(self.cx + xy[0] * self.sx)), int(round(self.cy - xy[1] * self.sy))

    def draw_base(self, frame: np.ndarray) -> None:
        panel(frame, self.x0, self.y0, self.x0 + self.w, self.y0 + self.h, 0.78, (54, 40, 30))
        for _, a, b in self.court.line_segments():
            cv2.line(frame, self.to_px(a), self.to_px(b), WHITE, 1, cv2.LINE_AA)
        cv2.line(frame, self.to_px(np.array([-6.4, 0.0])), self.to_px(np.array([6.4, 0.0])), (0, 140, 255), 2,
                 cv2.LINE_AA)


def ground_ellipse_px(H: np.ndarray, mean: np.ndarray, cov: np.ndarray, k: float = 2.4477) -> np.ndarray:
    """95% ground-plane ellipse (k = sqrt(chi2_2(0.95))) projected into the image."""
    w, V = np.linalg.eigh(cov)
    w = np.sqrt(np.clip(w, 0, None))
    ang = np.linspace(0, 2 * np.pi, 48)
    circ = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return apply_h(H, mean + k * (circ * w) @ V.T)


@dataclass
class OverlayState:
    """Everything the renderer needs for one frame (built by the pipeline)."""

    frame_idx: int
    t: float
    calib_ok: bool
    H: np.ndarray | None
    calib_note: str
    ball_px: np.ndarray | None
    ball_trail: list[np.ndarray]
    players: dict[str, dict]  # side -> {box, world, speed_kmh, dist_m, peak_kmh, track_id}
    last_shot_speed: float | None
    bounces: list[dict]  # {xy, cov, verdict, p_in, margin_m, sigma_m, line, age}
    latest_call: dict | None
    score: list[dict] | None = None  # rows read from the broadcast graphic: name, games, points, serving
    names: dict[str, str] | None = None  # side -> player name, when the graphic names them


class Renderer:
    def __init__(self, width: int, height: int, court: CourtModel) -> None:
        self.w, self.h = width, height
        self.court = court
        self.mini = MiniCourt(width, height, court)
        self.s = height / 1080.0

    def _u(self, v: float) -> int:
        return max(1, int(round(v * self.s)))

    def classic(self, frame: np.ndarray, st: OverlayState) -> np.ndarray:
        """The reference look, but every overlay comes from a verified calibration."""
        out = frame.copy()
        texts: list[TextItem] = []
        u = self._u
        if st.calib_ok and st.H is not None:
            for _, a, b in self.court.line_segments():
                cv2.line(out, _pt(apply_h(st.H, a)), _pt(apply_h(st.H, b)), WHITE, u(1), cv2.LINE_AA)
            for k, p in enumerate(apply_h(st.H, self.court.keypoints())):
                cv2.circle(out, _pt(p), u(7), YELLOW, -1, cv2.LINE_AA)
                cv2.circle(out, _pt(p), u(7), (0, 0, 0), 1, cv2.LINE_AA)
                texts.append(TextItem((_pt(p)[0] + u(9), _pt(p)[1] - u(30)), str(k), u(24), YELLOW))
        self._players(out, texts, st, thickness=u(3))
        if st.ball_px is not None:
            c, r = _pt(st.ball_px), u(10)
            cv2.rectangle(out, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), YELLOW, -1)
            cv2.rectangle(out, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), (0, 0, 0), 1)
        self._mini(out, st, pro=False)
        self._bottom_bar(out, st, texts)
        return draw_texts(out, texts)

    def pro(self, frame: np.ndarray, st: OverlayState) -> np.ndarray:
        out = frame.copy()
        texts: list[TextItem] = []
        u = self._u
        if st.calib_ok and st.H is not None:
            for _, a, b in self.court.line_segments():
                cv2.line(out, _pt(apply_h(st.H, a)), _pt(apply_h(st.H, b)), (230, 230, 230), 1, cv2.LINE_AA)
            for bnc in st.bounces:
                self._bounce_marker(out, texts, st.H, bnc)
        n = len(st.ball_trail)
        for i in range(1, n):  # comet trail
            t = i / max(n - 1, 1)
            col = (int(40 + 200 * t), int(200 * t), 255)
            cv2.line(out, _pt(st.ball_trail[i - 1]), _pt(st.ball_trail[i]), col, max(1, int(u(5) * t)), cv2.LINE_AA)
        if st.ball_px is not None:
            cv2.circle(out, _pt(st.ball_px), u(9), YELLOW, u(2), cv2.LINE_AA)
        self._players(out, texts, st, thickness=u(2))
        self._mini(out, st, pro=True)
        self._var_card(out, st, texts)
        self._health_badge(out, st, texts)
        self._bottom_bar(out, st, texts)
        return draw_texts(out, texts)

    def _players(self, out: np.ndarray, texts: list[TextItem], st: OverlayState, thickness: int) -> None:
        u = self._u
        for label, side in (("Oyuncu 1", "near"), ("Oyuncu 2", "far")):
            if st.names and side in st.names:
                label = st.names[side].title()
            p = st.players.get(side)
            if p is None or p["box"] is None:
                continue
            b = p["box"]
            cv2.rectangle(out, _pt(b[:2]), _pt(b[2:]), GREEN, thickness, cv2.LINE_AA)
            spd = p["speed_kmh"]
            txt = f"{label}  {num(spd)} km/sa" if spd is not None else label
            texts.append(TextItem((int(b[0]), int(b[1]) - u(36)), txt, u(30), GREEN))

    def _bounce_marker(self, out: np.ndarray, texts: list[TextItem], H: np.ndarray, bnc: dict) -> None:
        u = self._u
        col = VERDICT_COLOR.get(bnc["verdict"], AMBER)
        ell = np.round(ground_ellipse_px(H, np.asarray(bnc["xy"]), np.asarray(bnc["cov"]))).astype(np.int32)
        fade = max(0.25, 1.0 - bnc["age"] / 60.0)
        ov = out.copy()
        cv2.fillPoly(ov, [ell], col, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.35 * fade, out, 1 - 0.35 * fade, 0, out)
        cv2.polylines(out, [ell], True, col, u(2), cv2.LINE_AA)
        c = _pt(apply_h(H, np.asarray(bnc["xy"])))
        cv2.drawMarker(out, c, col, cv2.MARKER_CROSS, u(14), u(2), cv2.LINE_AA)
        if bnc["age"] < 45:
            word = VERDICT_UPPER[bnc["verdict"]] if bnc["verdict"] != "REVIEW" else (
                "İÇERİDE?" if lean_in(bnc) else "DIŞARIDA?")
            label = word if bnc["verdict"] == "REVIEW" else f"{word}  {prob(bnc['p_in'], 0)}"
            texts.append(TextItem((c[0] + u(16), c[1] - u(4)), label, u(26), col))

    def _mini(self, out: np.ndarray, st: OverlayState, pro: bool) -> None:
        m, u = self.mini, self._u
        m.draw_base(out)
        if pro:
            for bnc in st.bounces:
                col = VERDICT_COLOR.get(bnc["verdict"], AMBER)
                c = m.to_px(np.asarray(bnc["xy"]))
                w, V = np.linalg.eigh(np.asarray(bnc["cov"]))
                ax = (max(1, int(2.45 * np.sqrt(max(w[1], 0)) * m.sx)),
                      max(1, int(2.45 * np.sqrt(max(w[0], 0)) * m.sy)))
                cv2.ellipse(out, c, ax, -np.degrees(np.arctan2(V[1, 1], V[0, 1])), 0, 360, col, 1, cv2.LINE_AA)
                cv2.circle(out, c, u(4), col, -1, cv2.LINE_AA)
        for side in ("near", "far"):
            p = st.players.get(side)
            if p is None or p["world"] is None:
                continue
            c = m.to_px(np.asarray(p["world"]))
            cv2.circle(out, c, u(8), GREEN, -1, cv2.LINE_AA)
            cv2.circle(out, c, u(8), (0, 0, 0), 1, cv2.LINE_AA)

    def _bottom_bar(self, out: np.ndarray, st: OverlayState, texts: list[TextItem]) -> None:
        u = self._u
        y0 = self.h - u(92)
        panel(out, 0, y0, self.w, self.h, 0.85)
        cols = [int(self.w * f) for f in (0.015, 0.27, 0.53, 0.78)]
        for x, side, name in ((cols[0], "near", "OYUNCU 1"), (cols[1], "far", "OYUNCU 2")):
            p = st.players.get(side) or {}
            label = st.names[side] if st.names and side in st.names else name
            where = "YAKIN" if side == "near" else "UZAK"
            texts.append(TextItem((x, y0 + u(12)), f"{label} · {where}", u(22), GREY))
            spd, dist, peak = p.get("speed_kmh"), p.get("dist_m"), p.get("peak_kmh")
            val = f"{num(spd)} km/sa  |  {dist:.0f} m  |  en yüksek {num(peak)}" if spd is not None else "—"
            texts.append(TextItem((x, y0 + u(42)), val, u(32), WHITE))
        texts.append(TextItem((cols[2], y0 + u(12)), "SON VURUŞ HIZI · 3B UYDURMA", u(22), GREY))
        ss = f"{st.last_shot_speed:.0f} km/sa" if st.last_shot_speed else "—"
        texts.append(TextItem((cols[2], y0 + u(42)), ss, u(32), WHITE))
        if st.score:
            texts.append(TextItem((cols[3], y0 + u(6)), "SKOR · YAYINDAN OKUNDU", u(18), GREY))
            for k, row in enumerate(st.score):
                yy = y0 + u(30) + k * u(28)
                if row.get("serving"):
                    cv2.circle(out, (cols[3] + u(7), yy + u(13)), u(5), YELLOW, -1, cv2.LINE_AA)
                games = "  ".join(str(g) for g in row["games"])
                pts = row.get("points") or ""
                texts.append(TextItem((cols[3] + u(20), yy), f"{row['name'][:12]:<12}", u(24), WHITE))
                texts.append(TextItem((cols[3] + u(210), yy), f"{games}   {pts}", u(24), WHITE))
        else:
            texts.append(TextItem((cols[3], y0 + u(12)), "SÜRE", u(22), GREY))
            texts.append(TextItem((cols[3], y0 + u(42)), f"{num(st.t)} sn", u(32), WHITE))

    def _var_card(self, out: np.ndarray, st: OverlayState, texts: list[TextItem]) -> None:
        c = st.latest_call
        if c is None:
            return
        u = self._u
        x0, y0, w, h = u(24), u(24), u(500), u(200)
        col = VERDICT_COLOR.get(c["verdict"], AMBER)
        panel(out, x0, y0, x0 + w, y0 + h, 0.82)
        cv2.rectangle(out, (x0, y0), (x0 + u(10), y0 + h), col, -1)
        texts.append(TextItem((x0 + u(26), y0 + u(12)), "SEKME · OTONOM ÇİZGİ KARARI", u(22), GREY))
        # Autonomous: below 95 % the most likely verdict is still given, marked as low confidence.
        if c["verdict"] == "REVIEW":
            verdict = ("İÇERİDE" if lean_in(c) else "DIŞARIDA") + " · DÜŞÜK GÜVEN"
        else:
            verdict = VERDICT_UPPER[c["verdict"]]
        texts.append(TextItem((x0 + u(26), y0 + u(38)), verdict, u(50), col))
        line1 = f"içeride olasılığı {prob(c['p_in'])}   ·   {line_name(c['line'])}"
        line2 = f"çizgiye mesafe {100 * c['margin_m']:+.0f} cm  ± {100 * c['sigma_m']:.0f} cm (1σ)"
        if c["verdict"] == "REVIEW" and c.get("play_continued"):
            line2 = "geometri belirsiz · karşı oyuncu topa vurdu, oyun sürdü"
        texts.append(TextItem((x0 + u(26), y0 + u(102)), line1, u(24), WHITE))
        texts.append(TextItem((x0 + u(26), y0 + u(134)), line2, u(24), WHITE))
        extra = []
        if c.get("speed_kmh"):
            extra.append(f"vuruş {c['speed_kmh']:.0f} km/sa")
        if c.get("net_clearance_m") is not None:
            sd = c.get("net_clearance_sd_m")
            extra.append(f"file üstü {c['net_clearance_m']:+.2f}".replace(".", ",")
                         + (f" ± {num(sd, 2)} m" if sd is not None else " m"))
        if extra:
            texts.append(TextItem((x0 + u(26), y0 + u(166)), "   ·   ".join(extra), u(22), GREY))

    def _health_badge(self, out: np.ndarray, st: OverlayState, texts: list[TextItem]) -> None:
        u = self._u
        x1, y0 = self.mini.x0 - u(16), self.mini.y0
        ok = st.calib_ok
        txt = "KORT KİLİTLİ" if ok else "KORT KİLİDİ YOK · ÖLÇÜM YAPILMIYOR"
        col = GREEN if ok else RED
        w = int(_font(u(24)).getlength(txt)) + u(60)
        panel(out, x1 - w, y0, x1, y0 + u(44), 0.8)
        cv2.circle(out, (x1 - w + u(22), y0 + u(22)), u(8), col, -1, cv2.LINE_AA)
        texts.append(TextItem((x1 - w + u(40), y0 + u(8)), txt, u(24), col))
        if st.calib_note:
            texts.append(TextItem((x1 - w + u(10), y0 + u(50)), st.calib_note, u(20), GREY))
