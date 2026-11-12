"""On-device text recognition with Apple's Vision framework (macOS only, optional).

``VNRecognizeTextRequest`` runs locally on the Neural Engine/GPU; nothing leaves the machine. When
the bindings (``pyobjc-framework-Vision``) are missing, :func:`available` is False and callers skip
the broadcast checks instead of failing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:  # pragma: no cover - depends on the platform
    import Vision
    from Foundation import NSData

    _HAVE_VISION = True
except ImportError:  # pragma: no cover
    _HAVE_VISION = False


@dataclass(frozen=True)
class TextBox:
    """Recognised text with its box in normalised image coordinates (origin top-left)."""

    text: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def x1(self) -> float:
        return self.x + self.w


def available() -> bool:
    return _HAVE_VISION


def recognize(img_bgr: np.ndarray, accurate: bool = True) -> list[TextBox]:
    """Text boxes in a BGR image; empty when Vision is unavailable or nothing is found."""
    if not _HAVE_VISION:
        return []
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelAccurate if accurate else Vision.VNRequestTextRecognitionLevelFast)
    req.setUsesLanguageCorrection_(False)  # names and scores are not dictionary words
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    if not handler.performRequests_error_([req], None)[0]:
        return []
    out = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        b = obs.boundingBox()  # normalised, origin bottom-left
        out.append(TextBox(str(cand[0].string()), float(cand[0].confidence()), float(b.origin.x),
                           float(1.0 - b.origin.y - b.size.height), float(b.size.width), float(b.size.height)))
    return out
