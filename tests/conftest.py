from __future__ import annotations

import numpy as np
import pytest

from courtvision.calib.camera import Camera


def look_at_camera(center: np.ndarray, target: np.ndarray, f: float, size: tuple[int, int]) -> Camera:
    """Pinhole camera at ``center`` looking at ``target`` (world z up, image y down)."""
    fwd = target - center
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # rows: camera axes in world coordinates
    t = -R @ center
    K = np.array([[f, 0.0, size[0] / 2], [0.0, f, size[1] / 2], [0.0, 0.0, 1.0]])
    return Camera(K=K, R=R, t=t)


@pytest.fixture
def broadcast_camera() -> Camera:
    """Typical TV main camera: high behind the near baseline, 1920x1080."""
    return look_at_camera(np.array([0.0, -28.0, 11.0]), np.array([0.0, 1.5, 0.0]), 1750.0, (1920, 1080))
