import numpy as np
import torch

from courtvision.geometry.court import CourtModel
from courtvision.models.court_unet import ARCH, IN_H, IN_W, KEYPOINTS, CourtUNet, court_targets, flip_points


def test_court_unet_maps_a_frame_to_fourteen_heatmaps():
    net = CourtUNet().eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, IN_H, IN_W))
    assert out.shape == (1, KEYPOINTS, IN_H, IN_W)
    assert torch.sigmoid(out).max() < 0.05  # starts at the 1 % prior


def test_flip_keeps_each_keypoint_meaning():
    # Mirror the court model's own keypoints: flipped index i must land where index i is (x -> -x).
    world = CourtModel().keypoints()
    pts = [(IN_W / 2 + 20 * x, IN_H / 2 - 10 * y) for x, y in world]
    flipped = flip_points(pts)
    mirrored_world = [(IN_W - 1 - (IN_W / 2 + 20 * x), IN_H / 2 - 10 * y) for x, y in -world * [1, -1]]
    np.testing.assert_allclose(np.array(flipped), np.array(mirrored_world), atol=1e-9)
    assert flip_points(flipped) == pts


def test_targets_and_detector_loading(tmp_path):
    from courtvision.perception.court import CourtKeypointDetector

    t = court_targets([(10.0, 10.0)] + [None] * (KEYPOINTS - 1))
    assert t.shape == (KEYPOINTS, IN_H, IN_W) and t[0, 10, 10] == 1 and t[1:].sum() == 0
    path = tmp_path / "court.pt"
    torch.save({"arch": ARCH, "model": CourtUNet().state_dict()}, path)
    det = CourtKeypointDetector(str(path), torch.device("cpu"))
    kps, scores = det.detect_batch([np.zeros((720, 1280, 3), np.uint8)])
    assert kps.shape == (1, KEYPOINTS, 2) and np.isnan(kps).all()  # untrained: nothing above 0.5
