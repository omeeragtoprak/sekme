import numpy as np
import torch

from courtvision.models.pose_unet import ARCH, FLIP, IN_H, IN_W, KEYPOINTS, PoseUNet, crop_box, pose_targets


def test_pose_unet_shapes_and_flip_table():
    net = PoseUNet().eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, IN_H, IN_W))
    assert out.shape == (1, KEYPOINTS, IN_H, IN_W)
    assert sorted(FLIP) == list(range(KEYPOINTS))
    assert all(FLIP[FLIP[i]] == i for i in range(KEYPOINTS))  # a flip undone by a flip
    assert FLIP[9] == 10 and FLIP[5] == 6 and FLIP[0] == 0  # wrists, shoulders swap; nose stays
    assert pose_targets([None] * KEYPOINTS).sum() == 0


def test_crop_box_matches_the_production_crop_and_rejects_tiny_boxes():
    assert crop_box(np.array([100.0, 100.0, 140.0, 200.0]), 1280, 720) == (76, 65, 164, 210)
    assert crop_box(np.array([0.0, 0.0, 2.0, 2.0]), 1280, 720) is None


def test_estimator_uses_our_checkpoint(tmp_path):
    from courtvision.perception.pose import SekmePoseEstimator, pose_estimator

    path = tmp_path / "pose.pt"
    torch.save({"arch": ARCH, "model": PoseUNet().state_dict()}, path)
    est = pose_estimator(str(path), "cpu")
    assert isinstance(est, SekmePoseEstimator)
    pose = est.estimate(np.zeros((720, 1280, 3), np.uint8), np.array([600.0, 300.0, 660.0, 450.0]))
    assert pose is not None and pose.kps.shape == (KEYPOINTS, 2)
