import numpy as np
import torch

from courtvision.models.ball_unet import IN_H, IN_W, BallUNet, disk_targets, wbce_loss


def test_unet_maps_three_frames_to_three_full_size_heatmaps():
    net = BallUNet().eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 9, IN_H, IN_W))
    assert out.shape == (2, 3, IN_H, IN_W)
    assert 1.5e6 < sum(p.numel() for p in net.parameters()) < 3e6


def test_disk_targets_mark_the_ball_and_leave_missing_frames_empty():
    t = disk_targets([(100.0, 50.0), None, (0.0, 0.0)], w=200, h=100, radius=2.5)
    assert t.shape == (3, 100, 200)
    assert t[0, 50, 100] == 1 and t[0, 50, 103] == 0 and t[0].sum() == 21  # radius-2.5 disk: 21 px
    assert t[1].sum() == 0
    assert t[2, 0, 0] == 1


def test_wbce_is_low_for_a_correct_map_and_high_for_an_inverted_one():
    target = torch.from_numpy(disk_targets([(20.0, 10.0)], w=40, h=20))[None]
    good = torch.where(target > 0, 8.0, -8.0)
    assert wbce_loss(good, target) < 1e-4 < wbce_loss(-good, target)
    assert np.isfinite(float(wbce_loss(torch.zeros_like(target), target)))


def test_detector_loads_our_checkpoint_as_the_unet(tmp_path):
    from courtvision.models.ball_unet import ARCH
    from courtvision.perception.ball import BallDetector

    path = tmp_path / "ball.pt"
    torch.save({"arch": ARCH, "model": BallUNet().state_dict()}, path)
    det = BallDetector(str(path), torch.device("cpu"))
    assert det.arch == ARCH
    frame = np.zeros((720, 1280, 3), np.uint8)
    hm = det.heatmaps([det._prep(frame)] * 3)
    assert hm.shape == (3, IN_H, IN_W)


def test_window_stride_spaces_the_three_frames_and_keeps_three_votes(tmp_path, monkeypatch):
    from courtvision.models.ball_unet import ARCH
    from courtvision.perception.ball import BallDetector

    path = tmp_path / "ball.pt"
    torch.save({"arch": ARCH, "model": BallUNet().state_dict()}, path)
    det = BallDetector(str(path), torch.device("cpu"))
    monkeypatch.setattr(det, "_prep", lambda frame: np.full((1, 2, 2), float(frame)))
    windows = []

    def fake_heatmaps(prepped):
        windows.append([int(p[0, 0, 0]) for p in prepped])
        return np.zeros((3, IN_H, IN_W), np.float32)

    monkeypatch.setattr(det, "heatmaps", fake_heatmaps)
    det.start((1280, 720), stride=2)
    for i in range(10):
        det.push(i, i)
    det.finish(10)
    assert windows[0] == [0, 2, 4] and windows[-1] == [5, 7, 9]
    votes = {j: sum(j in w for w in windows) for j in range(10)}
    assert all(votes[j] == 3 for j in range(4, 6))
