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


def test_motion_blur_never_moves_the_ball_away_from_its_label():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_ball import motion_blur

    yy, xx = np.mgrid[0:41, 0:41]
    for seed in range(100):
        img = np.zeros((41, 41, 3), np.float32)
        img[20, 20] = 1.0
        out = motion_blur(img, np.random.default_rng(seed))[..., 0]
        assert out.sum() == np.float32(1.0) or abs(out.sum() - 1.0) < 1e-4  # brightness kept
        cx, cy = (out * xx).sum() / out.sum(), (out * yy).sum() / out.sum()
        assert abs(cx - 20) < 0.05 and abs(cy - 20) < 0.05, (seed, cx, cy)  # centre unmoved


def test_training_state_resumes_exactly_and_refuses_another_recipe(tmp_path):
    import sys
    from pathlib import Path

    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_ball import load_state, recipe, save_state, state_path

    args = {"epochs": 2, "batch": 4, "lr": 1e-2, "train_games": 7, "blur": 0.3, "centre_labels": True,
            "out": "x", "resume": True}
    rec = recipe(args, steps_per_epoch=3)
    xs = torch.randn(6, 4, 3)

    def run(steps: range, net, opt, sched) -> None:
        for i in steps:
            loss = net(xs[i]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    def fresh():
        torch.manual_seed(1)
        net = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(net.parameters(), lr=args["lr"])
        return net, opt, torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args["lr"], total_steps=6)

    straight = fresh()
    run(range(6), *straight)

    first = fresh()
    run(range(3), *first)
    path = state_path(tmp_path / "ball.pt")
    assert path.name == "ball.state.pt"
    save_state(path, *first, epoch=1, best=0.5, step=3, history=[{"epoch": 1}], recipe=rec)
    second = fresh()  # a new process: same init, nothing trained
    assert load_state(path, *second, rec) == (1, 0.5, 3, [{"epoch": 1}])
    run(range(3, 6), *second)
    for a, b in zip(straight[0].parameters(), second[0].parameters(), strict=True):
        assert torch.equal(a, b)  # optimizer moments and the one-cycle position came back too

    with pytest.raises(SystemExit):
        load_state(path, *fresh(), {**rec, "epochs": 3})
    with pytest.raises(SystemExit):  # another clip set: the one-cycle schedule would not line up
        load_state(path, *fresh(), {**rec, "steps_per_epoch": 4})
    old = state_path(tmp_path / "old.pt")  # written before steps_per_epoch was recorded
    save_state(old, *fresh(), epoch=1, best=0.5, step=3, history=[],
               recipe={k: v for k, v in rec.items() if k != "steps_per_epoch"})
    assert load_state(old, *fresh(), rec)[0] == 1
    older = state_path(tmp_path / "older.pt")  # written before the noise augmentation existed: it had none
    save_state(older, *fresh(), epoch=1, best=0.5, step=3, history=[],
               recipe={k: v for k, v in rec.items() if k != "noise"})
    assert rec["noise"] == 0.0 and load_state(older, *fresh(), rec)[0] == 1
    with pytest.raises(SystemExit):
        load_state(older, *fresh(), {**rec, "noise": 0.3})


def test_sensor_noise_keeps_the_frame_in_range_and_the_ball_where_it_was():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_ball import sensor_noise

    img = np.full((40, 40, 3), 0.5, np.float32)
    img[20, 20] = 1.0
    out = sensor_noise(img, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert 0.003 < float(np.std(out[:10, :10] - img[:10, :10])) < 0.07  # some noise, at most ~15/255
    assert abs(float(out.mean()) - float(img.mean())) < 0.01  # no brightness shift
    assert not np.array_equal(out, sensor_noise(img, np.random.default_rng(1)))  # differs frame to frame
