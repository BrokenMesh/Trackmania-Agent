from __future__ import annotations

import numpy as np

from tmagent.data.augment import augment_frames, augment_sample

from .test_helpers import small_cfg

K, R, C, H, W, L = 6, 3, 3, 24, 32, 4


def make_sample(seed: int = 0, n_invalid: int = 2) -> dict[str, np.ndarray]:
    """A window with the first n_invalid steps padded, like an episode start."""
    rng = np.random.default_rng(seed)
    frame_valid = np.arange(K) >= n_invalid
    frames = rng.integers(0, 256, size=(K, C, H, W), dtype=np.uint8)
    frames[~frame_valid] = 0
    hist_valid = np.arange(K) >= n_invalid + 1
    hist = np.stack(
        [
            rng.uniform(-1, 1, (K, R)),
            rng.integers(0, 2, (K, R)),
            rng.integers(0, 2, (K, R)),
        ],
        axis=-1,
    ).astype(np.float32)
    hist[~hist_valid] = 0
    target = np.stack(
        [rng.uniform(-1, 1, (K, L)), rng.integers(0, 2, (K, L)), rng.integers(0, 2, (K, L))],
        axis=-1,
    ).astype(np.float32)
    target_valid = np.ones((K, L), dtype=bool) & frame_valid[:, None]
    target[~target_valid] = 0
    return {
        "frames": frames,
        "frame_valid": frame_valid,
        "hist_actions": hist,
        "hist_valid": hist_valid,
        "target": target,
        "target_valid": target_valid,
    }


def noop_cfg(**kw):
    base = dict(
        action_dropout=0.0,
        action_token_dropout=0.0,
        action_noise_steer=0.0,
        action_flip_prob=0.0,
        image_aug=False,
    )
    return small_cfg(**{**base, **kw})


def copy_of(sample):
    return {k: v.copy() for k, v in sample.items()}


def test_all_off_is_identity():
    s = make_sample()
    out = augment_sample(s, noop_cfg(), np.random.default_rng(0))
    for k in s:
        assert np.array_equal(out[k], s[k]), k


def test_never_touches_targets_or_validity_of_frames_and_never_mutates_input():
    cfg = small_cfg(
        action_dropout=0.5,
        action_token_dropout=0.5,
        action_noise_steer=0.5,
        action_flip_prob=0.5,
        image_aug=True,
    )
    for seed in range(40):
        s = make_sample(seed)
        before = copy_of(s)
        out = augment_sample(s, cfg, np.random.default_rng(seed))
        for k in s:  # input untouched
            assert np.array_equal(s[k], before[k]), k
        for k in ("target", "target_valid", "frame_valid"):
            assert np.array_equal(out[k], before[k]), k
        assert out["frames"].dtype == np.uint8 and out["frames"].shape == (K, C, H, W)
        assert out["hist_actions"].dtype == np.float32
        assert out["hist_valid"].dtype == bool
        # invalid history is always zeroed, never revived
        assert not (out["hist_valid"] & ~before["hist_valid"]).any()
        assert not out["hist_actions"][~out["hist_valid"]].any()
        # padded frames stay zero
        assert not out["frames"][~before["frame_valid"]].any()


def test_action_dropout_one_drops_whole_history():
    out = augment_sample(make_sample(), noop_cfg(action_dropout=1.0), np.random.default_rng(0))
    assert not out["hist_valid"].any() and not out["hist_actions"].any()


def test_token_dropout_one_drops_every_step_and_partial_dropout_is_per_step():
    s = make_sample(n_invalid=0)
    out = augment_sample(s, noop_cfg(action_token_dropout=1.0), np.random.default_rng(0))
    assert not out["hist_valid"].any()
    kept = []
    for seed in range(200):
        o = augment_sample(
            make_sample(n_invalid=0),
            noop_cfg(action_token_dropout=0.3),
            np.random.default_rng(seed),
        )
        kept.append(o["hist_valid"])
        assert np.array_equal(
            o["hist_actions"][o["hist_valid"]], s["hist_actions"][o["hist_valid"]]
        )
    rate = 1 - np.mean(np.array(kept)[:, s["hist_valid"]])  # only steps that had a history
    assert 0.25 < rate < 0.35
    # steps are dropped independently: some windows are only partially dropped
    assert any(0 < k.sum() < s["hist_valid"].sum() for k in kept)


def test_action_dropout_rate():
    drops = [
        not augment_sample(
            make_sample(n_invalid=0), noop_cfg(action_dropout=0.3), np.random.default_rng(i)
        )["hist_valid"].any()
        for i in range(400)
    ]
    assert 0.22 < np.mean(drops) < 0.38


def test_steer_noise_clipped_and_only_on_steer():
    s = make_sample(n_invalid=0)
    out = augment_sample(s, noop_cfg(action_noise_steer=0.2), np.random.default_rng(1))
    assert not np.array_equal(out["hist_actions"][..., 0], s["hist_actions"][..., 0])
    assert np.abs(out["hist_actions"][..., 0]).max() <= 1.0
    assert np.array_equal(out["hist_actions"][..., 1:], s["hist_actions"][..., 1:])
    diff = out["hist_actions"][..., 0] - s["hist_actions"][..., 0]
    assert 0.05 < diff.std() < 0.3
    huge = augment_sample(s, noop_cfg(action_noise_steer=50.0), np.random.default_rng(1))
    assert np.abs(huge["hist_actions"][..., 0]).max() <= 1.0


def test_flip_prob_flips_only_gas_and_brake():
    s = make_sample(n_invalid=0)
    out = augment_sample(s, noop_cfg(action_flip_prob=1.0), np.random.default_rng(0))
    v = s["hist_valid"]
    assert np.array_equal(out["hist_actions"][..., 0], s["hist_actions"][..., 0])
    assert np.array_equal(out["hist_actions"][v][..., 1:], 1.0 - s["hist_actions"][v][..., 1:])
    assert not out["hist_actions"][~v].any()  # invalid steps stay zero
    flips = []
    for i in range(50):
        o = augment_sample(s, noop_cfg(action_flip_prob=0.1), np.random.default_rng(i))
        flips.append((o["hist_actions"][v][..., 1:] != s["hist_actions"][v][..., 1:]).mean())
    assert 0.07 < np.mean(flips) < 0.13


def test_image_aug_same_transform_for_all_frames_and_changes_pixels():
    base = np.tile(np.linspace(20, 230, W, dtype=np.float32)[None, None, :], (C, H, 1))
    frames = np.broadcast_to(base.astype(np.uint8), (K, C, H, W)).copy()
    valid = np.ones(K, dtype=bool)
    changed = 0
    for seed in range(20):
        out = augment_frames(frames, valid, np.random.default_rng(seed))
        assert out.dtype == np.uint8 and out.shape == frames.shape
        for k in range(1, K):  # identical input frames -> identical output frames
            assert np.array_equal(out[k], out[0])
        changed += not np.array_equal(out[0], frames[0])
    assert changed >= 18


def test_image_aug_shift_bound_edge_replication_and_jitter_range():
    img = np.zeros((1, 1, 21, 21), dtype=np.uint8)
    img[..., 10, 10] = 100  # single bright pixel in the centre
    valid = np.ones(1, dtype=bool)
    shifts = set()
    for seed in range(300):
        out = augment_frames(img, valid, np.random.default_rng(seed))
        ys, xs = np.nonzero(out[0, 0] == out[0, 0].max())
        dy, dx = int(ys[0]) - 10, int(xs[0]) - 10
        assert abs(dy) <= 4 and abs(dx) <= 4
        shifts.add((dy, dx))
        # brightness/contrast jitter stays within +-10% each: bright pixel scales by <= ~1.21
        assert out.max() <= 100 * 1.1 * 1.1 + 2
    assert len(shifts) > 20  # shifts are spread over the +-4 px range
    # edge replication: a horizontal ramp shifted right keeps the border value, no zero fill
    ramp = np.tile((60 + 5 * np.arange(21)).astype(np.uint8)[None, None, None, :], (1, 1, 21, 1))
    for seed in range(50):
        out = augment_frames(ramp, valid, np.random.default_rng(seed))
        assert out.min() > 0


def test_padded_frames_stay_zero_and_do_not_bias_jitter():
    frames = np.zeros((K, C, H, W), dtype=np.uint8)
    frames[3:] = 200
    valid = np.arange(K) >= 3
    for seed in range(20):
        out = augment_frames(frames, valid, np.random.default_rng(seed))
        assert not out[:3].any()
        assert out[3:].min() >= 170 and out[3:].max() <= 255  # 200 +-(10% contrast/brightness)
    assert not augment_frames(
        np.zeros((K, C, H, W), np.uint8), np.zeros(K, bool), np.random.default_rng(0)
    ).any()


def test_deterministic_for_a_seed():
    cfg = small_cfg(image_aug=True)
    a = augment_sample(make_sample(), cfg, np.random.default_rng(7))
    b = augment_sample(make_sample(), cfg, np.random.default_rng(7))
    for k in a:
        assert np.array_equal(a[k], b[k]), k


class FixedRng:
    """Neutral jitter (x1.0) and a fixed (dy, dx) shift."""

    def __init__(self, shift: tuple[int, int]):
        self.shift = shift

    def uniform(self, low, high, size=None):
        return np.ones(size)

    def integers(self, low, high, size=None):
        return np.array(self.shift)


def test_shift_matches_numpy_edge_pad_reference():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(K, C, H, W), dtype=np.uint8)
    valid = np.ones(K, dtype=bool)
    padded = np.pad(frames, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="edge")
    for dy, dx in [(0, 0), (4, -4), (-3, 2), (-4, 4), (1, 0), (0, -1)]:
        out = augment_frames(frames, valid, FixedRng((dy, dx)))
        want = padded[:, :, 4 + dy : 4 + dy + H, 4 + dx : 4 + dx + W]
        assert np.array_equal(out, want), (dy, dx)
    assert frames.dtype == np.uint8
