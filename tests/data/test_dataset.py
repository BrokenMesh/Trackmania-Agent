from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tmagent.data.dataset import WindowDataset, build_window, check_alignment
from tmagent.data.episode_io import EpisodeReader, load_episode, read_index, save_episode

from .test_helpers import decode_tick, small_cfg, write_dataset

R = 3
CHUNK = 8
KEYS = {"frames", "frame_valid", "hist_actions", "hist_valid", "target", "target_valid"}


def cfg_for(history_s: float, **kw):
    kw.setdefault("image_aug", False)
    return small_cfg(history_s=history_s, chunk_len=CHUNK, **kw)


@pytest.fixture
def root(tmp_path):
    cfg = cfg_for(0.0)
    write_dataset(tmp_path, cfg, {"mapA": 300, "mapB": 170, "mapC": 120})
    (tmp_path / "splits.json").write_text(
        json.dumps({"mapA": "train", "mapB": "train", "mapC": "val"})
    )
    return tmp_path


def episodes_of(root, split, cfg):
    from tmagent.data.split import filter_index

    return [
        load_episode(root / e["path"]) for e in filter_index(read_index(root), split, cfg, root)
    ]


def test_k1_shapes_dtypes_and_length(root):
    cfg = cfg_for(0.0)
    assert cfg.num_steps == 1
    ds = WindowDataset(str(root), cfg, "train", train=False)
    eps = episodes_of(root, "train", cfg)
    assert len(ds) == sum(len(e.frames) for e in eps)
    s = ds[10]
    assert set(s) == KEYS
    assert all(isinstance(v, torch.Tensor) for v in s.values())
    h, w = cfg.resolution[1], cfg.resolution[0]
    assert s["frames"].shape == (1, 3, h, w) and s["frames"].dtype == torch.uint8
    assert s["frame_valid"].shape == (1,) and s["frame_valid"].dtype == torch.bool
    assert s["hist_actions"].shape == (1, R, 3) and s["hist_actions"].dtype == torch.float32
    assert s["hist_valid"].shape == (1,) and s["hist_valid"].dtype == torch.bool
    assert s["target"].shape == (1, CHUNK, 3) and s["target"].dtype == torch.float32
    assert s["target_valid"].shape == (1, CHUNK) and s["target_valid"].dtype == torch.bool


def test_k1_window_content_matches_episode_slices(root):
    cfg = cfg_for(0.0)
    ds = WindowDataset(str(root), cfg, "train", train=False)
    eps = episodes_of(root, "train", cfg)
    i = 0
    for e, ep in enumerate(eps):
        for fi in range(len(ep.frames)):
            s = ds[i]
            assert ds.locate(i) == (e, fi)
            i += 1
            assert np.array_equal(s["frames"][0].numpy(), ep.frames[fi].transpose(2, 0, 1))
            assert s["frame_valid"].all()
            lo = fi * R
            tgt = ep.actions[lo : lo + CHUNK]
            assert np.array_equal(s["target"][0, : len(tgt)].numpy(), tgt)
            assert not s["target"][0, len(tgt) :].any()
            assert s["target_valid"][0].tolist() == [True] * len(tgt) + [False] * (CHUNK - len(tgt))
            if fi == 0:
                assert not s["hist_valid"][0] and not s["hist_actions"].any()
            else:
                assert s["hist_valid"][0]
                assert np.array_equal(s["hist_actions"][0].numpy(), ep.actions[lo - R : lo])
    assert i == len(ds)


def test_end_of_episode_target_padding(root):
    cfg = cfg_for(0.0)
    ds = WindowDataset(str(root), cfg, "train", train=False)
    ep = episodes_of(root, "train", cfg)[0]
    t_a, last = len(ep.actions), len(ep.frames) - 1
    s = ds[last]  # last frame of the first episode
    n_valid = min(CHUNK, t_a - last * R)
    assert s["target_valid"][0].sum() == n_valid
    assert 1 <= n_valid < CHUNK  # the stub episode really ends inside the chunk
    assert not s["target"][0, n_valid:].any()


def test_k41_left_padding_and_alignment(root):
    cfg = cfg_for(2.0)
    k = cfg.num_steps
    assert k == 41
    ds = WindowDataset(str(root), cfg, "train", train=False)
    ep = episodes_of(root, "train", cfg)[0]
    for k_end in (0, 1, 3, 40, 41, len(ep.frames) - 1):
        s = ds[k_end]
        assert s["frames"].shape == (k, 3, 24, 32)
        assert s["hist_actions"].shape == (k, R, 3) and s["target"].shape == (k, CHUNK, 3)
        for step in range(k):
            fi = k_end - (k - 1) + step
            if fi < 0:
                assert not s["frame_valid"][step] and not s["hist_valid"][step]
                assert not s["target_valid"][step].any()
                assert not s["frames"][step].any() and not s["hist_actions"][step].any()
                assert not s["target"][step].any()
                continue
            assert s["frame_valid"][step]
            assert s["hist_valid"][step] == (fi >= 1)
            assert np.array_equal(s["frames"][step].numpy(), ep.frames[fi].transpose(2, 0, 1))
            if fi >= 1:
                assert np.array_equal(
                    s["hist_actions"][step].numpy(), ep.actions[fi * R - R : fi * R]
                )
            else:
                assert not s["hist_actions"][step].any()
            tgt = ep.actions[fi * R : fi * R + CHUNK]
            assert np.array_equal(s["target"][step, : len(tgt)].numpy(), tgt)
            assert s["target_valid"][step].sum() == len(tgt)


def test_frame_action_sync_through_the_whole_pipeline(root):
    """Frame at step k encodes tick t; target[k, 0] must be the timeline action of tick t."""
    from .test_helpers import make_timeline

    cfg = cfg_for(2.0)
    tl = make_timeline(300)
    ds = WindowDataset(str(root), cfg, "train", train=False)
    k = cfg.num_steps
    for idx in (0, 7, 20, 41, 59):  # mapA is the first episode (60 frames)
        s = ds[idx]
        for step in range(k):
            if not s["frame_valid"][step]:
                continue
            tick = decode_tick(s["frames"][step].permute(1, 2, 0).numpy())
            assert np.array_equal(s["target"][step, 0].numpy(), tl.actions[tick])
            # history = the R actions held in the preceding 1/frame_hz seconds
            if s["hist_valid"][step]:
                prev_ticks = (np.rint((np.arange(R) - R) * 1000 / 60) + tick * 10) // 10
                got = s["hist_actions"][step].numpy()
                assert np.array_equal(got, tl.actions[prev_ticks.astype(int)])


def test_item_reads_only_its_window_frames(root, monkeypatch):
    calls: list[list[int]] = []
    orig = EpisodeReader.frames

    def spy(self, indices):
        calls.append(np.asarray(indices).tolist())
        return orig(self, indices)

    monkeypatch.setattr(EpisodeReader, "frames", spy)
    ds = WindowDataset(str(root), cfg_for(2.0), "train", train=False)
    ds[50]  # 41 steps ending at frame 50 of the first episode
    assert calls == [list(range(10, 51))]
    calls.clear()
    ds[3]  # steps before the episode start are padding: not read at all
    assert calls == [[0, 1, 2, 3]]
    calls.clear()
    WindowDataset(str(root), cfg_for(0.0), "train", train=False)[7]
    assert calls == [[7]]


def test_dataset_caches_readers_not_frames(root):
    ds = WindowDataset(str(root), cfg_for(0.0), "train", train=False)
    ds[0], ds[1], ds[70]
    assert all(isinstance(r, EpisodeReader) for r in ds._cache._items.values())
    assert len(ds._cache) == 2  # mapA and mapB


def test_stride_subsamples_k_end(root):
    cfg = cfg_for(0.5)
    full = WindowDataset(str(root), cfg, "train", train=False)
    ds = WindowDataset(str(root), cfg, "train", train=False, stride=4)
    eps = episodes_of(root, "train", cfg)
    assert len(ds) == sum(-(-len(e.frames) // 4) for e in eps)
    assert [ds.locate(i)[1] for i in range(-(-len(eps[0].frames) // 4))] == list(
        range(0, len(eps[0].frames), 4)
    )
    s, f = ds[3], full[12]  # second item of mapA with stride 4 == frame 12
    for key in KEYS:
        assert torch.equal(s[key], f[key])


def test_splits_are_separate(root):
    cfg = cfg_for(0.0)
    train = WindowDataset(str(root), cfg, "train", train=False)
    val = WindowDataset(str(root), cfg, "val", train=False)
    test = WindowDataset(str(root), cfg, "test", train=False)
    assert {e["map_uid"] for e in train.entries} == {"mapA", "mapB"}
    assert {e["map_uid"] for e in val.entries} == {"mapC"}
    assert len(test) == 0
    assert len(val) == len(episodes_of(root, "val", cfg_for(0.0))[0].frames) == 24
    with pytest.raises(IndexError):
        test[0]


def test_index_errors(root):
    ds = WindowDataset(str(root), cfg_for(0.0), "val", train=False)
    n = len(ds)
    assert torch.equal(ds[-1]["frames"], ds[n - 1]["frames"])
    with pytest.raises(IndexError):
        ds[n]
    with pytest.raises(IndexError):
        ds[-n - 1]
    with pytest.raises(ValueError):
        WindowDataset(str(root), cfg_for(0.0), "val", train=False, stride=0)


def test_config_mismatch_with_index_is_rejected(root):
    with pytest.raises(ValueError, match="disagrees"):
        WindowDataset(str(root), cfg_for(0.0, frame_hz=30), "train", train=False)
    with pytest.raises(ValueError, match="disagrees"):
        WindowDataset(str(root), cfg_for(0.0, resolution=[64, 48]), "train", train=False)


def test_misaligned_episode_is_rejected_on_load(root):
    cfg = cfg_for(0.0)
    entry = read_index(root)[0]
    path = root / entry["path"]
    ep = load_episode(path)
    ep.action_times_ms = ep.action_times_ms + 1
    save_episode(ep, root)
    ds = WindowDataset(str(root), cfg, "train", train=False)
    with pytest.raises(ValueError, match="action_times_ms"):
        ds[0]
    with pytest.raises(ValueError, match="action_times_ms"):
        check_alignment(ep, cfg)
    ep.action_times_ms = ep.action_times_ms - 1
    ep.frame_times_ms = ep.frame_times_ms[::-1].copy()
    with pytest.raises(ValueError, match="frame_times_ms"):
        check_alignment(ep, cfg)
    ep.frame_times_ms = ep.frame_times_ms[::-1].copy()
    ep.actions, ep.action_times_ms = ep.actions[:10], ep.action_times_ms[:10]
    with pytest.raises(ValueError, match="too few"):
        check_alignment(ep, cfg)


def test_train_augmentation_changes_history_but_never_targets(root):
    cfg = cfg_for(2.0, action_dropout=0.5, action_noise_steer=0.1, image_aug=True)
    clean = WindowDataset(str(root), cfg, "train", train=False)
    aug = WindowDataset(str(root), cfg, "train", train=True)
    assert len(clean) == len(aug)
    changed = 0
    for i in range(0, len(clean), 3):
        a, b = clean[i], aug[i]
        for key in ("target", "target_valid", "frame_valid"):
            assert torch.equal(a[key], b[key]), key
        assert b["frames"].dtype == torch.uint8 and b["frames"].shape == a["frames"].shape
        assert b["hist_actions"].dtype == torch.float32
        assert not b["hist_actions"][~b["hist_valid"]].any()
        changed += not torch.equal(a["hist_actions"], b["hist_actions"])
    assert changed > 0


def test_build_window_is_pure_numpy_and_matches_dataset(root):
    cfg = cfg_for(1.0)
    ds = WindowDataset(str(root), cfg, "train", train=False)
    ep = episodes_of(root, "train", cfg)[0]
    w = build_window(ep, 5, cfg)
    s = ds[5]
    assert set(w) == KEYS
    for key in KEYS:
        assert np.array_equal(w[key], s[key].numpy())


def test_dataset_pickles_without_cache(root):
    ds = WindowDataset(str(root), cfg_for(0.0), "train", train=False)
    ds[0]
    assert ds._cache is not None and len(ds._cache) == 1
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._cache is None and len(clone) == len(ds)
    assert torch.equal(clone[3]["frames"], ds[3]["frames"])


@pytest.mark.parametrize("train", [False, True])
def test_dataloader_with_workers(root, train):
    cfg = cfg_for(2.0, image_aug=train)
    ds = WindowDataset(str(root), cfg, "train", train=train)
    ds[0]  # warm the main-process cache first: workers must not inherit it or its RNG state
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, drop_last=False)
    n = 0
    seen_ticks: list[int] = []
    for batch in dl:
        b = batch["frames"].shape[0]
        n += b
        assert batch["frames"].shape == (b, 41, 3, 24, 32) and batch["frames"].dtype == torch.uint8
        assert batch["frame_valid"].shape == (b, 41) and batch["frame_valid"].dtype == torch.bool
        assert batch["hist_actions"].shape == (b, 41, R, 3)
        assert batch["hist_valid"].shape == (b, 41)
        assert batch["target"].shape == (b, 41, CHUNK, 3)
        assert batch["target_valid"].shape == (b, 41, CHUNK)
        assert batch["frame_valid"][:, -1].all()
        if not train:  # un-augmented: the current frame of every window is the right one
            seen_ticks += [decode_tick(f.permute(1, 2, 0).numpy()) for f in batch["frames"][:, -1]]
    assert n == len(ds)
    if not train:
        want = np.concatenate([e.frame_times_ms // 10 for e in episodes_of(root, "train", cfg)])
        assert sorted(seen_ticks) == sorted(want.tolist())  # every window exactly once


def test_per_process_state_is_rebuilt_in_a_new_process(root, monkeypatch):
    ds = WindowDataset(str(root), cfg_for(0.0), "train", train=True)
    cache, rng = ds._process_state()
    assert ds._process_state() == (cache, rng)
    monkeypatch.setattr("tmagent.data.dataset.os.getpid", lambda: 4242)  # as in a forked worker
    cache2, rng2 = ds._process_state()
    assert cache2 is not cache and rng2 is not rng


def test_dataloader_workers_augment(root):
    cfg = cfg_for(0.0, action_dropout=0.5, action_token_dropout=0.0, action_noise_steer=0.0)
    ds = WindowDataset(str(root), cfg, "train", train=True)
    dl = DataLoader(ds, batch_size=len(ds) // 2, num_workers=2)
    batches = list(dl)
    assert len(batches) >= 2
    # both workers produce a mix of dropped / kept histories, i.e. their RNGs are live
    for b in batches[:2]:
        assert 0 < b["hist_valid"].float().mean() < 1
