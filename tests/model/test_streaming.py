"""StreamingPolicy: equivalence with TMPolicy.forward on the dataset-style window."""

from __future__ import annotations

import numpy as np
import torch

from tmagent.config import Config, config_to_dict
from tmagent.interfaces import ChunkPolicy
from tmagent.model import StreamingPolicy, TMPolicy, load_policy, load_streaming_policy


def make_cfg(head="regression", use_actions=True) -> Config:
    cfg = Config()
    cfg.data.resolution = [32, 24]
    cfg.data.history_s = 0.2  # K = 5
    cfg.data.chunk_len = 4
    m = cfg.model
    m.d_model, m.n_layers, m.n_heads, m.tokens_per_frame, m.dropout = 32, 2, 4, 4, 0.0
    m.head, m.use_action_history, m.steer_bins = head, use_actions, 7
    cfg.validate()
    return cfg


def make_model(cfg: Config) -> TMPolicy:
    torch.manual_seed(0)
    model = TMPolicy(cfg.model, cfg.data)
    torch.nn.init.normal_(model.head[-1].weight, std=0.5)
    return model.eval()


def window_batch(frames, past, t, k, r):
    """Dataset-style batch (B=1) for the window of k steps ending at observation t."""
    n, (h, w, c) = len(frames), frames[0].shape
    fr = torch.zeros(1, k, c, h, w, dtype=torch.uint8)
    hist = torch.zeros(1, k, r, 3)
    fv = torch.zeros(1, k, dtype=torch.bool)
    hv = torch.zeros(1, k, dtype=torch.bool)
    for j in range(k):
        i = t - k + 1 + j
        if i < 0:
            continue  # left padding before the first observation
        fr[0, j] = torch.from_numpy(frames[i]).permute(2, 0, 1)
        fv[0, j] = True
        if i > 0:  # history of the first observation lies before the episode start
            hist[0, j] = torch.from_numpy(past[i])
            hv[0, j] = True
    assert n > t
    return {"frames": fr, "frame_valid": fv, "hist_actions": hist, "hist_valid": hv}


def run_equivalence(cfg: Config, binarize: bool) -> None:
    model = make_model(cfg)
    sp = StreamingPolicy(model, cfg.data, "cpu", "fp32", binarize=binarize)
    assert isinstance(sp, ChunkPolicy) and sp.chunk_len == cfg.data.chunk_len
    k, r = cfg.data.num_steps, cfg.data.actions_per_frame
    w, h = cfg.data.resolution
    rng = np.random.default_rng(0)
    steps = k + 4  # more than K: exercises the ring buffer wrap-around
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(steps)]
    past = [rng.random((r, 3)).astype(np.float32) for _ in range(steps)]
    for t in range(steps):
        sp.observe(frames[t], past[t])  # past[0] is non-zero but must be treated as invalid
        got = sp.predict()
        assert got.dtype == np.float32 and got.shape == (cfg.data.chunk_len, 3)
        with torch.no_grad():
            batch = window_batch(frames, past, t, k, r)
            want = model.decode(model(batch), binarize=binarize)[0, -1].numpy()
        np.testing.assert_allclose(got, want, atol=1e-4, err_msg=f"t={t}")


def test_streaming_matches_forward_regression():
    run_equivalence(make_cfg("regression"), binarize=False)


def test_streaming_matches_forward_discrete():
    run_equivalence(make_cfg("discrete"), binarize=False)


def test_streaming_matches_forward_no_actions():
    run_equivalence(make_cfg(use_actions=False), binarize=False)


def test_streaming_binarized_matches_forward():
    run_equivalence(make_cfg(), binarize=True)


def test_reset_and_errors():
    cfg = make_cfg()
    model = make_model(cfg)
    sp = StreamingPolicy(model, cfg.data, "cpu")
    w, h = cfg.data.resolution
    img = np.random.default_rng(1).integers(0, 256, (h, w, 3), dtype=np.uint8)
    acts = np.zeros((cfg.data.actions_per_frame, 3), np.float32)
    try:
        sp.predict()
        raise AssertionError("predict before observe must fail")
    except RuntimeError:
        pass
    sp.observe(img, acts)
    first = sp.predict()
    sp.observe(img[::-1].copy(), acts + 0.5)
    assert not np.allclose(sp.predict(), first)
    sp.reset()
    sp.observe(img, acts)
    np.testing.assert_allclose(sp.predict(), first, atol=1e-6)
    for bad_img, bad_acts in [(img[:-1], acts), (img, acts[:-1])]:
        try:
            sp.observe(bad_img, bad_acts)
            raise AssertionError("shape mismatch must fail")
        except ValueError:
            pass


def test_load_policy_roundtrip(tmp_path):
    cfg = make_cfg()
    model = make_model(cfg)
    path = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "config": config_to_dict(cfg), "step": 3}, path)
    loaded, cfg2 = load_policy(path, "cpu")
    assert cfg2 == cfg and not loaded.training
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert StreamingPolicy(loaded, cfg2.data).cfg is None
    sp = load_streaming_policy(path, device="cpu")
    assert sp.cfg == cfg
    w, h = cfg.data.resolution
    sp.observe(np.zeros((h, w, 3), np.uint8), np.zeros((cfg.data.actions_per_frame, 3), np.float32))
    assert sp.predict().shape == (cfg.data.chunk_len, 3)
