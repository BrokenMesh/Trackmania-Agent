"""TMPolicy: shapes, causality, padding, action-history paths, overfit, decode."""

from __future__ import annotations

import pytest
import torch

from tmagent.config import Config
from tmagent.model import TMPolicy, bc_loss, count_parameters


def make_cfg(head="regression", use_actions=True, history_s=0.2, channels=3) -> Config:
    cfg = Config()
    cfg.data.resolution = [32, 24]
    cfg.data.channels = channels
    cfg.data.history_s = history_s  # K = round(history_s * 20) + 1
    cfg.data.chunk_len = 4
    m = cfg.model
    m.d_model, m.n_layers, m.n_heads, m.tokens_per_frame, m.dropout = 32, 2, 4, 4, 0.0
    m.head, m.use_action_history, m.steer_bins = head, use_actions, 7
    cfg.validate()
    return cfg


def make_model(cfg: Config, seed: int = 0) -> TMPolicy:
    torch.manual_seed(seed)
    model = TMPolicy(cfg.model, cfg.data)
    torch.nn.init.normal_(model.head[-1].weight, std=0.3)  # make outputs input-sensitive
    return model.eval()


def make_batch(cfg: Config, b: int = 2, k: int | None = None, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    k = k or cfg.data.num_steps
    c, (w, h) = cfg.data.channels, cfg.data.resolution
    r, cl = cfg.data.actions_per_frame, cfg.data.chunk_len
    target = torch.rand(b, k, cl, 3, generator=g)
    target[..., 0] = target[..., 0] * 2 - 1
    target[..., 1:] = (target[..., 1:] > 0.5).float()
    return {
        "frames": torch.randint(0, 256, (b, k, c, h, w), generator=g, dtype=torch.uint8),
        "frame_valid": torch.ones(b, k, dtype=torch.bool),
        "hist_actions": torch.rand(b, k, r, 3, generator=g),
        "hist_valid": torch.ones(b, k, dtype=torch.bool),
        "target": target,
        "target_valid": torch.ones(b, k, cl, dtype=torch.bool),
    }


@pytest.mark.parametrize("head", ["regression", "discrete"])
@pytest.mark.parametrize("history_s", [0.0, 0.2])
def test_output_shapes(head, history_s):
    cfg = make_cfg(head=head, history_s=history_s)
    model = make_model(cfg)
    k = cfg.data.num_steps
    assert k == (1 if history_s == 0 else 5)
    out = model(make_batch(cfg, b=3))
    cl = cfg.data.chunk_len
    assert out["gas"].shape == out["brake"].shape == (3, k, cl)
    if head == "regression":
        assert out["steer"].shape == (3, k, cl)
    else:
        assert out["steer_logits"].shape == (3, k, cl, cfg.model.steer_bins)
    dec = model.decode(out)
    assert dec.shape == (3, k, cl, 3) and dec.dtype == torch.float32
    assert dec[..., 0].abs().max() <= 1.0
    assert set(dec[..., 1:].unique().tolist()) <= {0.0, 1.0}
    soft = model.decode(out, binarize=False)
    assert ((soft[..., 1:] >= 0) & (soft[..., 1:] <= 1)).all()


def test_fewer_steps_than_max_and_gray():
    cfg = make_cfg(channels=1, history_s=0.2)
    model = make_model(cfg)
    out = model(make_batch(cfg, k=2))  # K < data.num_steps is allowed
    assert out["gas"].shape[1] == 2
    with pytest.raises(ValueError):
        model(make_batch(cfg, k=cfg.data.num_steps + 1))


@pytest.mark.parametrize("head", ["regression", "discrete"])
def test_causality(head):
    cfg = make_cfg(head=head)
    model = make_model(cfg)
    batch = make_batch(cfg)
    base = model(batch)
    j = 2
    changed = {k: v.clone() for k, v in batch.items()}
    changed["frames"][:, j] = 255 - changed["frames"][:, j]
    changed["hist_actions"][:, j] = 1.0 - changed["hist_actions"][:, j]
    other = model(changed)
    for key in base:
        assert torch.allclose(base[key][:, :j], other[key][:, :j], atol=1e-6), key
        assert not torch.allclose(base[key][:, j:], other[key][:, j:], atol=1e-5), key


def test_padded_steps_are_finite_and_ignore_frame_content():
    cfg = make_cfg()
    model = make_model(cfg).train()
    batch = make_batch(cfg)
    batch["frame_valid"][:, :3] = False
    batch["hist_valid"][:, :3] = False
    batch["target_valid"][:, :3] = False
    batch["frames"][:, :3] = 0
    out = model(batch)
    assert all(torch.isfinite(v).all() for v in out.values())
    loss, metrics = bc_loss(out, batch, cfg.model, cfg.train)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # invalid frames are replaced by PAD tokens: their pixels and history values do not matter
    model.eval()
    ref = model(batch)
    batch["frames"][:, :3] = 200
    batch["hist_actions"][:, :3] = float("nan")
    out2 = model(batch)
    for key in ref:
        assert torch.allclose(ref[key], out2[key], atol=1e-6), key

    batch["frame_valid"][:] = False  # fully padded window must not produce NaN either
    batch["hist_valid"][:] = False
    assert all(torch.isfinite(v).all() for v in model(batch).values())


def _slice_steps(batch: dict, sl: slice) -> dict:
    return {k: v[:, sl].clone() for k, v in batch.items()}


def test_invalid_step_content_is_invisible():
    cfg = make_cfg()
    model = make_model(cfg)
    batch = make_batch(cfg)
    batch["frame_valid"][:, :2] = False
    base = model(batch)
    other = {k: v.clone() for k, v in batch.items()}
    other["frames"][:, :2] = 255 - other["frames"][:, :2]
    other["hist_actions"][:, :2] = float("nan")  # even garbage with hist_valid True
    other["hist_valid"][:, :2] = True
    changed = model(other)
    for key in base:
        assert torch.equal(base[key][:, 2:], changed[key][:, 2:]), key
        assert torch.isfinite(changed[key]).all(), key


def test_translation_and_truncation_invariance():
    cfg = make_cfg(history_s=0.3)  # K = 7
    model = make_model(cfg)
    full = make_batch(cfg, b=2, seed=3)
    k = full["frames"].shape[1]
    ref = model(full)  # all steps valid
    for end in range(1, k + 1):
        # (a) truncation: steps 0..end-1 only
        short = model(_slice_steps(full, slice(0, end)))
        # (b) the same steps, left padded with invalid steps up to K
        pad = k - end
        padded = _slice_steps(full, slice(0, end))
        for name, v in padded.items():
            filler = torch.zeros(v.shape[0], pad, *v.shape[2:], dtype=v.dtype)
            if name == "frames":
                filler = torch.randint(0, 256, filler.shape, dtype=torch.uint8)
            if name == "hist_actions":
                filler = torch.rand_like(filler)
            padded[name] = torch.cat([filler, v], dim=1)
        padded["hist_valid"][:, :pad] = True  # garbage-but-"valid" history on invalid steps
        left = model(padded)
        for key in ref:
            want = ref[key][:, end - 1]
            assert torch.allclose(short[key][:, -1], want, atol=1e-5), (key, end)
            assert torch.allclose(left[key][:, -1], want, atol=1e-5), (key, end)
            # every valid step, not just the last one
            assert torch.allclose(short[key], ref[key][:, :end], atol=1e-5), (key, end)
            assert torch.allclose(left[key][:, pad:], ref[key][:, :end], atol=1e-5), (key, end)


def test_relative_bias_is_used_and_trained():
    cfg = make_cfg()
    model = make_model(cfg).train()
    batch = make_batch(cfg)
    base = model(batch)["steer"].detach()
    loss, _ = bc_loss(model(batch), batch, cfg.model, cfg.train)
    loss.backward()
    g = model.rel_bias.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert not hasattr(model, "time_emb")
    with torch.no_grad():
        model.rel_bias.add_(torch.randn_like(model.rel_bias))
    assert not torch.allclose(model(batch)["steer"], base, atol=1e-4)


def test_no_action_path_when_history_disabled():
    cfg = make_cfg(use_actions=False)
    model = make_model(cfg)
    assert not hasattr(model, "action_mlp")
    batch = make_batch(cfg)
    a = model(batch)
    batch["hist_actions"] = torch.rand_like(batch["hist_actions"]) * 5 - 2
    batch["hist_valid"][:, 1] = False
    b = model(batch)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_action_history_matters_when_enabled():
    cfg = make_cfg(use_actions=True)
    model = make_model(cfg)
    batch = make_batch(cfg)
    a = model(batch)
    batch["hist_actions"] = 1.0 - batch["hist_actions"]
    b = model(batch)
    assert not torch.allclose(a["steer"], b["steer"], atol=1e-5)
    batch["hist_valid"][:] = False  # invalid history -> NO_ACTION, contents irrelevant
    c = model(batch)
    batch["hist_actions"] = torch.zeros_like(batch["hist_actions"])
    d = model(batch)
    assert torch.equal(c["steer"], d["steer"])


@pytest.mark.parametrize("head", ["regression", "discrete"])
def test_overfit_tiny_batch(head):
    cfg = make_cfg(head=head, history_s=0.1)  # K = 3
    torch.manual_seed(0)
    model = TMPolicy(cfg.model, cfg.data).train()
    batch = make_batch(cfg, b=4, seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(100):
        loss, _ = bc_loss(model(batch), batch, cfg.model, cfg.train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


def test_count_parameters():
    cfg = make_cfg()
    model = TMPolicy(cfg.model, cfg.data)
    c = count_parameters(model)
    assert c["total"] == sum(p.numel() for p in model.parameters())
    assert c["total"] == c["encoder"] + c["temporal"] + c["head"]
    assert min(c["encoder"], c["temporal"], c["head"]) > 0
    model.encoder.freeze()
    assert count_parameters(model)["trainable"] < c["total"]
