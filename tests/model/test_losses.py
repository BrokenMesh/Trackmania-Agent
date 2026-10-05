"""bc_loss: variants, masking, weights, metrics."""

from __future__ import annotations

import pytest
import torch

from tmagent.config import ModelConfig, TrainConfig
from tmagent.model.losses import HUBER_DELTA, bc_loss


def make(head="regression", b=2, k=3, c=4):
    torch.manual_seed(0)
    mcfg = ModelConfig(head=head, steer_bins=5)
    tcfg = TrainConfig()
    target = torch.rand(b, k, c, 3)
    target[..., 0] = target[..., 0] * 2 - 1
    target[..., 1:] = (target[..., 1:] > 0.5).float()
    batch = {"target": target, "target_valid": torch.ones(b, k, c, dtype=torch.bool)}
    out = {"gas": torch.randn(b, k, c), "brake": torch.randn(b, k, c)}
    if head == "discrete":
        out["steer_logits"] = torch.randn(b, k, c, 5)
    else:
        out["steer"] = torch.randn(b, k, c)
    return out, batch, mcfg, tcfg


@pytest.mark.parametrize("head", ["regression", "discrete"])
def test_metrics_keys_and_finite(head):
    out, batch, mcfg, tcfg = make(head)
    loss, m = bc_loss(out, batch, mcfg, tcfg)
    assert loss.ndim == 0 and torch.isfinite(loss)
    expected = {"loss", "loss_steer", "loss_gas", "loss_brake"}
    for base in ("steer_mae", "gas_acc", "brake_acc"):
        expected |= {base, f"{base}_step0"}
    assert set(m) == expected
    assert all(not v.requires_grad and torch.isfinite(v) for v in m.values())
    assert torch.allclose(m["loss"], loss.detach())


@pytest.mark.parametrize("kind", ["huber", "l1", "mse"])
def test_regression_steer_variants(kind):
    out, batch, mcfg, tcfg = make()
    tcfg.steer_loss = kind
    tcfg.loss_weights = {"steer": 1.0, "gas": 0.0, "brake": 0.0}
    loss, _ = bc_loss(out, batch, mcfg, tcfg)
    err = (torch.tanh(out["steer"]) - batch["target"][..., 0]).abs()
    expected = {
        "l1": err.mean(),
        "mse": (err**2).mean(),
        "huber": torch.where(
            err <= HUBER_DELTA, 0.5 * err**2, HUBER_DELTA * (err - 0.5 * HUBER_DELTA)
        ).mean(),
    }[kind]
    assert torch.allclose(loss, expected, atol=1e-6)
    tcfg.steer_loss = "bogus"
    with pytest.raises(ValueError):
        bc_loss(out, batch, mcfg, tcfg)


def test_discrete_target_is_nearest_bin():
    out, batch, mcfg, tcfg = make("discrete")
    tcfg.loss_weights = {"steer": 1.0, "gas": 0.0, "brake": 0.0}
    batch["target"][..., 0] = 0.26  # centers are -1, -.5, 0, .5, 1 -> nearest is 0.5 (index 3)
    logits = torch.full_like(out["steer_logits"], -10.0)
    logits[..., 3] = 10.0
    out["steer_logits"] = logits
    loss, m = bc_loss(out, batch, mcfg, tcfg)
    assert loss < 1e-3
    assert torch.allclose(m["steer_mae"], torch.tensor(0.24), atol=1e-3)


def test_mask_and_weights():
    out, batch, mcfg, tcfg = make()
    valid = torch.zeros_like(batch["target_valid"])
    valid[0, 1, :2] = True
    batch["target_valid"] = valid
    loss, m = bc_loss(out, batch, mcfg, tcfg)
    # corrupt only invalid entries: nothing changes
    out2 = {k: v.clone() for k, v in out.items()}
    for v in out2.values():
        v[~valid] = 123.0
    batch2 = {"target": batch["target"].clone(), "target_valid": valid}
    batch2["target"][~valid] = 7.0
    loss2, m2 = bc_loss(out2, batch2, mcfg, tcfg)
    assert torch.allclose(loss, loss2)
    assert torch.allclose(m["steer_mae"], m2["steer_mae"])
    # weights
    tcfg.loss_weights = {"steer": 2.0, "gas": 3.0, "brake": 5.0}
    loss3, m3 = bc_loss(out, batch, mcfg, tcfg)
    exp = 2 * m3["loss_steer"] + 3 * m3["loss_gas"] + 5 * m3["loss_brake"]
    assert torch.allclose(loss3, exp, atol=1e-6)
    # all-masked batch gives zero, not NaN
    batch["target_valid"] = torch.zeros_like(valid)
    loss4, m4 = bc_loss(out, batch, mcfg, tcfg)
    assert loss4 == 0 and all(torch.isfinite(v) for v in m4.values())


def test_step0_metrics_and_accuracy():
    out, batch, mcfg, tcfg = make()
    t = batch["target"]
    out["gas"] = torch.where(t[..., 1] > 0.5, 5.0, -5.0)  # perfect gas
    out["brake"] = torch.where(t[..., 2] > 0.5, -5.0, 5.0)  # perfectly wrong brake
    out["steer"] = torch.atanh(t[..., 0].clamp(-0.99, 0.99))
    batch["target"][..., 0] = t[..., 0].clamp(-0.99, 0.99)
    _, m = bc_loss(out, batch, mcfg, tcfg)
    assert m["gas_acc"] == 1 and m["gas_acc_step0"] == 1
    assert m["brake_acc"] == 0 and m["brake_acc_step0"] == 0
    assert m["steer_mae"] < 1e-5
    # make only chunk step 0 wrong for steer
    out["steer"][..., 0] += 1.0
    _, m = bc_loss(out, batch, mcfg, tcfg)
    assert m["steer_mae_step0"] > 0.1 and m["steer_mae"] > 0
    assert torch.allclose(m["steer_mae"], m["steer_mae_step0"] / 4, atol=1e-4)
