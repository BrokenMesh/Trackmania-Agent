"""Behavior-cloning loss: steer (regression or bins), gas / brake BCE, masked by target_valid."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tmagent.config import ModelConfig, TrainConfig
from tmagent.model.policy import decode_outputs, steer_bin_centers

HUBER_DELTA = 0.1  # steer lives in [-1, 1]; L1 beyond 0.1 keeps it robust to multimodality


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def bc_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted steer + gas + brake loss over all (step, chunk) entries with target_valid.

    Returns (loss, metrics); metrics are detached 0-dim tensors (call float() to log):
    loss, loss_{steer,gas,brake}, steer_mae, gas_acc, brake_acc and the same three metrics
    restricted to chunk step 0 (suffix `_step0`).
    """
    target = batch["target"].float()  # [B, K, C, 3]
    mask = batch["target_valid"].float()  # [B, K, C]
    t_steer = target[..., 0]
    t_gas = (target[..., 1] >= 0.5).float()
    t_brake = (target[..., 2] >= 0.5).float()
    gas, brake = outputs["gas"].float(), outputs["brake"].float()

    if model_cfg.head == "discrete":
        logits = outputs["steer_logits"].float()
        centers = steer_bin_centers(model_cfg.steer_bins).to(logits)
        tgt_bin = (t_steer[..., None] - centers).abs().argmin(-1)
        l_steer = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt_bin.reshape(-1), reduction="none"
        ).reshape(t_steer.shape)
    else:
        pred = torch.tanh(outputs["steer"].float())
        kind = train_cfg.steer_loss
        if kind == "huber":
            l_steer = F.huber_loss(pred, t_steer, reduction="none", delta=HUBER_DELTA)
        elif kind == "l1":
            l_steer = (pred - t_steer).abs()
        elif kind == "mse":
            l_steer = (pred - t_steer) ** 2
        else:
            raise ValueError(f"unknown train.steer_loss {kind!r} (huber | l1 | mse)")
    l_gas = F.binary_cross_entropy_with_logits(gas, t_gas, reduction="none")
    l_brake = F.binary_cross_entropy_with_logits(brake, t_brake, reduction="none")

    ls, lg, lb = (_masked_mean(x, mask) for x in (l_steer, l_gas, l_brake))
    w = train_cfg.loss_weights
    loss = w.get("steer", 1.0) * ls + w.get("gas", 1.0) * lg + w.get("brake", 1.0) * lb

    with torch.no_grad():
        dec = decode_outputs(outputs, steer_bin_centers(model_cfg.steer_bins), binarize=False)
        mae = (dec[..., 0] - t_steer).abs()
        gas_ok = ((gas > 0) == (t_gas > 0.5)).float()
        brake_ok = ((brake > 0) == (t_brake > 0.5)).float()
        m0 = mask[..., 0]
        metrics = {
            "loss": loss.detach(),
            "loss_steer": ls.detach(),
            "loss_gas": lg.detach(),
            "loss_brake": lb.detach(),
            "steer_mae": _masked_mean(mae, mask),
            "gas_acc": _masked_mean(gas_ok, mask),
            "brake_acc": _masked_mean(brake_ok, mask),
            "steer_mae_step0": _masked_mean(mae[..., 0], m0),
            "gas_acc_step0": _masked_mean(gas_ok[..., 0], m0),
            "brake_acc_step0": _masked_mean(brake_ok[..., 0], m0),
        }
    return loss, metrics
