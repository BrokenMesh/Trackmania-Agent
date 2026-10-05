"""Temporal causal transformer policy (see docs/ARCHITECTURE.md, "Model").

Sequence per frame step k: [a_k, f_k,1 .. f_k,P]. Tokens of step k attend to all
tokens of steps <= k. The last frame token of every step is read out into a
chunk of `chunk_len` actions.

Time is encoded only RELATIVELY: a learned per-head attention bias indexed by the step
distance d = step_q - step_k >= 0 (no absolute / distance-to-window-end embedding), and
steps with frame_valid False are key-masked (their tokens, the learned PAD frame tokens,
are only visible to themselves, so no row is ever fully masked). Consequently the output
at a valid step k depends only on the valid steps <= k and their relative distances: it is
invariant to the content of invalid steps and to where the window starts (truncation and
left padding give the same output). Dense supervision at every step k therefore trains
exactly the function that is evaluated at inference, where the current step is the last
one; an absolute distance-to-last-step embedding would make K-1 of the K supervised
steps see positions that never occur at inference.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tmagent.config import DataConfig, ModelConfig
from tmagent.model.encoders import build_encoder


def resolve_device(name: str | torch.device = "auto") -> torch.device:
    """'auto' -> cuda if available else cpu."""
    if isinstance(name, torch.device):
        return name
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast_ctx(device: torch.device, precision: str) -> Any:
    """bf16 autocast only on cuda, otherwise a no-op context."""
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def steer_bin_centers(bins: int) -> torch.Tensor:
    """Centers of the discrete steer classes, evenly spaced in [-1, 1]."""
    return torch.linspace(-1.0, 1.0, bins)


def decode_outputs(
    outputs: dict[str, torch.Tensor], centers: torch.Tensor | None = None, binarize: bool = True
) -> torch.Tensor:
    """Raw outputs -> float actions [..., 3] (steer, gas, brake) in the interfaces.py layout.

    Regression: steer = tanh. Discrete (`steer_logits` present): expectation over `centers`.
    gas / brake = sigmoid, thresholded at 0.5 when `binarize`.
    """
    if "steer_logits" in outputs:
        assert centers is not None
        p = torch.softmax(outputs["steer_logits"].float(), dim=-1)
        steer = (p * centers.to(p)).sum(-1)
    else:
        steer = torch.tanh(outputs["steer"].float())
    gas = torch.sigmoid(outputs["gas"].float())
    brake = torch.sigmoid(outputs["brake"].float())
    if binarize:
        gas, brake = (gas > 0.5).float(), (brake > 0.5).float()
    return torch.stack([steer, gas, brake], dim=-1)


class AttnBlock(nn.Module):
    """Pre-norm transformer block whose attention takes an additive bias [B, H, L, L]."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, length, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype), dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, length, d)))
        return x + self.drop(self.mlp(self.norm2(x)))


class TMPolicy(nn.Module):
    """Behavior-cloning policy: frame encoder + action tokens + causal temporal transformer."""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        d = model_cfg.d_model
        self.num_tokens = model_cfg.tokens_per_frame
        self.hist_len = data_cfg.actions_per_frame
        self.chunk_len = data_cfg.chunk_len
        self.max_steps = data_cfg.num_steps
        self.use_actions = model_cfg.use_action_history
        w, h = data_cfg.resolution
        self.encoder = build_encoder(model_cfg, data_cfg.channels, (h, w))

        if self.use_actions:
            self.action_mlp = nn.Sequential(
                nn.Linear(self.hist_len * 3, d), nn.GELU(), nn.Linear(d, d)
            )
        self.no_action = nn.Parameter(torch.zeros(d))
        self.pad_frame = nn.Parameter(torch.zeros(self.num_tokens, d))
        self.type_emb = nn.Embedding(2, d)  # 0 = action, 1 = frame
        for p in (self.no_action, self.pad_frame, self.type_emb.weight):
            nn.init.trunc_normal_(p, std=0.02)
        # learned attention bias per head, indexed by the step distance q - k >= 0
        self.rel_bias = nn.Parameter(torch.zeros(model_cfg.n_heads, self.max_steps))
        self.blocks = nn.ModuleList(
            AttnBlock(d, model_cfg.n_heads, model_cfg.dropout) for _ in range(model_cfg.n_layers)
        )
        self.final_norm = nn.LayerNorm(d)

        self.discrete = model_cfg.head == "discrete"
        self.steer_dim = model_cfg.steer_bins if self.discrete else 1
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(d, self.chunk_len * (self.steer_dim + 2)),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("steer_centers", steer_bin_centers(model_cfg.steer_bins), False)

    # ---- frames -------------------------------------------------------------------------
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames uint8 [N, C, H, W] (or float in [0, 1]) -> tokens [N, P, d_model]."""
        x = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames.float()
        return self.encoder(x)

    # ---- temporal model -----------------------------------------------------------------
    def _action_tokens(self, hist_actions: torch.Tensor, hist_valid: torch.Tensor) -> torch.Tensor:
        b, k = hist_valid.shape
        no_action = self.no_action.expand(b, k, -1)
        if not self.use_actions:
            return no_action
        valid = hist_valid.bool()
        hist = torch.where(valid[:, :, None, None], hist_actions.float(), 0.0)
        tok = self.action_mlp(hist.reshape(b, k, -1))
        return torch.where(valid[:, :, None], tok.to(no_action.dtype), no_action)

    def attention_bias(self, frame_valid: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Float bias [B, H, K * n_tokens, K * n_tokens]: relative-distance bias, -inf where
        a query may not attend (future steps; invalid keys for valid queries; anything but
        its own step for invalid queries)."""
        valid = frame_valid.bool()
        k = valid.shape[1]
        step = torch.arange(k, device=valid.device)
        dist = step[:, None] - step[None, :]  # [K, K], q - k
        rel = self.rel_bias[:, dist.clamp(min=0)]  # [H, K, K]
        own = dist == 0
        blocked = (
            (dist < 0)[None]
            | (valid[:, :, None] & ~valid[:, None, :])
            | (~valid[:, :, None] & ~own[None])
        )  # [B, K, K]
        bias = rel[None].masked_fill(blocked[:, None], float("-inf"))
        return bias.repeat_interleave(n_tokens, dim=2).repeat_interleave(n_tokens, dim=3)

    def forward_tokens(
        self,
        frame_tokens: torch.Tensor,
        frame_valid: torch.Tensor,
        hist_actions: torch.Tensor,
        hist_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """frame_tokens [B, K, P, d] -> outputs with leading dims [B, K, chunk_len, ...].

        Invalid frames (frame_valid False) get the learned PAD tokens and are key-masked
        for all valid steps (see module docstring).
        """
        b, k, p, d = frame_tokens.shape
        if k > self.max_steps:
            raise ValueError(f"K={k} exceeds data.num_steps={self.max_steps}")
        fv = frame_valid.bool()
        ft = torch.where(
            fv[:, :, None, None], frame_tokens.to(self.pad_frame.dtype), self.pad_frame
        )
        at = self._action_tokens(hist_actions, hist_valid.bool() & fv)  # invalid step: NO_ACTION
        x = torch.cat([at[:, :, None], ft], dim=2)  # [B, K, 1 + P, d]
        type_ids = torch.tensor([0] + [1] * p, device=x.device)
        x = x + self.type_emb(type_ids)[None, None]
        n = 1 + p
        bias = self.attention_bias(frame_valid, n)
        x = x.reshape(b, k * n, d)
        for block in self.blocks:
            x = block(x, bias)
        h = self.final_norm(x).reshape(b, k, n, d)[:, :, -1]  # last frame token of each step
        out = self.head(h).reshape(b, k, self.chunk_len, self.steer_dim + 2)
        gas, brake = out[..., -2], out[..., -1]
        if self.discrete:
            return {"steer_logits": out[..., : self.steer_dim], "gas": gas, "brake": brake}
        return {"steer": out[..., 0], "gas": gas, "brake": brake}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """batch per ARCHITECTURE.md "Training batch format" (frames uint8 [B, K, C, H, W])."""
        frames = batch["frames"]
        b, k = frames.shape[:2]
        tokens = self.encode_frames(frames.flatten(0, 1)).reshape(b, k, self.num_tokens, -1)
        return self.forward_tokens(
            tokens, batch["frame_valid"], batch["hist_actions"], batch["hist_valid"]
        )

    def decode(self, outputs: dict[str, torch.Tensor], binarize: bool = True) -> torch.Tensor:
        """Raw outputs -> float32 [B, K, chunk_len, 3] in the interfaces.py action layout."""
        return decode_outputs(outputs, self.steer_centers, binarize)


def count_parameters(model: TMPolicy) -> dict[str, int]:
    """Parameter counts: total, encoder, temporal (everything else), head, trainable."""

    def n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    total, enc, head = n(model), n(model.encoder), n(model.head)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "encoder": enc,
        "temporal": total - enc - head,
        "head": head,
        "trainable": trainable,
    }
