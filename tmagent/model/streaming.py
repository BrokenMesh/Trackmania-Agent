"""Live inference: StreamingPolicy keeps the last K encoded frames (each encoded once)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from tmagent.config import Config, DataConfig, config_from_dict
from tmagent.model.policy import TMPolicy, autocast_ctx, resolve_device


class StreamingPolicy:
    """Implements tmagent.interfaces.ChunkPolicy on top of a TMPolicy.

    State = ring buffer of the last K = data_cfg.num_steps steps: encoded frame tokens, the R
    actions executed before each frame, and validity flags. Before K frames were observed the
    missing steps are invalid, exactly like the dataset's left padding (the model is invariant
    to that padding, so predict() only runs on the observed steps). The first observe's
    past_actions count as invalid history (dataset behavior at episode start).

    `cfg` is the full checkpoint Config when built via load_streaming_policy, else None.
    """

    def __init__(
        self,
        model: TMPolicy,
        data_cfg: DataConfig,
        device: str | torch.device = "cpu",
        precision: str = "fp32",
        binarize: bool = True,
        cfg: Config | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(device)
        self.precision = precision
        self.binarize = binarize
        self.model = model.to(self.device).eval()
        self.chunk_len = data_cfg.chunk_len
        self.k = data_cfg.num_steps
        self.hist_len = data_cfg.actions_per_frame
        w, h = data_cfg.resolution
        self.image_shape = (h, w, data_cfg.channels)
        self.reset()

    def reset(self) -> None:
        """Forget all observed frames."""
        m, dev = self.model, self.device
        self._tokens = torch.zeros(self.k, m.num_tokens, m.model_cfg.d_model, device=dev)
        self._frame_valid = torch.zeros(self.k, dtype=torch.bool, device=dev)
        self._hist = torch.zeros(self.k, self.hist_len, 3, device=dev)
        self._hist_valid = torch.zeros(self.k, dtype=torch.bool, device=dev)
        self._n_obs = 0

    @torch.inference_mode()
    def observe(self, image: np.ndarray, past_actions: np.ndarray) -> None:
        """image uint8 (H, W, C); past_actions float32 (R, 3), oldest first."""
        img = np.asarray(image)
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape != self.image_shape or img.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 image {self.image_shape}, got {img.dtype} {img.shape}"
            )
        past = np.asarray(past_actions, dtype=np.float32)
        if past.shape != (self.hist_len, 3):
            raise ValueError(f"past_actions must be ({self.hist_len}, 3), got {past.shape}")
        x = torch.from_numpy(np.ascontiguousarray(img)).to(self.device).permute(2, 0, 1)[None]
        with autocast_ctx(self.device, self.precision):
            tok = self.model.encode_frames(x)[0].float()
        first = self._n_obs == 0
        hist = torch.from_numpy(past).to(self.device)
        self._tokens = torch.cat([self._tokens[1:], tok[None]])
        self._frame_valid = torch.cat([self._frame_valid[1:], self._frame_valid.new_ones(1)])
        self._hist = torch.cat(
            [self._hist[1:], torch.zeros_like(hist)[None] if first else hist[None]]
        )
        self._hist_valid = torch.cat(
            [self._hist_valid[1:], self._hist_valid.new_full((1,), not first)]
        )
        self._n_obs += 1

    @torch.inference_mode()
    def predict(self) -> np.ndarray:
        """float32 (chunk_len, 3) for the last observed frame."""
        if self._n_obs == 0:
            raise RuntimeError("predict() called before observe()")
        sl = slice(self.k - min(self._n_obs, self.k), None)  # drop the invalid left padding
        with autocast_ctx(self.device, self.precision):
            out = self.model.forward_tokens(
                self._tokens[None, sl],
                self._frame_valid[None, sl],
                self._hist[None, sl],
                self._hist_valid[None, sl],
            )
        chunk = self.model.decode(out, binarize=self.binarize)[0, -1]
        return chunk.float().cpu().numpy().astype(np.float32)


def load_policy(
    ckpt_path: str | Path, device: str | torch.device = "auto"
) -> tuple[TMPolicy, Config]:
    """Load a train_bc checkpoint -> (model in eval mode on `device`, its Config).

    Pretrained encoder weights are not fetched: they come from the checkpoint.
    """
    dev = resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = config_from_dict(ckpt["config"])
    model_cfg = dataclasses.replace(cfg.model, encoder_pretrained=False)
    model = TMPolicy(model_cfg, cfg.data)
    model.load_state_dict(ckpt["model"])
    return model.to(dev).eval(), cfg


def load_streaming_policy(
    ckpt_path: str | Path, device: str | torch.device = "auto"
) -> StreamingPolicy:
    """Checkpoint -> ready-to-use StreamingPolicy (precision from the checkpoint's
    cfg.runtime.precision); the checkpoint Config is kept as `.cfg`."""
    dev = resolve_device(device)
    model, cfg = load_policy(ckpt_path, dev)
    return StreamingPolicy(model, cfg.data, dev, cfg.runtime.precision, cfg=cfg)
