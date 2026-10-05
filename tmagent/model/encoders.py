"""Frame encoders: image -> P tokens of size d_model.

`tiny_cnn` trains from scratch. `timm:<name>` and `hf:<repo_id>` wrap external
backbones: the libraries are imported lazily, and weights are only fetched when
`ModelConfig.encoder_pretrained` is set (tests never do).
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tmagent.config import ModelConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def choose_grid(num_tokens: int, height: int, width: int) -> tuple[int, int]:
    """Factor `num_tokens = gh * gw` with gw / gh closest to the image aspect ratio."""
    target = math.log(width / height)
    pairs = [(g, num_tokens // g) for g in range(1, num_tokens + 1) if num_tokens % g == 0]
    return min(pairs, key=lambda p: (abs(math.log(p[1] / p[0]) - target), abs(p[0] - p[1])))


def _lazy_import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"encoder needs '{module}': pip install -e .[vision]") from e


class TokenHead(nn.Module):
    """Feature map [N, Cf, h, w] -> pool to a (gh, gw) grid -> LayerNorm -> Linear + 2D pos emb."""

    def __init__(self, in_dim: int, d_model: int, grid: tuple[int, int]) -> None:
        super().__init__()
        self.grid = grid
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(grid[0] * grid[1], d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(fmap, self.grid).flatten(2).transpose(1, 2)  # [N, P, Cf]
        return self.proj(self.norm(x)) + self.pos_emb


class FrameEncoder(nn.Module):
    """forward(x float [N, C, H, W] in [0, 1]) -> tokens [N, P, d_model].

    Subclasses provide `backbone` (a module) and `features(x) -> [N, Cf, h, w]`.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, cfg: ModelConfig, hw: tuple[int, int]):
        super().__init__()
        self.backbone = backbone
        self.num_tokens = cfg.tokens_per_frame
        self.d_model = cfg.d_model
        self.frozen = False
        grid = choose_grid(cfg.tokens_per_frame, *hw)
        self.token_head = TokenHead(feat_dim, cfg.d_model, grid)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def freeze(self) -> None:
        """Freeze backbone weights (the token projection stays trainable)."""
        self.frozen = True
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> FrameEncoder:
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            fmap = self.features(x)
        return self.token_head(fmap)


class TinyCNNEncoder(FrameEncoder):
    """Four stride-2 conv blocks (conv + GroupNorm + GELU) and one stride-1 block."""

    WIDTHS = (32, 64, 96, 128)

    def __init__(self, cfg: ModelConfig, channels: int, hw: tuple[int, int]) -> None:
        layers: list[nn.Module] = []
        cin = channels
        for cout, stride in [(w, 2) for w in self.WIDTHS] + [(self.WIDTHS[-1], 1)]:
            layers += [
                nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                nn.GroupNorm(math.gcd(8, cout), cout),
                nn.GELU(),
            ]
            cin = cout
        super().__init__(nn.Sequential(*layers), cin, cfg, hw)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ExternalEncoder(FrameEncoder):
    """Base for pretrained backbones: gray -> RGB, resize to the native size, normalize."""

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        cfg: ModelConfig,
        hw: tuple[int, int],
        in_size: tuple[int, int],
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        super().__init__(backbone, feat_dim, cfg, hw)
        self.in_size = (int(in_size[0]), int(in_size[1]))
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), False
        )
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), False)

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if tuple(x.shape[-2:]) != self.in_size:
            x = F.interpolate(x, size=self.in_size, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class TimmEncoder(ExternalEncoder):
    """`timm:<name>`: CNN feature maps (NCHW/NHWC) or ViT tokens (prefix tokens dropped)."""

    def __init__(self, name: str, cfg: ModelConfig, channels: int, hw: tuple[int, int]) -> None:
        timm = _lazy_import("timm")
        net = timm.create_model(name, pretrained=cfg.encoder_pretrained, num_classes=0)
        pcfg = getattr(net, "pretrained_cfg", None) or {}
        in_size = tuple(pcfg.get("input_size", (3, 224, 224)))[1:]
        mean = tuple(pcfg.get("mean", IMAGENET_MEAN))
        std = tuple(pcfg.get("std", IMAGENET_STD))
        was_training = net.training
        net.eval()
        with torch.no_grad():
            f = net.forward_features(torch.zeros(1, 3, *in_size))
        net.train(was_training)
        nf = getattr(net, "num_features", None)
        prefix, grid, nhwc = 0, (0, 0), False
        if f.ndim == 4:
            nhwc = nf is not None and f.shape[-1] == nf and f.shape[1] != nf
            feat_dim = f.shape[-1] if nhwc else f.shape[1]
        else:
            feat_dim = f.shape[-1]
            prefix = int(getattr(net, "num_prefix_tokens", 0))
            n_patch = f.shape[1] - prefix
            gs = getattr(getattr(net, "patch_embed", None), "grid_size", None)
            if gs is not None and int(gs[0]) * int(gs[1]) == n_patch:
                grid = (int(gs[0]), int(gs[1]))
            else:
                g = math.isqrt(n_patch)
                if g * g != n_patch:
                    raise ValueError(f"cannot infer the token grid of {name} ({n_patch} patches)")
                grid = (g, g)
        super().__init__(net, feat_dim, cfg, hw, in_size, mean, std)
        self._prefix, self._grid, self._nhwc = prefix, grid, nhwc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone.forward_features(self.prepare(x))
        if f.ndim == 4:
            return f.permute(0, 3, 1, 2) if self._nhwc else f
        tok = f[:, self._prefix :]
        return tok.transpose(1, 2).reshape(f.shape[0], f.shape[-1], *self._grid)


class HFEncoder(ExternalEncoder):
    """`hf:<repo_id>`: vision tower of a transformers model (SigLIP/SigLIP2, CLIP, ViT)."""

    def __init__(self, repo_id: str, cfg: ModelConfig, channels: int, hw: tuple[int, int]) -> None:
        tf = _lazy_import("transformers")
        if cfg.encoder_pretrained:
            model = tf.AutoModel.from_pretrained(repo_id)
        else:  # random init, config only
            model = tf.AutoModel.from_config(tf.AutoConfig.from_pretrained(repo_id))
        net = getattr(model, "vision_model", model)
        vcfg = net.config
        patch = int(vcfg.patch_size)
        size = getattr(vcfg, "image_size", 224)
        in_size = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        try:
            proc = tf.AutoImageProcessor.from_pretrained(repo_id)
            mean, std = tuple(proc.image_mean), tuple(proc.image_std)
        except Exception:  # processor config unavailable: keep the SigLIP-style defaults
            pass
        if hasattr(net, "use_head"):  # SigLIP attention-pooling head is unused here
            net.use_head = False
        super().__init__(net, int(vcfg.hidden_size), cfg, hw, in_size, mean, std)
        self._grid = (in_size[0] // patch, in_size[1] // patch)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=self.prepare(x))
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        n_patch = self._grid[0] * self._grid[1]
        tok = h[:, h.shape[1] - n_patch :]  # prefix tokens (CLS) come first
        return tok.transpose(1, 2).reshape(h.shape[0], h.shape[-1], *self._grid)


def build_encoder(cfg: ModelConfig, channels: int, resolution: tuple[int, int]) -> FrameEncoder:
    """Build the frame encoder named by cfg.encoder. `resolution` is (H, W)."""
    name = cfg.encoder
    if name == "tiny_cnn":
        enc: FrameEncoder = TinyCNNEncoder(cfg, channels, resolution)
    elif name.startswith("timm:"):
        enc = TimmEncoder(name[5:], cfg, channels, resolution)
    elif name.startswith("hf:"):
        enc = HFEncoder(name[3:], cfg, channels, resolution)
    else:
        raise ValueError(f"unknown model.encoder {name!r} (tiny_cnn | timm:<name> | hf:<repo_id>)")
    if cfg.encoder_frozen:
        enc.freeze()
    return enc
