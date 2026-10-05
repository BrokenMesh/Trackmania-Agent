"""Frame encoders: tiny_cnn shapes, grid choice, frozen backbone, lazy timm/hf (faked)."""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from tmagent.config import ModelConfig
from tmagent.model.encoders import build_encoder, choose_grid


def mcfg(**kw) -> ModelConfig:
    base = dict(tokens_per_frame=4, d_model=16)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize(
    "p,h,w,expected",
    [(16, 96, 128, (4, 4)), (4, 48, 64, (2, 2)), (8, 96, 128, (2, 4)), (16, 64, 256, (2, 8))]
    + [(1, 10, 10, (1, 1)), (13, 96, 128, (1, 13))],
)
def test_choose_grid(p, h, w, expected):
    assert choose_grid(p, h, w) == expected


@pytest.mark.parametrize("channels", [1, 3])
def test_tiny_cnn_shapes(channels):
    enc = build_encoder(mcfg(tokens_per_frame=6), channels, (48, 64))
    x = torch.rand(5, channels, 48, 64)
    assert enc(x).shape == (5, 6, 16)
    assert enc.token_head.grid[0] * enc.token_head.grid[1] == 6


def test_tiny_cnn_frozen_keeps_projection_trainable():
    enc = build_encoder(mcfg(encoder_frozen=True), 3, (24, 32))
    enc.train()
    assert not any(p.requires_grad for p in enc.backbone.parameters())
    assert all(p.requires_grad for p in enc.token_head.parameters())
    enc(torch.rand(2, 3, 24, 32)).sum().backward()
    assert all(p.grad is None for p in enc.backbone.parameters())
    assert enc.token_head.proj.weight.grad is not None


def test_unknown_encoder():
    with pytest.raises(ValueError):
        build_encoder(mcfg(encoder="resnet"), 3, (24, 32))


def test_timm_missing_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", None)  # import raises ImportError
    with pytest.raises(ImportError, match="pip install"):
        build_encoder(mcfg(encoder="timm:foo"), 3, (24, 32))


class FakeNet(nn.Module):
    """Stand-in for a timm model: kind in {cnn, nhwc, vit}."""

    pretrained_cfg = {"input_size": (3, 32, 32), "mean": (0.5, 0.5, 0.5), "std": (0.25,) * 3}
    num_features = 8

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.num_prefix_tokens = 1 if kind == "vit" else 0
        k, s = (8, 8) if kind == "vit" else (3, 2)
        self.conv = nn.Conv2d(3, 8, k, s, 0 if kind == "vit" else 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, 8))
        self.patch_embed = types.SimpleNamespace(grid_size=(4, 4))

    def forward_features(self, x):
        f = self.conv(x)
        if self.kind == "cnn":
            return f
        if self.kind == "nhwc":
            return f.permute(0, 2, 3, 1)
        tok = f.flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)


@pytest.mark.parametrize("kind", ["cnn", "nhwc", "vit"])
def test_timm_paths_with_fake_library(monkeypatch, kind):
    calls = {}

    def create_model(name, pretrained, num_classes):
        calls.update(name=name, pretrained=pretrained, num_classes=num_classes)
        return FakeNet(kind)

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=create_model))
    enc = build_encoder(mcfg(encoder="timm:fake", encoder_frozen=True), 1, (24, 32))
    assert calls == {"name": "fake", "pretrained": False, "num_classes": 0}
    enc.train()
    assert not enc.backbone.training
    x = torch.rand(3, 1, 24, 32)  # gray, resized to 32x32, repeated to RGB
    tokens = enc(x)
    assert tokens.shape == (3, 4, 16)
    tokens.sum().backward()
    assert enc.token_head.proj.weight.grad is not None
    assert all(p.grad is None for p in enc.backbone.parameters())


def test_hf_path_with_fake_library(monkeypatch):
    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=8, image_size=32, patch_size=8)
            self.conv = nn.Conv2d(3, 8, 8, 8)
            self.cls = nn.Parameter(torch.zeros(1, 1, 8))

        def forward(self, pixel_values):
            tok = self.conv(pixel_values).flatten(2).transpose(1, 2)
            tok = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], dim=1)
            return types.SimpleNamespace(last_hidden_state=tok)

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = Vision()
            self.text_model = nn.Linear(100, 100)

    seen = []
    fake = types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(
            from_pretrained=lambda r: seen.append(("pre", r)) or Full(),
            from_config=lambda c: seen.append(("cfg", c)) or Full(),
        ),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda r: f"config:{r}"),
        AutoImageProcessor=types.SimpleNamespace(
            from_pretrained=lambda r: types.SimpleNamespace(
                image_mean=[0.5] * 3, image_std=[0.5] * 3, size={"height": 32, "width": 32}
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    enc = build_encoder(mcfg(encoder="hf:org/siglip", encoder_frozen=True), 3, (24, 32))
    assert seen == [("cfg", "config:org/siglip")]  # encoder_pretrained False: no weights
    assert not hasattr(enc.backbone, "text_model")
    assert enc(torch.rand(2, 3, 24, 32)).shape == (2, 4, 16)
    build_encoder(mcfg(encoder="hf:org/siglip", encoder_pretrained=True), 3, (24, 32))
    assert seen[-1] == ("pre", "org/siglip")
