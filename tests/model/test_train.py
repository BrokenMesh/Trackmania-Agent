"""train_bc: end-to-end on a synthetic dataset, resume, speed warning, CLI dry run, configs."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from tmagent.config import Config, load_config
from tmagent.experiment import create_run
from tmagent.model import load_streaming_policy
from tmagent.train import train_bc

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class SynthDataset(Dataset):
    """Per-sample dicts in the batch format (no batch dim)."""

    def __init__(self, cfg: Config, n: int = 12, seed: int = 0) -> None:
        self.cfg, self.n, self.seed = cfg, n, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        d = self.cfg.data
        g = torch.Generator().manual_seed(self.seed * 1000 + i)
        k, (w, h) = d.num_steps, d.resolution
        target = torch.rand(k, d.chunk_len, 3, generator=g)
        target[..., 0] = target[..., 0] * 2 - 1
        target[..., 1:] = (target[..., 1:] > 0.5).float()
        fv = torch.ones(k, dtype=torch.bool)
        fv[: i % 3] = False  # some windows start before the episode
        return {
            "frames": torch.randint(0, 256, (k, d.channels, h, w), generator=g, dtype=torch.uint8),
            "frame_valid": fv,
            "hist_actions": torch.rand(k, d.actions_per_frame, 3, generator=g),
            "hist_valid": fv.clone(),
            "target": target,
            "target_valid": torch.ones(k, d.chunk_len, dtype=torch.bool),
            "progress": torch.rand(k, generator=g),
        }


def tiny_cfg(**train_kw) -> Config:
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg.data.resolution = [32, 24]
    cfg.data.history_s = 0.1  # K = 3
    cfg.data.chunk_len = 4
    cfg.model.d_model, cfg.model.n_layers, cfg.model.tokens_per_frame = 32, 1, 4
    cfg.train.warmup_steps, cfg.train.log_every = 1, 1
    for k, v in train_kw.items():
        setattr(cfg.train, k, v)
    cfg.validate()
    return cfg


def test_train_end_to_end_and_resume(tmp_path, capsys):
    cfg = tiny_cfg(steps=5, eval_every=3, ckpt_every=2)
    run = create_run("e2e", cfg, base=tmp_path)
    res = train_bc.train(cfg, SynthDataset(cfg), SynthDataset(cfg, 8, seed=1), run)
    assert res["step"] == 5 and np.isfinite(res["train_loss"]) and "steer_mae" in res["val"]
    out = capsys.readouterr().out
    assert "[model] params total" in out and "head" in out
    ckpts = sorted(p.name for p in (run / "checkpoints").iterdir())
    assert ckpts == ["last.pt", "step_000002.pt", "step_000004.pt", "step_000005.pt"]
    ck = torch.load(run / "checkpoints" / "last.pt", weights_only=True)
    assert ck["step"] == 5 and set(ck) == {"model", "optimizer", "scheduler", "step", "config"}
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert {r["step"] for r in rows if "train/loss" in r} == {1, 2, 3, 4, 5}
    assert {r["step"] for r in rows if "val/loss" in r} == {3, 5}
    assert rows[0]["train/lr"] > 0

    cfg.train.steps = 8
    res2 = train_bc.train(cfg, SynthDataset(cfg), None, run, resume=run / "checkpoints/last.pt")
    assert res2["step"] == 8
    assert "[resume]" in capsys.readouterr().out
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if "train/loss" in r][-4:] == [5, 6, 7, 8]
    assert torch.load(run / "checkpoints" / "last.pt", weights_only=True)["step"] == 8
    # finished runs resume as a no-op
    res3 = train_bc.train(cfg, SynthDataset(cfg), None, run, resume=run / "checkpoints/last.pt")
    assert res3["step"] == 8

    # the checkpoint is directly usable for live inference
    sp = load_streaming_policy(run / "checkpoints" / "last.pt", device="cpu")
    w, h = cfg.data.resolution
    sp.observe(np.zeros((h, w, 3), np.uint8), np.zeros((cfg.data.actions_per_frame, 3), np.float32))
    assert sp.predict().shape == (cfg.data.chunk_len, 3)


def test_speed_estimate_prints_after_20_steps(tmp_path, capsys):
    cfg = tiny_cfg(steps=22, eval_every=0, ckpt_every=0, log_every=100)
    train_bc.train(cfg, SynthDataset(cfg), None, create_run("speed", cfg, base=tmp_path))
    out = capsys.readouterr().out
    assert "[speed]" in out and "s/step" in out and "WARNING" not in out


def test_speed_report_warns_over_30_minutes():
    info, warn = train_bc.speed_report(0.5, 1000)
    assert "0.500 s/step" in info and warn is None
    info, warn = train_bc.speed_report(1.0, 2000)
    assert warn is not None and warn.startswith("WARNING") and "30 min" in warn


def test_optimizer_param_groups():
    cfg = tiny_cfg()
    from tmagent.model import TMPolicy

    model = TMPolicy(cfg.model, cfg.data)
    opt = train_bc.build_optimizer(model, cfg.train)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == cfg.train.weight_decay and no_decay["weight_decay"] == 0
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(p.ndim >= 2 for p in decay["params"])
    nd = {names[id(p)] for p in no_decay["params"]}
    assert {"type_emb.weight", "rel_bias", "pad_frame", "no_action"} <= nd
    assert not any(n.endswith("bias") for n in (names[id(p)] for p in decay["params"]))
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


def test_lr_schedule():
    cfg = tiny_cfg(steps=100, warmup_steps=10)
    f = [train_bc.lr_factor(s, cfg.train) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0) and f[10] == pytest.approx(1.0)
    assert f[-1] < 0.01 and all(a >= b - 1e-9 for a, b in zip(f[10:], f[11:], strict=False))


def test_cli_dry_run_with_fake_dataset_module(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)  # a dry run must not write into ./experiments
    fake = types.ModuleType("tmagent.data.dataset")

    def window_dataset(root, cfg, split, train):
        return SynthDataset(Config(data=cfg), n=0 if split == "val" else 8)

    fake.WindowDataset = window_dataset
    monkeypatch.setitem(sys.modules, "tmagent.data.dataset", fake)
    code = train_bc.main(["--config", str(CONFIGS / "smoke.yaml"), "--dry-run"])
    assert code == 0 and "[dry-run] ok" in capsys.readouterr().out
    assert not (tmp_path / "experiments").exists()


def test_cli_does_not_import_dataset_at_module_level():
    src = Path(train_bc.__file__).read_text()
    head = src.split("def main(")[0]
    assert "tmagent.data" not in head


@pytest.mark.parametrize(
    "name", ["smoke", "baseline_single_frame", "context_2s", "context_2s_no_actions"]
)
def test_configs_load(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg.name == name
    if name == "smoke":
        assert (cfg.model.d_model, cfg.model.n_layers, cfg.model.tokens_per_frame) == (64, 2, 4)
        assert cfg.data.history_s == 0.5 and cfg.data.resolution == [64, 48]
        assert (cfg.train.steps, cfg.train.batch_size, cfg.train.num_workers) == (20, 4, 0)
        assert (cfg.train.device, cfg.train.precision, cfg.data.root) == (
            "cpu",
            "fp32",
            "data/fake",
        )
    elif name == "baseline_single_frame":
        assert cfg.data.num_steps == 1 and not cfg.model.use_action_history
    elif name == "context_2s":
        assert cfg.data.history_s == 2 and cfg.model.use_action_history
    else:
        assert cfg.data.history_s == 2 and not cfg.model.use_action_history
