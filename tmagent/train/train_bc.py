"""Behavior-cloning training loop and CLI.

    python -m tmagent.train.train_bc --config configs/smoke.yaml [--set train.steps=10 ...]
        [--resume experiments/<run>/checkpoints/last.pt] [--dry-run]

The core is `train(cfg, train_ds, val_ds, run_dir, resume)`, which works with any torch
Dataset yielding per-sample dicts in the batch format of docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tmagent.config import Config, TrainConfig, config_to_dict, load_config, save_config
from tmagent.experiment import create_run, log_metrics
from tmagent.model.losses import bc_loss
from tmagent.model.policy import TMPolicy, autocast_ctx, count_parameters, resolve_device

SPEED_STEPS = 20  # steps after which s/step and the total-time estimate are printed
WARN_MINUTES = 30.0  # project rule: tell the user before GPU runs longer than this
MAX_VAL_BATCHES = 50  # cap on validation batches per evaluation
NO_DECAY_TAGS = ("emb", "cls_token", "reg_token", "pad_frame", "no_action", "rel_bias")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: torch.nn.Module, tcfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW without weight decay on norms, biases and embeddings (incl. learned tokens)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        skip = p.ndim < 2 or any(tag in name for tag in NO_DECAY_TAGS)
        (no_decay if skip else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": tcfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tcfg.lr, betas=(0.9, 0.95))


def lr_factor(step: int, tcfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to 0, as a multiplier of the base lr."""
    if step < tcfg.warmup_steps:
        return (step + 1) / tcfg.warmup_steps
    span = max(1, tcfg.steps - tcfg.warmup_steps)
    progress = min(1.0, (step - tcfg.warmup_steps) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(
    ds: Dataset, tcfg: TrainConfig, train: bool, device: torch.device, seed: int
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    n, bs = len(ds), tcfg.batch_size  # type: ignore[arg-type]
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=train,
        drop_last=train and n >= bs,
        num_workers=tcfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=tcfg.num_workers > 0,
        generator=gen if train else None,
    )


def _cycle(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def _add(acc: dict[str, torch.Tensor], metrics: dict[str, torch.Tensor]) -> None:
    for k, v in metrics.items():
        acc[k] = acc[k] + v.detach().float() if k in acc else v.detach().float()


@torch.no_grad()
def evaluate(
    model: TMPolicy, loader: DataLoader, cfg: Config, device: torch.device
) -> dict[str, float]:
    """Mean of per-batch loss/metrics over (up to MAX_VAL_BATCHES) validation batches."""
    was_training = model.training
    model.eval()
    acc: dict[str, torch.Tensor] = {}
    n = 0
    for i, batch in enumerate(loader):
        if i >= MAX_VAL_BATCHES:
            break
        batch = to_device(batch, device)
        with autocast_ctx(device, cfg.train.precision):
            out = model(batch)
        _, metrics = bc_loss(out, batch, cfg.model, cfg.train)
        _add(acc, metrics)
        n += 1
    model.train(was_training)
    return {k: float(v) / max(n, 1) for k, v in acc.items()}


def save_checkpoint(
    run_dir: Path, step: int, model: TMPolicy, opt: Any, sched: Any, cfg: Config
) -> Path:
    """Write checkpoints/step_XXXXXX.pt and a copy as last.pt."""
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "config": config_to_dict(cfg),
    }
    ckpt_dir = run_dir / "checkpoints"
    path = ckpt_dir / f"step_{step:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    shutil.copyfile(path, tmp)
    os.replace(tmp, ckpt_dir / "last.pt")
    return path


def speed_report(s_per_step: float, steps_to_run: int) -> tuple[str, str | None]:
    """(info line, warning line or None) for the measured speed."""
    est_min = s_per_step * steps_to_run / 60.0
    info = f"[speed] {s_per_step:.3f} s/step -> est. {est_min:.1f} min for {steps_to_run} steps"
    warn = None
    if est_min > WARN_MINUTES:
        warn = (
            f"WARNING: estimated run time {est_min:.0f} min exceeds {WARN_MINUTES:.0f} min. "
            "Project rule: inform the user before GPU runs longer than 30 min."
        )
    return info, warn


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def train(
    cfg: Config,
    train_ds: Dataset,
    val_ds: Dataset | None,
    run_dir: str | Path,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train until cfg.train.steps (counting resumed steps). Returns a summary dict:
    step, train_loss (last log window), val (last evaluation), run_dir, last_ckpt, s_per_step."""
    tcfg = cfg.train
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if len(train_ds) == 0:  # type: ignore[arg-type]
        raise ValueError("training dataset is empty")
    seed_everything(tcfg.seed)
    device = resolve_device(tcfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = TMPolicy(cfg.model, cfg.data).to(device)
    counts = count_parameters(model)
    mp = {k: f"{v / 1e6:.3f}M" for k, v in counts.items()}
    print(
        f"[model] params total {mp['total']} | encoder {mp['encoder']} | temporal "
        f"{mp['temporal']} | head {mp['head']} | trainable {mp['trainable']} | device {device}"
    )
    fwd = torch.compile(model) if tcfg.compile else model
    opt = build_optimizer(model, tcfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, tcfg))

    step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step = int(ckpt["step"])
        print(f"[resume] {resume} at step {step}")
    start_step = step

    train_loader = make_loader(train_ds, tcfg, True, device, tcfg.seed + start_step)
    has_val = val_ds is not None and len(val_ds) > 0  # type: ignore[arg-type]
    val_loader = make_loader(val_ds, tcfg, False, device, 0) if has_val else None
    batches = _cycle(train_loader)
    params = [p for p in model.parameters() if p.requires_grad]

    acc: dict[str, torch.Tensor] = {}
    n_acc = 0
    result: dict[str, Any] = {"train_loss": float("nan"), "val": {}, "s_per_step": float("nan")}
    t_first = t_log = time.perf_counter()
    model.train()
    while step < tcfg.steps:
        batch = to_device(next(batches), device)
        with autocast_ctx(device, tcfg.precision):
            out = fwd(batch)
        loss, metrics = bc_loss(out, batch, cfg.model, tcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.grad_clip > 0:
            metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
        lr = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        step += 1
        _add(acc, metrics)
        n_acc += 1

        done = step - start_step
        if done == 1:  # exclude loader start-up / compile from the speed estimate
            _sync(device)
            t_first = time.perf_counter()
        elif done == SPEED_STEPS and tcfg.steps - step > 0:
            _sync(device)
            s_per_step = (time.perf_counter() - t_first) / (SPEED_STEPS - 1)
            info, warn = speed_report(s_per_step, tcfg.steps - start_step)
            print(info)
            if warn:
                print(warn)
            result["s_per_step"] = s_per_step

        if step % tcfg.log_every == 0 or step == tcfg.steps:
            mean = {f"train/{k}": float(v) / n_acc for k, v in acc.items()}
            if not math.isfinite(mean["train/loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            _sync(device)
            now = time.perf_counter()
            mean["train/lr"] = lr
            mean["train/s_per_step"] = (now - t_log) / n_acc
            log_metrics(run_dir, step, mean)
            print(
                f"[train] step {step}/{tcfg.steps} loss {mean['train/loss']:.4f} "
                f"steer_mae {mean['train/steer_mae']:.4f} lr {mean['train/lr']:.2e} "
                f"{mean['train/s_per_step']:.3f} s/step"
            )
            result["train_loss"] = mean["train/loss"]
            acc, n_acc, t_log = {}, 0, now

        last = step == tcfg.steps
        if val_loader is not None and (
            (tcfg.eval_every > 0 and step % tcfg.eval_every == 0) or last
        ):
            val = evaluate(model, val_loader, cfg, device)
            log_metrics(run_dir, step, {f"val/{k}": v for k, v in val.items()})
            print(f"[val] step {step} loss {val['loss']:.4f} steer_mae {val['steer_mae']:.4f}")
            result["val"] = val
        if (tcfg.ckpt_every > 0 and step % tcfg.ckpt_every == 0) or last:
            result["last_ckpt"] = str(save_checkpoint(run_dir, step, model, opt, sched, cfg))

    if step == start_step and "last_ckpt" not in result:  # nothing to do (already finished)
        result["last_ckpt"] = str(Path(resume)) if resume is not None else None
    result.update(step=step, run_dir=str(run_dir))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Behavior-cloning training.")
    ap.add_argument("--config", required=True, help="YAML config (see configs/)")
    ap.add_argument(
        "--set",
        dest="overrides",
        nargs="+",
        action="extend",
        default=[],
        metavar="key=value",
        help="config overrides, e.g. train.steps=100",
    )
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps, exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.dry_run:
        cfg.train.steps = 2
        cfg.train.warmup_steps = min(cfg.train.warmup_steps, 1)
        cfg.train.eval_every = 2
        cfg.train.ckpt_every = 0

    try:
        from tmagent.data.dataset import WindowDataset  # not imported at module level
    except ImportError as e:
        print(f"error: cannot import tmagent.data.dataset ({e})")
        return 1
    train_ds = WindowDataset(cfg.data.root, cfg.data, "train", True)
    val_ds = WindowDataset(cfg.data.root, cfg.data, "val", False)
    if len(train_ds) == 0 and args.dry_run and len(val_ds) > 0:
        print("[dry-run] no train windows, using the val split")
        train_ds = val_ds
    if len(train_ds) == 0:
        print(f"error: no training windows under {cfg.data.root!r}")
        return 1
    print(f"[data] train windows {len(train_ds)}, val windows {len(val_ds)}")

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run(cfg.name + "-dryrun", cfg, base=tmp)
            train(cfg, train_ds, val_ds, run_dir)
        print("[dry-run] ok")
        return 0

    resume = Path(args.resume) if args.resume else None
    if (
        resume is not None
        and resume.parent.name == "checkpoints"
        and (resume.parent.parent / "config.yaml").exists()
    ):
        run_dir = resume.parent.parent  # continue in the same run directory
        save_config(cfg, run_dir / "config_resume.yaml")
    else:
        run_dir = create_run(cfg.name, cfg)
    print(f"[run] {run_dir}")
    train(cfg, train_ds, val_ds, run_dir, resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
