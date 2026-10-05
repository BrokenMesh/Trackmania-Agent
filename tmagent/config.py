"""Experiment configuration: YAML -> nested dataclasses.

Usage:
    cfg = load_config("configs/smoke.yaml", overrides=["train.steps=10"])

Unknown keys are errors. Every module reads only its own section.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/tmnf"  # dataset root (episodes/ + index.jsonl)
    frame_hz: int = 20
    control_hz: int = 60
    resolution: list[int] = field(default_factory=lambda: [128, 96])  # [W, H]
    channels: int = 3
    history_s: float = 2.0  # K = round(history_s * frame_hz) + 1 frame steps; 0 -> K = 1
    chunk_len: int = 8  # actions predicted per step
    split_seed: int = 0
    val_frac: float = 0.1
    test_frac: float = 0.1
    # augmentation (train split only)
    action_dropout: float = 0.3  # P(drop the whole action history of a sample)
    action_token_dropout: float = 0.1  # P(drop a single step's action token)
    action_noise_steer: float = 0.05  # std of gaussian noise on history steer
    action_flip_prob: float = 0.02  # P(flip a binary gas/brake history value)
    image_aug: bool = True  # small brightness/contrast/shift jitter

    @property
    def num_steps(self) -> int:
        return int(round(self.history_s * self.frame_hz)) + 1

    @property
    def actions_per_frame(self) -> int:
        return self.control_hz // self.frame_hz


@dataclass
class ModelConfig:
    encoder: str = "tiny_cnn"  # tiny_cnn | timm:<name> | hf:<repo_id>
    encoder_pretrained: bool = False
    encoder_frozen: bool = False
    tokens_per_frame: int = 16
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    use_action_history: bool = False  # True only with copycat countermeasures, D-013
    head: str = "regression"  # regression | discrete
    steer_bins: int = 21


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda
    precision: str = "bf16"  # bf16 | fp32 (bf16 only on cuda)
    compile: bool = False
    batch_size: int = 32
    num_workers: int = 4
    steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 1000
    ckpt_every: int = 2000
    log_every: int = 50
    steer_loss: str = "huber"  # huber | l1 | mse
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"steer": 1.0, "gas": 0.5, "brake": 0.5}
    )


@dataclass
class RuntimeConfig:
    control_hz: int = 60
    hold_s: float = 0.25  # hold last action this long after a chunk runs out
    spin_s: float = 0.002  # busy-wait window before each control deadline
    device: str = "auto"
    precision: str = "bf16"
    compile: bool = False


@dataclass
class EvalConfig:
    maps: list[str] = field(default_factory=list)  # map refs (paths or uids)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    mode: str = "sync"  # sync | realtime
    timeout_s: float = 120.0
    stuck_speed_kmh: float = 5.0
    stuck_s: float = 3.0
    offtrack_dist: float = 40.0  # world units from reference polyline
    action_noise: float = 0.0  # eval-time steer noise (robustness tests)


@dataclass
class GameConfig:
    backend: str = "fake"  # fake | tmnf
    # tmnf backend (see docs/setup_windows.md)
    tmi_host: str = "127.0.0.1"
    tmi_port: int = 8477  # TMAgentLink plugin listen port (TMI `set custom_port` + offset)
    connect_timeout_s: float = 30.0
    game_speed: float = 1.0  # live play speed (1.0 = real time)
    render_speed: float = 10.0  # sim speed between captured frames when rendering replays
    steer_mode: str = "binary"  # binary (left/right keys, verified API) | analog (InputType::Steer)
    steer_threshold: float = 0.5  # |steer| above this -> key press in binary mode
    window_size: list[int] = field(default_factory=lambda: [640, 480])
    camera: str = "cam1"
    map_dir: str = "data/tmnf/maps"  # .Challenge.Gbx files, resolved by map_uid or name
    # first-run knobs for UNVERIFIED TMI behaviour (tmagent/game/tmnf/PROTOCOL.md)
    restart_method: str = "rewind"  # rewind (to saved start state) | give_up
    map_path_style: str = "auto"  # how the `map` command argument is written
    frame_settle_renders: int = 1  # Render() calls to wait before a capture
    capture_flip_vertical: bool = False  # screenshot row order
    focus_window: bool = False  # focus the game window before each map load (intro needs focus)
    # fake backend
    fake_track: str = "oval"


@dataclass
class Config:
    name: str = "unnamed"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    game: GameConfig = field(default_factory=GameConfig)

    def validate(self) -> None:
        d = self.data
        if d.control_hz % d.frame_hz != 0:
            raise ValueError(f"control_hz ({d.control_hz}) must be a multiple of frame_hz")
        if self.runtime.control_hz != d.control_hz:
            raise ValueError("runtime.control_hz must equal data.control_hz")
        if d.channels not in (1, 3):
            raise ValueError("data.channels must be 1 or 3")
        if self.model.head not in ("regression", "discrete"):
            raise ValueError(f"unknown model.head {self.model.head!r}")
        if self.model.d_model % self.model.n_heads != 0:
            raise ValueError("model.d_model must be divisible by model.n_heads")


def _resolve_type(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _resolve_type(args[0]) if len(args) == 1 else tp
    return tp


def _from_dict(cls: type, raw: dict[str, Any], path: str = "") -> Any:
    if not isinstance(raw, dict):
        raise TypeError(f"{path or 'config'}: expected mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise KeyError(f"{path or 'config'}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for key, value in raw.items():
        tp = _resolve_type(hints[key])
        if dataclasses.is_dataclass(tp):
            kwargs[key] = _from_dict(tp, value, f"{path}{key}.")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _apply_override(raw: dict[str, Any], override: str) -> None:
    key, sep, value = override.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got {override!r}")
    node = raw
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    for ov in overrides or []:
        _apply_override(raw, ov)
    cfg = _from_dict(Config, raw)
    cfg.validate()
    return cfg


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Validated Config from a plain dict (e.g. stored in a checkpoint)."""
    cfg = _from_dict(Config, raw)
    cfg.validate()
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
