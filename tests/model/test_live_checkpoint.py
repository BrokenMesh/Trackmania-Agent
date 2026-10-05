"""runtime.live adopts the checkpoint's data settings and loads the policy before the game."""

from __future__ import annotations

import sys
import types

import pytest

from tmagent.config import Config, save_config
from tmagent.runtime import live


def ckpt_config() -> Config:
    cfg = Config()
    cfg.data.resolution, cfg.data.channels = [64, 48], 1
    cfg.data.frame_hz, cfg.data.control_hz, cfg.runtime.control_hz = 20, 40, 40
    cfg.data.history_s, cfg.data.chunk_len = 1.0, 4
    cfg.data.val_frac = 0.3  # not a live-relevant field: adopted silently
    cfg.validate()
    return cfg


def test_adopt_checkpoint_data():
    cfg = Config()  # data defaults: 128x96x3, 60 Hz, 2 s, chunk 8
    cfg.runtime.hold_s = 0.5
    cfg.game.backend = "tmnf"
    ck = ckpt_config()
    diff = live.adopt_checkpoint_data(cfg, types.SimpleNamespace(cfg=ck))
    assert diff == ["control_hz", "resolution", "channels", "history_s", "chunk_len"]
    assert cfg.data == ck.data and cfg.data is not ck.data
    assert cfg.runtime.control_hz == 40 and cfg.runtime.hold_s == 0.5
    assert cfg.game.backend == "tmnf"
    assert live.adopt_checkpoint_data(cfg, types.SimpleNamespace(cfg=ck)) == []  # now equal


def test_adopt_is_a_noop_without_checkpoint_config():
    cfg = Config()
    before = Config()
    assert live.adopt_checkpoint_data(cfg, types.SimpleNamespace(cfg=None)) == []
    assert live.adopt_checkpoint_data(cfg, object()) == []
    assert cfg == before


def test_main_loads_policy_before_game_and_uses_checkpoint_resolution(
    monkeypatch, tmp_path, capsys
):
    order: list[str] = []
    seen: dict = {}

    class Stop(Exception):
        pass

    def load_streaming_policy(ckpt, device):
        order.append("policy")
        return types.SimpleNamespace(cfg=ckpt_config())

    def make_game(game_cfg, data_cfg):
        order.append("game")
        seen["data"] = data_cfg
        raise Stop

    for modname, attr, fn in [
        ("tmagent.game", "make_game", make_game),
        ("tmagent.model.streaming", "load_streaming_policy", load_streaming_policy),
    ]:
        mod = types.ModuleType(modname)
        setattr(mod, attr, fn)
        monkeypatch.setitem(sys.modules, modname, mod)
    cfg_path = tmp_path / "cfg.yaml"
    save_config(Config(), cfg_path)
    args = ["--config", str(cfg_path), "--ckpt", "m.pt", "--map", "A01", "--out", str(tmp_path)]
    with pytest.raises(Stop):
        live.main(args)
    assert order == ["policy", "game"]
    assert seen["data"].resolution == [64, 48] and seen["data"].channels == 1
    out = capsys.readouterr().out
    assert out.startswith("WARNING") and "resolution: [128, 96] -> [64, 48]" in out
