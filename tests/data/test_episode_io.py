from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from tmagent.data import episode_io
from tmagent.data.episode_io import (
    EpisodeCache,
    EpisodeReader,
    append_index,
    episode_path,
    load_episode,
    read_index,
    save_episode,
)

from .test_helpers import make_episode, small_cfg


def assert_same(a, b):
    for k in ("frames", "frame_times_ms", "actions", "action_times_ms", "positions", "speeds_kmh"):
        x, y = getattr(a, k), getattr(b, k)
        assert x.dtype == y.dtype and x.shape == y.shape, k
        assert np.array_equal(x, y), k
    assert {**a.meta, "format": 2} == b.meta


def noisy_episode(n_ticks=200, **kw):
    """Episode whose frames are incompressible-ish random noise (catches decode mix-ups)."""
    ep = make_episode(small_cfg(), n_ticks, **kw)
    rng = np.random.default_rng(0)
    ep.frames = rng.integers(0, 256, size=ep.frames.shape, dtype=np.uint8)
    return ep


def test_save_load_roundtrip(tmp_path):
    ep = noisy_episode(120, map_uid="mapA", source="fake:1")
    path = save_episode(ep, tmp_path)
    assert path == tmp_path / "episodes" / "mapA" / ep.meta["episode_id"] and path.is_dir()
    loaded = load_episode(path)
    assert_same(ep, loaded)
    assert loaded.meta["format"] == 2 and "format" not in ep.meta  # input meta untouched


def test_directory_layout_v2(tmp_path):
    ep = noisy_episode(60)
    path = save_episode(ep, tmp_path)
    assert sorted(p.name for p in path.iterdir()) == ["arrays.npz", "frames.bin", "meta.json"]
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # no tmp / old dirs
    assert json.loads((path / "meta.json").read_text()) == {**ep.meta, "format": 2}
    with np.load(path / "arrays.npz", allow_pickle=False) as z:
        assert set(z.files) == {
            "frame_offsets",
            "frame_shape",
            "frame_times_ms",
            "actions",
            "action_times_ms",
            "positions",
            "speeds_kmh",
        }
        t_f = len(ep.frames)
        assert z["frame_offsets"].shape == (t_f + 1,) and z["frame_offsets"].dtype == np.int64
        assert z["frame_offsets"][0] == 0
        assert z["frame_offsets"][-1] == (path / "frames.bin").stat().st_size
        assert np.all(np.diff(z["frame_offsets"]) > 0)
        assert z["frame_shape"].tolist() == [24, 32, 3]


def test_reader_random_access_frames(tmp_path):
    ep = noisy_episode(200)
    reader = EpisodeReader(save_episode(ep, tmp_path))
    assert reader.num_frames == len(ep.frames) and reader.frame_shape == (24, 32, 3)
    assert reader.meta["format"] == 2 and reader.meta["map_uid"] == ep.meta["map_uid"]
    for name in ("frame_times_ms", "actions", "action_times_ms", "positions", "speeds_kmh"):
        assert np.array_equal(getattr(reader, name), getattr(ep, name))
    for idx in ([0], [7], [3, 4, 5, 6], [5, 1, 9], [2, 2, 3], [0, 19], list(range(len(ep.frames)))):
        got = reader.frames(idx)
        assert got.dtype == np.uint8 and got.shape == (len(idx), 24, 32, 3)
        assert np.array_equal(got, ep.frames[idx]), idx
    assert reader.frames([]).shape == (0, 24, 32, 3)
    assert reader.frames(np.array([4, 5])).shape[0] == 2
    for bad in ([-1], [len(ep.frames)], [0, 10**6]):
        with pytest.raises(IndexError):
            reader.frames(bad)


def test_reader_is_lazy_and_holds_no_open_file(tmp_path):
    ep = noisy_episode(60)
    path = save_episode(ep, tmp_path)
    reader = EpisodeReader(path)
    (path / "frames.bin").unlink()  # arrays and meta were loaded eagerly...
    assert len(reader.actions) == len(ep.actions) and reader.meta["episode_id"]
    with pytest.raises(FileNotFoundError):  # ...frames are only read on demand
        reader.frames([0])


def test_reader_survives_pickle_and_fork(tmp_path):
    import multiprocessing as mp

    ep = noisy_episode(60)
    reader = EpisodeReader(save_episode(ep, tmp_path))
    clone = pickle.loads(pickle.dumps(reader))
    assert np.array_equal(clone.frames([1, 2]), ep.frames[[1, 2]])
    reader.frames([0])  # used in the parent before the fork

    with mp.get_context("fork").Pool(2) as pool:
        results = pool.map(_decode_in_worker, [(str(reader.path), i) for i in range(4)])
    for i, got in enumerate(results):
        assert np.array_equal(got, ep.frames[i : i + 3])


def _decode_in_worker(args):
    path, i = args
    return EpisodeReader(path).frames([i, i + 1, i + 2])


def test_corrupt_or_truncated_frames_raise(tmp_path):
    ep = noisy_episode(60)
    path = save_episode(ep, tmp_path)
    blob = (path / "frames.bin").read_bytes()
    reader = EpisodeReader(path)
    (path / "frames.bin").write_bytes(blob[: len(blob) // 2])
    with pytest.raises(ValueError, match="truncated"):
        reader.frames([len(ep.frames) - 1])
    assert np.array_equal(reader.frames([0]), ep.frames[[0]])  # the intact prefix still reads
    garbled = bytearray(blob)
    off = int(reader._offsets[3])
    garbled[off + 4 : off + 12] = b"\xff" * 8
    (path / "frames.bin").write_bytes(bytes(garbled))
    with pytest.raises(ValueError, match="corrupt|bytes"):
        reader.frames([3])


def test_unsupported_format_is_rejected(tmp_path):
    path = save_episode(noisy_episode(30), tmp_path)
    meta = json.loads((path / "meta.json").read_text())
    meta["format"] = 1
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format"):
        EpisodeReader(path)


def test_empty_episode_roundtrip(tmp_path):
    ep = make_episode(small_cfg(), 0)
    loaded = load_episode(save_episode(ep, tmp_path))
    assert loaded.frames.shape == (0, 24, 32, 3) and loaded.actions.shape == (0, 3)
    assert EpisodeReader(save_episode(ep, tmp_path)).frames([]).shape == (0, 24, 32, 3)


def test_meta_with_numpy_values_is_serialized(tmp_path):
    ep = make_episode(small_cfg(), 30, extra=np.int64(5), arr=np.arange(3))
    loaded = load_episode(save_episode(ep, tmp_path))
    assert loaded.meta["extra"] == 5 and loaded.meta["arr"] == [0, 1, 2]


def test_overwrite_replaces_directory(tmp_path):
    cfg = small_cfg()
    a = make_episode(cfg, 60, source="s")
    b = make_episode(cfg, 90, source="s")  # same episode_id
    assert save_episode(a, tmp_path) == save_episode(b, tmp_path)
    path = episode_path(tmp_path, "mapA", b.meta["episode_id"])
    assert_same(b, load_episode(path))
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_stale_tmp_dir_from_a_crash_is_ignored(tmp_path):
    ep = make_episode(small_cfg(), 30)
    path = episode_path(tmp_path, "mapA", ep.meta["episode_id"])
    stale = path.with_name(path.name + ".tmp")
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    save_episode(ep, tmp_path)
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_failed_write_is_atomic(tmp_path, monkeypatch):
    cfg = small_cfg()
    ep = make_episode(cfg, 60)
    path = save_episode(ep, tmp_path)
    before = {p.name: p.read_bytes() for p in path.iterdir()}

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(episode_io.np, "savez", boom)
    with pytest.raises(OSError):
        save_episode(make_episode(cfg, 90), tmp_path)  # same id as ep
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before  # old episode intact
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # tmp cleaned up

    other = make_episode(cfg, 60, map_uid="mapB")
    with pytest.raises(OSError):
        save_episode(other, tmp_path)
    assert not episode_path(tmp_path, "mapB", other.meta["episode_id"]).exists()
    assert list((tmp_path / "episodes" / "mapB").iterdir()) == []


def test_rejects_non_uint8_frames(tmp_path):
    ep = make_episode(small_cfg(), 30)
    ep.frames = ep.frames.astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        save_episode(ep, tmp_path)
    assert list((tmp_path / "episodes" / "mapA").iterdir()) == []


def test_path_components_are_sanitized(tmp_path):
    ep = make_episode(small_cfg(), 20, map_uid="../evil/uid", episode_id="../../x")
    path = save_episode(ep, tmp_path)
    assert tmp_path in path.resolve().parents
    assert path.parent.parent == tmp_path / "episodes"


def test_index_roundtrip(tmp_path):
    cfg = small_cfg()
    eps = [make_episode(cfg, 60 + 30 * i, map_uid=f"map{i % 2}", source=f"s{i}") for i in range(3)]
    paths = []
    for ep in eps:
        paths.append(save_episode(ep, tmp_path))
        append_index(tmp_path, ep, paths[-1])
    entries = read_index(tmp_path)
    assert len(entries) == 3
    for e, ep, p in zip(entries, eps, paths, strict=True):
        assert e["path"] == p.relative_to(tmp_path).as_posix()  # the episode directory
        assert e["num_frames"] == len(ep.frames) and e["format"] == 2
        assert {k: e[k] for k in ep.meta} == ep.meta
        assert EpisodeReader(tmp_path / e["path"]).meta["episode_id"] == ep.meta["episode_id"]
    lines = (tmp_path / "index.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(line)["path"] for line in lines)


def test_index_works_with_relative_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ep = make_episode(small_cfg(), 30)
    path = save_episode(ep, "ds")
    append_index("ds", ep, path)
    (entry,) = read_index("ds")
    assert entry["path"] == f"episodes/mapA/{ep.meta['episode_id']}"


def test_read_index_missing_dedupes_and_survives_torn_lines(tmp_path):
    assert read_index(tmp_path) == []
    cfg = small_cfg()
    ep = make_episode(cfg, 60)
    path = save_episode(ep, tmp_path)
    append_index(tmp_path, ep, path)
    append_index(tmp_path, ep, path)  # re-indexed: one entry survives
    with open(tmp_path / "index.jsonl", "a") as f:
        f.write('{"truncated": tru')  # torn write
    with pytest.warns(UserWarning, match="skipping"):
        entries = read_index(tmp_path)
    assert len(entries) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read_index(tmp_path / "nonexistent") == []


def test_episode_cache_is_lru_over_readers(tmp_path):
    loads: list[str] = []

    def loader(p: Path):
        loads.append(p.name)
        return EpisodeReader(p)

    cfg = small_cfg()
    paths = [save_episode(make_episode(cfg, 30, source=f"s{i}"), tmp_path) for i in range(3)]
    cache = EpisodeCache(size=2, loader=loader)
    a = cache.get(paths[0])
    assert isinstance(a, EpisodeReader)
    assert cache.get(str(paths[0])) is a  # str and Path keys are the same entry
    cache.get(paths[1])
    cache.get(paths[0])  # refresh 0 -> 1 is now least recent
    cache.get(paths[2])  # evicts 1
    assert len(cache) == 2 and paths[0] in cache and paths[1] not in cache
    assert loads == [p.name for p in (paths[0], paths[1], paths[2])]
    cache.get(paths[1])  # reload
    assert loads[-1] == paths[1].name and paths[0] not in cache
    cache.clear()
    assert len(cache) == 0
    assert EpisodeCache(size=0).size == 1  # never smaller than one
    assert EpisodeCache().size >= 8  # readers are cheap, the default holds many
