from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from tmagent.data.split import filter_index, load_overrides, split_of

from .test_helpers import small_cfg


def reference_u(map_uid: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{map_uid}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def test_split_is_deterministic_and_follows_hash_thresholds():
    for uid in ("A", "B", "x-y_z", "0123456789"):
        for seed in (0, 1, 42):
            s = split_of(uid, seed, 0.2, 0.1)
            assert s == split_of(uid, seed, 0.2, 0.1)
            u = reference_u(uid, seed)
            assert s == ("test" if u < 0.1 else "val" if u < 0.3 else "train")


def test_split_changes_with_seed_and_fractions_hold():
    uids = [f"map{i:04d}" for i in range(4000)]
    counts = Counter(split_of(u, 0, 0.1, 0.2) for u in uids)
    assert abs(counts["test"] / 4000 - 0.2) < 0.03
    assert abs(counts["val"] / 4000 - 0.1) < 0.03
    assert abs(counts["train"] / 4000 - 0.7) < 0.03
    other = [split_of(u, 1, 0.1, 0.2) for u in uids]
    assert other != [split_of(u, 0, 0.1, 0.2) for u in uids]


def test_edge_fractions():
    assert split_of("m", 0, 0.0, 0.0) == "train"
    assert split_of("m", 0, 1.0, 0.0) == "val"
    assert split_of("m", 0, 0.0, 1.0) == "test"
    with pytest.raises(ValueError):
        split_of("m", 0, 0.6, 0.6)
    with pytest.raises(ValueError):
        split_of("m", 0, -0.1, 0.1)


def entries_for(uids):
    return [{"map_uid": u, "path": f"episodes/{u}/e{i}.npz"} for i, u in enumerate(uids)]


def test_filter_index_partitions_entries_by_map(tmp_path):
    cfg = small_cfg(split_seed=3, val_frac=0.2, test_frac=0.2)
    entries = entries_for([f"m{i % 40}" for i in range(120)])  # 3 episodes per map
    parts = {s: filter_index(entries, s, cfg, tmp_path) for s in ("train", "val", "test")}
    assert sum(map(len, parts.values())) == 120
    maps = {s: {e["map_uid"] for e in es} for s, es in parts.items()}
    assert not (maps["train"] & maps["val"]) and not (maps["train"] & maps["test"])
    assert not (maps["val"] & maps["test"])
    for s, es in parts.items():
        assert all(split_of(e["map_uid"], 3, 0.2, 0.2) == s for e in es)
    with pytest.raises(ValueError):
        filter_index(entries, "dev", cfg, tmp_path)


def test_splits_json_overrides(tmp_path):
    cfg = small_cfg(val_frac=0.0, test_frac=0.0)  # everything is train by hash
    entries = entries_for(["a", "b", "c"])
    assert len(filter_index(entries, "train", cfg, tmp_path)) == 3
    (tmp_path / "splits.json").write_text(json.dumps({"a": "val", "c": "test", "unused": "val"}))
    assert [e["map_uid"] for e in filter_index(entries, "val", cfg, tmp_path)] == ["a"]
    assert [e["map_uid"] for e in filter_index(entries, "test", cfg, tmp_path)] == ["c"]
    assert [e["map_uid"] for e in filter_index(entries, "train", cfg, tmp_path)] == ["b"]
    assert load_overrides(tmp_path)["a"] == "val"
    assert load_overrides(None) == {}


def test_invalid_override_is_an_error(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps({"a": "validation"}))
    with pytest.raises(ValueError, match="invalid split"):
        filter_index(entries_for(["a"]), "train", small_cfg(), tmp_path)
