"""tools/system_report.py: works without a GPU, capacity estimate formulas, nvidia-smi parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import system_report as sr

GIB = sr.GIB


def info_with(gpus, ram_gib=32.0, smi=None, torch_gpus=True):
    return {
        "date": "2026-01-01T00:00:00",
        "os": "TestOS 1",
        "machine": "x86_64",
        "cpu": "Test CPU",
        "cpu_logical": 8,
        "ram_gib": ram_gib,
        "python": "3.11.0 (CPython)",
        "python_exe": "python",
        "torch": {
            "installed": True,
            "version": "2.9.0",
            "cuda_available": bool(gpus) and torch_gpus,
            "cuda_build": "12.4" if gpus else None,
            "cudnn": 90100,
            "gpus": gpus if torch_gpus else [],
            "bf16": bool(gpus),
            "cpu_threads": 8,
        },
        "nvidia_smi": smi,
    }


def test_collect_and_markdown_run_here():
    text = sr.render_markdown(sr.collect())
    for heading in ("# System report", "## Machine", "## GPU", "## Capacity estimate"):
        assert heading in text
    assert "ESTIMATE" in text and "Python" in text and "torch" in text


def test_main_writes_markdown(tmp_path: Path, capsys):
    out = tmp_path / "sub" / "system.md"
    assert sr.main(["--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# System report")
    assert "written to" in capsys.readouterr().out


def test_estimate_formulas():
    # inference: (VRAM - 1 GiB reserve) / (2 B * 1.2); training: 50 % of VRAM / 16 B
    assert sr.max_params_inference(25) == pytest.approx(24 * GIB / 2.4)
    assert sr.max_params_training(24) == pytest.approx(24 * GIB * 0.5 / 16)
    assert sr.max_params_inference(0.5) == 0.0
    # the table inverts the same formulas
    assert sr.inference_gib(sr.max_params_inference(12)) == pytest.approx(12)
    assert sr.training_gib(sr.max_params_training(12)) == pytest.approx(12)


def test_gpu_report_and_capacity_table():
    gpu = {"name": "Test RTX 24", "memory_gib": 24.0, "capability": "8.9", "sms": 128}
    text = sr.render_markdown(info_with([gpu]))
    assert "Test RTX 24" in text and "24.0" in text and "8.9" in text
    assert "bf16 on GPU | yes" in text and "available (build 12.4" in text
    assert "bf16 inference VRAM GiB" in text and "inference RAM GiB" not in text
    rows = {
        line.split("|")[1].strip(): [c.strip() for c in line.split("|")[4:6]]
        for line in text.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().endswith((" M", " B"))
    }
    # 3 B bf16 inference fits 24 GiB, 3 B AdamW training does not, 0.3 B does
    assert rows["3.00 B"] == ["yes", "no"]
    assert rows["300 M"] == ["yes", "yes"]


def test_cpu_only_machine_uses_ram():
    text = sr.render_markdown(info_with([], ram_gib=16.0))
    assert "No CUDA GPU" in text and "system RAM (16.0 GiB)" in text and "inference RAM GiB" in text
    assert "speed on CPU" in text
    no_mem = sr.render_markdown(info_with([], ram_gib=None))
    assert "no estimate possible" in no_mem and "unknown" in no_mem


def test_torch_missing_is_reported():
    info = info_with([])
    info["torch"] = {"installed": False}
    assert "| torch | not installed |" in sr.render_markdown(info)


def test_nvidia_smi_parsing(monkeypatch):
    csv = "name, memory.total [MiB], driver_version\nNVIDIA GeForce RTX 4090, 24564 MiB, 551.23"
    plain = "| NVIDIA-SMI 551.23   Driver Version: 551.23   CUDA Version: 12.4 |"

    def fake_run(cmd, timeout=10.0):
        return csv if any("--query-gpu" in c for c in cmd) else plain

    monkeypatch.setattr(sr.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sr, "_run", fake_run)
    smi = sr.nvidia_smi()
    assert smi["cuda_version"] == "12.4" and smi["csv"] == csv
    (g,) = smi["gpus"]
    assert g["name"] == "NVIDIA GeForce RTX 4090" and g["driver"] == "551.23"
    assert g["memory_gib"] == pytest.approx(24564 / 1024)
    # torch without CUDA support: the GPU table falls back to nvidia-smi
    text = sr.render_markdown(info_with([g], smi=smi, torch_gpus=False))
    assert "NVIDIA GeForce RTX 4090" in text and "551.23" in text and "CUDA Version" not in text
    assert "driver CUDA version (nvidia-smi) | 12.4" in text

    monkeypatch.setattr(sr, "_run", lambda cmd, timeout=10.0: None)  # nvidia-smi fails
    assert sr.nvidia_smi()["gpus"] == []
    monkeypatch.setattr(sr.shutil, "which", lambda name: None)
    assert sr.nvidia_smi() is None
