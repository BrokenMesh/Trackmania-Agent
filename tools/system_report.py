"""Phase 0.1: hardware and software report with a rough model-size capacity estimate.

    python tools/system_report.py [--out docs/system.md]

Collects OS, CPU, RAM, Python, torch, CUDA, GPU names, VRAM and bf16 support (torch plus
`nvidia-smi` when available), writes a markdown file (default docs/system.md) and prints
it. Works on CPU-only machines. The capacity table is a formula-based ESTIMATE that
must be replaced by measurements (tools/measure_latency.py, the [model]/[speed] lines of
tmagent.train.train_bc).
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

GIB = 1024**3
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "system.md"

# Capacity estimate (ESTIMATE, see the table notes in the report).
INFER_BYTES_PER_PARAM = 2.0  # bf16 weights
INFER_OVERHEAD = 0.2  # activations / buffers, fraction of the weights
INFER_RESERVE_GIB = 1.0  # CUDA context, framework workspace
TRAIN_BYTES_PER_PARAM = 16.0  # fp32 master weights 4 + grads 4 + AdamW m and v 8
TRAIN_HEADROOM = 0.5  # fraction of memory left for activations and fragmentation
REFERENCE_PARAMS = (0.05e9, 0.3e9, 0.5e9, 1e9, 3e9)


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def cpu_name() -> str:
    """Marketing name of the CPU, best effort."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            import winreg  # type: ignore[import-not-found]

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        elif sys.platform == "darwin":
            name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if name:
                return name
    except (OSError, ImportError, IndexError, ValueError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def ram_bytes() -> int | None:
    """Installed physical memory in bytes, best effort."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        elif sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong),
                    ("tpage", ctypes.c_ulonglong),
                    ("apage", ctypes.c_ulonglong),
                    ("tvirt", ctypes.c_ulonglong),
                    ("avirt", ctypes.c_ulonglong),
                    ("ext", ctypes.c_ulonglong),
                ]

            st = MemStatus()
            st.length = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return int(st.total)
        elif sys.platform == "darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            return int(out) if out else None
    except (OSError, ValueError, ImportError):
        pass
    return None


def nvidia_smi() -> dict[str, Any] | None:
    """nvidia-smi query (name, memory.total, driver_version) and driver CUDA version, or None."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    csv = _run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv"])
    if csv is None:
        return {"gpus": [], "csv": None, "cuda_version": None, "error": "nvidia-smi failed"}
    gpus = []
    for row in csv.splitlines()[1:]:  # first line is the header
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 3:
            m = re.match(r"([\d.]+)\s*MiB", parts[1])
            gpus.append(
                {
                    "name": parts[0],
                    "memory_gib": float(m.group(1)) * 1024**2 / GIB if m else None,
                    "driver": parts[2],
                }
            )
    plain = _run([exe]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", plain)
    return {"gpus": gpus, "csv": csv, "cuda_version": m.group(1) if m else None, "error": None}


def torch_info() -> dict[str, Any]:
    """torch version, CUDA build/availability, per-GPU properties, bf16 support."""
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_build": torch.version.cuda,
        "cudnn": None,
        "gpus": [],
        "bf16": False,
        "cpu_threads": torch.get_num_threads(),
    }
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "name": p.name,
                    "memory_gib": p.total_memory / GIB,
                    "capability": f"{p.major}.{p.minor}",
                    "sms": p.multi_processor_count,
                }
            )
        try:
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass
    return info


def collect() -> dict[str, Any]:
    """Everything the report shows, as a plain dict."""
    ram = ram_bytes()
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "cpu_logical": os.cpu_count(),
        "ram_gib": ram / GIB if ram else None,
        "python": f"{platform.python_version()} ({platform.python_implementation()})",
        "python_exe": sys.executable,
        "torch": torch_info(),
        "nvidia_smi": nvidia_smi(),
    }


def max_params_inference(mem_gib: float) -> float:
    """Largest bf16 model (parameters) that fits `mem_gib` for inference (estimate)."""
    usable = max(mem_gib - INFER_RESERVE_GIB, 0.0) * GIB
    return usable / (INFER_BYTES_PER_PARAM * (1 + INFER_OVERHEAD))


def max_params_training(mem_gib: float) -> float:
    """Largest model fully trained with AdamW (16 B/param, 50 % headroom; estimate)."""
    return mem_gib * GIB * TRAIN_HEADROOM / TRAIN_BYTES_PER_PARAM


def inference_gib(params: float) -> float:
    return params * INFER_BYTES_PER_PARAM * (1 + INFER_OVERHEAD) / GIB + INFER_RESERVE_GIB


def training_gib(params: float) -> float:
    return params * TRAIN_BYTES_PER_PARAM / GIB / TRAIN_HEADROOM


def _yes(ok: bool) -> str:
    return "yes" if ok else "no"


def _fmt_params(n: float) -> str:
    return f"{n / 1e9:.2f} B" if n >= 1e9 else f"{n / 1e6:.0f} M"


def _gpu_rows(info: dict[str, Any]) -> list[dict[str, Any]]:
    """GPUs from torch, else from nvidia-smi (torch may be a CPU build)."""
    rows = list(info["torch"].get("gpus", []))
    if not rows and info.get("nvidia_smi"):
        rows = [
            {"name": g["name"], "memory_gib": g["memory_gib"], "capability": "?", "sms": "?"}
            for g in info["nvidia_smi"]["gpus"]
        ]
    return rows


def render_markdown(info: dict[str, Any]) -> str:
    """The report as markdown."""
    t, smi = info["torch"], info.get("nvidia_smi")
    gpus = _gpu_rows(info)
    ram = info["ram_gib"]
    lines = [
        "# System report (Phase 0.1)",
        "",
        f"Generated {info['date']} by `tools/system_report.py`.",
        "",
        "## Machine",
        "",
        "| item | value |",
        "|---|---|",
        f"| OS | {info['os']} |",
        f"| CPU | {info['cpu']} ({info['cpu_logical']} logical cores, {info['machine']}) |",
        f"| RAM | {f'{ram:.1f} GiB' if ram else 'unknown'} |",
        f"| Python | {info['python']} |",
    ]
    if t["installed"]:
        cuda = (
            f"available (build {t['cuda_build']}, cudnn {t['cudnn']})"
            if t["cuda_available"]
            else f"not available (build {t['cuda_build'] or 'CPU-only'})"
        )
        lines += [
            f"| torch | {t['version']} |",
            f"| torch CUDA | {cuda} |",
            f"| bf16 on GPU | {'yes' if t['bf16'] else 'no'} |",
            f"| torch CPU threads | {t['cpu_threads']} |",
        ]
    else:
        lines.append("| torch | not installed |")
    if smi and smi.get("cuda_version"):
        lines.append(f"| driver CUDA version (nvidia-smi) | {smi['cuda_version']} |")

    lines += ["", "## GPU", ""]
    if gpus:
        lines += [
            "| # | name | VRAM GiB | compute capability | SMs |",
            "|---|---|---|---|---|",
        ]
        for i, g in enumerate(gpus):
            vram = f"{g['memory_gib']:.1f}" if g["memory_gib"] else "?"
            lines.append(f"| {i} | {g['name']} | {vram} | {g['capability']} | {g['sms']} |")
    else:
        lines.append("No CUDA GPU found by torch or nvidia-smi (CPU-only machine or CPU torch).")
    if smi and smi.get("csv"):
        lines += ["", "`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`:", ""]
        lines += ["```", smi["csv"], "```"]
    elif smi and smi.get("error"):
        lines += ["", f"nvidia-smi: {smi['error']}"]
    elif smi is None:
        lines += ["", "nvidia-smi: not found on PATH."]

    lines += _capacity_section(gpus, ram)
    return "\n".join(lines) + "\n"


def _capacity_section(gpus: list[dict[str, Any]], ram_gib: float | None) -> list[str]:
    usable = [g for g in gpus if g["memory_gib"]]
    if usable:
        target, label = max(usable, key=lambda g: g["memory_gib"]), "VRAM"
        mem = float(target["memory_gib"])
        what = f"GPU `{target['name']}` ({mem:.1f} GiB VRAM)"
    elif ram_gib:
        mem, label, what = ram_gib, "RAM", f"CPU only, system RAM ({ram_gib:.1f} GiB)"
    else:
        return ["", "## Capacity estimate", "", "No memory size known, no estimate possible."]
    lines = [
        "",
        "## Capacity estimate (ESTIMATE, to be replaced by measurement)",
        "",
        f"Basis: {what}. Formula-based only; real limits depend on the model, sequence "
        f"length, batch size and framework overhead. Measure with `tools/measure_latency.py` "
        f"and the `[model]`/`[speed]` lines of `tmagent.train.train_bc`.",
        "",
        f"- bf16 inference: ~{INFER_BYTES_PER_PARAM:.0f} B/param + {INFER_OVERHEAD:.0%} overhead"
        f" + {INFER_RESERVE_GIB:.0f} GiB reserve -> max "
        f"**{_fmt_params(max_params_inference(mem))}** parameters",
        f"- full AdamW training: ~{TRAIN_BYTES_PER_PARAM:.0f} B/param (fp32 weights, grads, two "
        f"moments) with {TRAIN_HEADROOM:.0%} of the memory kept free for activations -> max "
        f"**{_fmt_params(max_params_training(mem))}** parameters",
    ]
    if label == "RAM":
        lines.append("- CPU numbers are memory limits only; speed on CPU is far below GPU speed.")
    lines += [
        "",
        f"| parameters | bf16 inference {label} GiB | AdamW training {label} GiB "
        "(incl. headroom) | inference fits | training fits |",
        "|---|---|---|---|---|",
    ]
    for n in REFERENCE_PARAMS:
        i_gib, t_gib = inference_gib(n), training_gib(n)
        lines.append(
            f"| {_fmt_params(n)} | {i_gib:.1f} | {t_gib:.1f} | {_yes(i_gib <= mem)} "
            f"| {_yes(t_gib <= mem)} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT), help=f"markdown output (default {DEFAULT_OUT})"
    )
    args = ap.parse_args(argv)
    text = render_markdown(collect())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
