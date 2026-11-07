"""
hardware.py — Real System Hardware Profiler & Intelligent Model Recommendation Engine.

Inspects actual host resources (CPU model, physical/logical cores, AVX2/NEON instruction sets,
total/available RAM, swap usage, GPU adapter, OS/Wayland/X11/Windows audio stack, disk space)
without fabricating benchmarks. Classifies systems into hardware tiers and provides grounded
recommendations for STT models, semantic embedding brains, tokenizers, and audio encoding formats.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HardwareProfile:
    os_name: str
    os_release: str
    arch: str
    display_server: str
    audio_subsystem: str
    cpu_model: str
    logical_cores: int
    physical_cores_est: int
    has_avx2: bool
    has_fma: bool
    has_neon: bool
    total_ram_mb: int
    available_ram_mb: int
    used_ram_percent: float
    swap_total_mb: int
    swap_used_mb: int
    gpu_model: str
    has_discrete_gpu: bool
    free_disk_gb: float
    pocketsphinx_c_available: bool
    vosk_python_available: bool
    faster_whisper_available: bool
    onnxruntime_available: bool
    model2vec_available: bool
    hardware_tier: str  # "low_end", "mid_range", "high_end"
    recommended_stt_model: str
    recommended_embedding_model: str
    recommended_tokenizer: str
    recommended_audio_format: str
    recommended_arena_cap_mb: int
    recommendation_rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _detect_cpu_and_flags() -> tuple[str, bool, bool, bool]:
    cpu_model = platform.processor() or "Unknown CPU"
    has_avx2 = False
    has_fma = False
    has_neon = "aarch64" in platform.machine().lower() or "arm" in platform.machine().lower()

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                elif line.lower().startswith("flags") and ":" in line:
                    flags = set(line.split(":", 1)[1].strip().lower().split())
                    has_avx2 = "avx2" in flags
                    has_fma = "fma" in flags
                elif line.lower().startswith("features") and ":" in line:
                    feats = set(line.split(":", 1)[1].strip().lower().split())
                    if "asimd" in feats or "neon" in feats:
                        has_neon = True
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True, timeout=2.0, stderr=subprocess.DEVNULL
            )
            lines = [ln.strip() for ln in out.splitlines() if ln.strip() and ln.strip().lower() != "name"]
            if lines:
                cpu_model = lines[0]
        except Exception:
            pass
        has_avx2 = "x86_64" in platform.machine().lower() or "amd64" in platform.machine().lower()

    return cpu_model, has_avx2, has_fma, has_neon


def _detect_memory_mb() -> tuple[int, int, int, int]:
    total_mb = 4096
    avail_mb = 1024
    swap_total_mb = 0
    swap_used_mb = 0

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            vals: Dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    num = "".join(ch for ch in v if ch.isdigit())
                    if num:
                        vals[k.strip()] = int(num) // 1024
            total_mb = vals.get("MemTotal", total_mb)
            avail_mb = vals.get("MemAvailable", vals.get("MemFree", avail_mb))
            swap_total_mb = vals.get("SwapTotal", 0)
            swap_free_mb = vals.get("SwapFree", 0)
            swap_used_mb = max(0, swap_total_mb - swap_free_mb)
            return total_mb, avail_mb, swap_total_mb, swap_used_mb
        except Exception:
            pass

    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            total_mb = int(stat.ullTotalPhys // (1024 * 1024))
            avail_mb = int(stat.ullAvailPhys // (1024 * 1024))
        except Exception:
            pass

    return total_mb, avail_mb, swap_total_mb, swap_used_mb


def _detect_gpu() -> tuple[str, bool]:
    gpu_model = "Integrated / Standard Graphics"
    has_discrete = False

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
                timeout=2.0,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return f"NVIDIA {out.splitlines()[0]}", True
        except Exception:
            pass

    if shutil.which("lspci"):
        try:
            out = subprocess.check_output(
                ["lspci"], text=True, timeout=2.0, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                low = line.lower()
                if "vga compatible controller" in low or "3d controller" in low or "display controller" in low:
                    parts = line.split(":", 2)
                    desc = parts[-1].strip() if len(parts) >= 3 else line.strip()
                    gpu_model = desc
                    if any(k in desc.lower() for k in ("nvidia", "geforce", "rtx", "gtx", "radeon rx", "arc a")):
                        has_discrete = True
                    break
        except Exception:
            pass

    return gpu_model, has_discrete


def _detect_display_and_audio() -> tuple[str, str]:
    system = platform.system()
    if system == "Windows":
        return "Win32 / DWM", "WASAPI"
    if system == "Darwin":
        return "Quartz Compositor", "CoreAudio"

    wayland = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "")
    disp = f"Wayland ({desktop})" if wayland and desktop else ("Wayland" if wayland else (f"X11 ({desktop})" if desktop else "X11 / Headless"))

    audio = "ALSA"
    if shutil.which("pw-cli") or shutil.which("pipewire"):
        audio = "PipeWire (PulseAudio Compat)" if shutil.which("pactl") else "PipeWire"
    elif shutil.which("pactl") or shutil.which("pulseaudio"):
        audio = "PulseAudio"

    return disp, audio


def _check_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _check_pocketsphinx_c() -> bool:
    candidates = [
        Path("/usr/lib/x86_64-linux-gnu/libpocketsphinx.so.3"),
        Path("/usr/lib64/libpocketsphinx.so.3"),
        Path("/usr/lib/libpocketsphinx.so.3"),
    ]
    model_dir = Path("/usr/share/pocketsphinx/model/en-us/en-us")
    return any(c.exists() for c in candidates) and model_dir.exists()


def inspect_system_hardware(target_dir: Optional[Path] = None) -> HardwareProfile:
    os_name = platform.system()
    os_rel = platform.release()
    arch = platform.machine()
    disp, audio = _detect_display_and_audio()
    cpu_model, has_avx2, has_fma, has_neon = _detect_cpu_and_flags()
    logical_cores = os.cpu_count() or 2
    physical_est = max(1, logical_cores // 2) if logical_cores >= 4 else logical_cores

    total_mb, avail_mb, swap_total_mb, swap_used_mb = _detect_memory_mb()
    used_pct = round(max(0.0, min(100.0, (1.0 - (avail_mb / max(1, total_mb))) * 100.0)), 1)
    gpu_model, has_discrete = _detect_gpu()

    check_path = target_dir if (target_dir and target_dir.exists()) else Path.home()
    try:
        disk_usage = shutil.disk_usage(check_path)
        free_disk_gb = round(disk_usage.free / (1024**3), 2)
    except Exception:
        free_disk_gb = 10.0

    ps_c = _check_pocketsphinx_c()
    vosk_avail = _check_module("vosk")
    fw_avail = _check_module("faster_whisper")
    ort_avail = _check_module("onnxruntime")
    m2v_avail = _check_module("model2vec")

    rationale: List[str] = []

    # Classify hardware tier based on real available memory and CPU capability
    if avail_mb < 1400 or logical_cores <= 4 and "i5-6" in cpu_model.lower():
        tier = "low_end"
        rec_audio = "ulaw_12k"
        rec_arena_mb = 150 if free_disk_gb > 5.0 else 64
        if ps_c and not vosk_avail:
            rec_stt = "pocketsphinx-en-us"
            rationale.append(
                f"Detected {avail_mb} MB available RAM ({used_pct}% used). Recommending native C "
                "PocketSphinx (`pocketsphinx-en-us`, ~38 MB RAM) which is already installed on this machine."
            )
        else:
            rec_stt = "vosk-small-en"
            rationale.append(
                f"Detected {avail_mb} MB available RAM on {cpu_model} ({logical_cores} threads). "
                "Recommending `vosk-small-en` (40 MB download, ~80 MB runtime RAM) + posterior confidence filtering (>= 0.65) to avoid swap pressure."
            )
        rec_emb = "cmu-arpabet-hybrid"
        rec_tok = "cmu-arpabet-g2p"
        rationale.append(
            "Using `cmu-arpabet-hybrid` static phonetic/lexical vectors (< 8 MB RAM) and 12 kHz 8-bit μ-law (12 KB/s) to keep total footprint under 130 MB."
        )
    elif avail_mb < 4096 or not has_discrete:
        tier = "mid_range"
        rec_audio = "ulaw_12k"
        rec_arena_mb = 200
        if ort_avail or fw_avail:
            rec_stt = "vosk-lgraph-en" if vosk_avail else ("pocketsphinx-en-us" if ps_c else "whisper-tiny-en")
        else:
            rec_stt = "vosk-lgraph-en" if vosk_avail else ("pocketsphinx-en-us" if ps_c else "vosk-small-en")
        rec_emb = "potion-base-8M" if m2v_avail else "cmu-arpabet-hybrid"
        rec_tok = "cmu-arpabet-g2p"
        rationale.append(
            f"Detected {avail_mb} MB free RAM and {logical_cores} logical CPU cores ({'AVX2 supported' if has_avx2 else 'standard ISA'}). "
            f"Recommending `{rec_stt}` for balanced transcription accuracy and low CPU overhead while gaming."
        )
        rationale.append(
            "12 kHz 8-bit μ-law arena encoding preserves recognizable voice timbre at 720 KB/minute."
        )
    else:
        tier = "high_end"
        rec_audio = "pcm_16k"
        rec_arena_mb = 350
        rec_stt = "whisper-base-en" if fw_avail else ("vosk-lgraph-en" if vosk_avail else ("pocketsphinx-en-us" if ps_c else "whisper-tiny-en"))
        rec_emb = "potion-base-32M" if m2v_avail else "potion-base-8M"
        rec_tok = "whisper-bpe-en"
        rationale.append(
            f"Detected {avail_mb} MB free RAM, {logical_cores} threads, and {gpu_model}. "
            f"System can comfortably run `{rec_stt}` with 16 kHz uncompressed PCM arena storage."
        )

    return HardwareProfile(
        os_name=os_name,
        os_release=os_rel,
        arch=arch,
        display_server=disp,
        audio_subsystem=audio,
        cpu_model=cpu_model,
        logical_cores=logical_cores,
        physical_cores_est=physical_est,
        has_avx2=has_avx2,
        has_fma=has_fma,
        has_neon=has_neon,
        total_ram_mb=total_mb,
        available_ram_mb=avail_mb,
        used_ram_percent=used_pct,
        swap_total_mb=swap_total_mb,
        swap_used_mb=swap_used_mb,
        gpu_model=gpu_model,
        has_discrete_gpu=has_discrete,
        free_disk_gb=free_disk_gb,
        pocketsphinx_c_available=ps_c,
        vosk_python_available=vosk_avail,
        faster_whisper_available=fw_avail,
        onnxruntime_available=ort_avail,
        model2vec_available=m2v_avail,
        hardware_tier=tier,
        recommended_stt_model=rec_stt,
        recommended_embedding_model=rec_emb,
        recommended_tokenizer=rec_tok,
        recommended_audio_format=rec_audio,
        recommended_arena_cap_mb=rec_arena_mb,
        recommendation_rationale=rationale,
    )
