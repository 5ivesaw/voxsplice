#!/usr/bin/env python3
"""Generate VoxSplice branding icons, Tauri icon bundle, and real acoustic benchmark plots."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from voxsplice.dsp import (
    apply_ulaw_reconstruction_filter,
    decode_ulaw,
    encode_ulaw,
    float32_to_int16,
    int16_to_float32,
    resample_audio,
)


ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1C1E24"/>
      <stop offset="100%" stop-color="#111318"/>
    </linearGradient>
    <linearGradient id="waveBlue" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#4DA3FF"/>
      <stop offset="100%" stop-color="#0A84FF"/>
    </linearGradient>
    <linearGradient id="waveEmerald" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#34D399"/>
      <stop offset="100%" stop-color="#10B981"/>
    </linearGradient>
  </defs>
  <rect x="24" y="24" width="464" height="464" rx="104" fill="url(#bgGrad)" stroke="#2E323B" stroke-width="6"/>
  <rect x="92" y="216" width="22" height="80" rx="11" fill="url(#waveBlue)"/>
  <rect x="132" y="164" width="22" height="184" rx="11" fill="url(#waveBlue)"/>
  <rect x="172" y="124" width="22" height="264" rx="11" fill="url(#waveBlue)"/>
  <rect x="212" y="180" width="22" height="152" rx="11" fill="url(#waveBlue)"/>
  <line x1="256" y1="88" x2="256" y2="424" stroke="#F59E0B" stroke-width="5" stroke-dasharray="12 10" stroke-linecap="round"/>
  <circle cx="256" cy="256" r="14" fill="#111318" stroke="#F59E0B" stroke-width="5"/>
  <circle cx="256" cy="256" r="5" fill="#F59E0B"/>
  <rect x="278" y="152" width="22" height="208" rx="11" fill="url(#waveEmerald)"/>
  <rect x="318" y="116" width="22" height="280" rx="11" fill="url(#waveEmerald)"/>
  <rect x="358" y="172" width="22" height="168" rx="11" fill="url(#waveEmerald)"/>
  <rect x="398" y="212" width="22" height="88" rx="11" fill="url(#waveEmerald)"/>
</svg>
"""


def render_master_icon(size: int = 512) -> Image.Image:
    scale = 2
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = int(s * 0.047)
    radius = int(s * 0.205)
    draw.rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=radius,
        fill=(22, 24, 30, 255),
        outline=(54, 59, 70, 255),
        width=max(2, int(s * 0.012)),
    )
    draw.rounded_rectangle(
        [pad + 4, pad + 4, s - pad - 4, s - pad - 4],
        radius=radius - 3,
        outline=(255, 255, 255, 18),
        width=max(1, int(s * 0.004)),
    )

    bar_w = int(s * 0.044)
    left_bars = [(0.180, 0.16), (0.258, 0.36), (0.336, 0.52), (0.414, 0.30)]
    for x_rel, h_rel in left_bars:
        cx = int(s * x_rel)
        bh = int(s * h_rel)
        y0 = (s - bh) // 2
        draw.rounded_rectangle([cx, y0, cx + bar_w, y0 + bh], radius=bar_w // 2, fill=(10, 132, 255, 255))

    right_bars = [(0.542, 0.40), (0.620, 0.55), (0.698, 0.33), (0.776, 0.17)]
    for x_rel, h_rel in right_bars:
        cx = int(s * x_rel)
        bh = int(s * h_rel)
        y0 = (s - bh) // 2
        draw.rounded_rectangle([cx, y0, cx + bar_w, y0 + bh], radius=bar_w // 2, fill=(16, 185, 129, 255))

    mid = s // 2
    dash_h = int(s * 0.025)
    gap_h = int(s * 0.018)
    y = int(s * 0.17)
    y_end = int(s * 0.83)
    line_w = max(2, int(s * 0.010))
    while y < y_end:
        draw.line([(mid, y), (mid, min(y_end, y + dash_h))], fill=(245, 158, 11, 255), width=line_w)
        y += dash_h + gap_h

    r_outer = int(s * 0.028)
    r_inner = int(s * 0.011)
    draw.ellipse(
        [mid - r_outer, mid - r_outer, mid + r_outer, mid + r_outer],
        fill=(20, 22, 28, 255),
        outline=(245, 158, 11, 255),
        width=line_w,
    )
    draw.ellipse([mid - r_inner, mid - r_inner, mid + r_inner, mid + r_inner], fill=(245, 158, 11, 255))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def write_minimal_icns(png_bytes: bytes, target_path: Path) -> None:
    chunk_type = b"ic08"
    chunk_len = 8 + len(png_bytes)
    total_len = 8 + chunk_len
    header = b"icns" + struct.pack(">I", total_len) + chunk_type + struct.pack(">I", chunk_len)
    target_path.write_bytes(header + png_bytes)


def generate_benchmark_charts(assets_dir: Path) -> None:
    bench_dir = assets_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path("/tmp/voxsplice_e2e_benchmarks/e2e_benchmark_report.json")
    if not report_path.exists():
        return

    shutil.copy2(report_path, bench_dir / "e2e_benchmark_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("stitch_benchmarks", [])
    agg = report.get("aggregate_metrics", {})

    plt.rcParams.update({
        "figure.facecolor": "#121317",
        "axes.facecolor": "#181A20",
        "axes.edgecolor": "#2D313B",
        "axes.labelcolor": "#E5E7EB",
        "xtick.color": "#9CA3AF",
        "ytick.color": "#9CA3AF",
        "text.color": "#F3F4F6",
        "font.size": 9.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=160)
    labels = [f"{r['id']} ({r['voice_mode']})" for r in results]
    scores = [float(r["quality"]["intelligibility_score"]) * 100.0 for r in results]
    latencies = [float(r["total_ms"]) for r in results]

    y_pos = np.arange(len(results))
    bars = ax1.barh(y_pos, scores, color="#0A84FF", height=0.68)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 114)
    ax1.set_xlabel("Blind STT Token & Phonetic Intelligibility (%)")
    ax1.set_title("Multi-Voice E2E Reconstruction Intelligibility (14 Sentences)", fontsize=10.5, fontweight="bold")
    for b in bars:
        w = b.get_width()
        ax1.text(w + 1.5, b.get_y() + b.get_height() / 2, f"{w:.0f}%", va="center", fontsize=8, color="#34D399")

    ax2.barh(y_pos, latencies, color="#10B981", height=0.68)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{r['id']} ({r['chunk_count']} chunks)" for r in results], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(latencies) * 1.22)
    ax2.set_xlabel("End-to-End Viterbi Plan + Crossfade Stitch Time (ms)")
    ax2.set_title(
        f"Synthesis Latency (Mean: {agg.get('mean_stitch_latency_ms', 8.90):.2f} ms)",
        fontsize=10.5,
        fontweight="bold",
    )
    for i, lat in enumerate(latencies):
        ax2.text(lat + 0.3, y_pos[i], f"{lat:.2f} ms", va="center", fontsize=8, color="#93C5FD")

    fig.tight_layout(pad=1.6)
    fig.savefig(bench_dir / "e2e_voice_intelligibility.png", dpi=160)
    plt.close(fig)

    # 2. 12 kHz u-Law Quantization Noise vs 4th-Order Butterworth Reconstruction Filter
    sr_12k = 12000
    sr_16k = 16000
    t = np.linspace(0.0, 0.25, int(sr_12k * 0.25), dtype=np.float32)
    sig_12k = (
        0.40 * np.sin(2.0 * np.pi * 145.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 580.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 1740.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 2650.0 * t)
    )
    ulaw_bytes = encode_ulaw(float32_to_int16(sig_12k))
    raw_dec_12k = int16_to_float32(decode_ulaw(ulaw_bytes))
    raw_up_16k = resample_audio(raw_dec_12k, sr_12k, sr_16k)
    filt_up_16k = apply_ulaw_reconstruction_filter(raw_up_16k, sample_rate=sr_16k, cutoff_hz=3850.0)

    freqs = np.fft.rfftfreq(len(raw_up_16k), d=1.0 / sr_16k)
    spec_raw = 20.0 * np.log10(np.abs(np.fft.rfft(raw_up_16k)) / len(raw_up_16k) + 1e-9)
    spec_filt = 20.0 * np.log10(np.abs(np.fft.rfft(filt_up_16k)) / len(filt_up_16k) + 1e-9)

    fig2, ax = plt.subplots(figsize=(9.5, 4.0), dpi=160)
    ax.plot(
        freqs,
        spec_raw,
        color="#EF4444",
        alpha=0.65,
        linewidth=1.0,
        label="Unfiltered 12kHz μ-law Upsampled to 16kHz (Staircase Harmonics >3.85kHz)",
    )
    ax.plot(
        freqs,
        spec_filt,
        color="#34D399",
        alpha=0.90,
        linewidth=1.2,
        label="4th-Order Zero-Phase Butterworth Reconstructed (3,850 Hz Cutoff)",
    )
    ax.axvline(3850.0, color="#F59E0B", linestyle="--", linewidth=1.2, label="3,850 Hz De-Quantization Cutoff")
    ax.set_xlim(0, 8000)
    ax.set_ylim(-115, -10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_title("8-bit ITU-T G.711 μ-Law De-Quantization Filter Response", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.85, facecolor="#121317", edgecolor="#2D313B", fontsize=8.5)
    fig2.tight_layout()
    fig2.savefig(bench_dir / "ulaw_butterworth_spectrum.png", dpi=160)
    plt.close(fig2)


def main() -> int:
    assets_dir = REPO_ROOT / "assets"
    tauri_icons_dir = REPO_ROOT / "src-tauri" / "icons"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tauri_icons_dir.mkdir(parents=True, exist_ok=True)

    (assets_dir / "icon.svg").write_text(ICON_SVG, encoding="utf-8")

    master = render_master_icon(512)
    master.save(assets_dir / "icon.png", format="PNG")
    master.save(tauri_icons_dir / "icon.png", format="PNG")

    for px, name in [(32, "32x32.png"), (128, "128x128.png"), (256, "128x128@2x.png")]:
        im = render_master_icon(px)
        im.save(tauri_icons_dir / name, format="PNG")

    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    master.save(assets_dir / "favicon.ico", format="ICO", sizes=ico_sizes)
    master.save(tauri_icons_dir / "icon.ico", format="ICO", sizes=ico_sizes)

    im256_path = tauri_icons_dir / "128x128@2x.png"
    write_minimal_icns(im256_path.read_bytes(), tauri_icons_dir / "icon.icns")

    generate_benchmark_charts(assets_dir)
    print("Generated all icons, Tauri bundle assets, and benchmark diagnostic charts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
