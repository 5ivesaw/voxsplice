"""
fx_rack.py — Real-Time DSP Effects & Corruptify Rack for VoxSplice.

Modes:
- `clean`      : Unmodified studio concatenation (only RMS normalization + phase-aligned crossfade).
- `retro_8bit` : Authentic 8-bit ITU-T G.711 μ-law quantization with gentle 3.3 kHz rolloff.
- `radio`      : 350–3,100 Hz comms bandpass, soft-knee overdrive, and carrier squelch tail.
- `scratch`    : Brief vinyl/tape micro-scrubbing at chunk transitions.
- `corrupt`    : Skinwalker / horror-mimic mode — glitch micro-stutter, ring-mod, and bitcrush.
"""

from __future__ import annotations

import zlib
import numpy as np

from .dsp import decode_ulaw, encode_ulaw, float32_to_int16, int16_to_float32, vocal_bandpass_filter


def apply_retro_8bit_fx(
    samples_f32: np.ndarray, sample_rate: int = 16000, intensity: float = 0.45
) -> np.ndarray:
    if len(samples_f32) < 32:
        return samples_f32.copy()
    bp = vocal_bandpass_filter(samples_f32, sample_rate=sample_rate, low_hz=260.0, high_hz=3300.0)
    pcm16 = float32_to_int16(bp)
    ulaw = encode_ulaw(pcm16)
    dec = int16_to_float32(decode_ulaw(ulaw))
    return ((1.0 - 0.35 * intensity) * samples_f32 + (0.35 * intensity) * dec).astype(np.float32)


def apply_radio_fx(
    samples_f32: np.ndarray, sample_rate: int = 16000, intensity: float = 0.45
) -> np.ndarray:
    if len(samples_f32) < 32:
        return samples_f32.copy()
    bp = vocal_bandpass_filter(samples_f32, sample_rate=sample_rate, low_hz=380.0, high_hz=2900.0)
    drive = 1.0 + 1.8 * float(np.clip(intensity, 0.1, 1.0))
    saturated = np.tanh(bp * drive) * 0.85

    tail_len = min(len(saturated) // 6, int(sample_rate * 0.012))
    if tail_len > 8:
        seed = zlib.crc32(samples_f32[: min(32, len(samples_f32))].tobytes())
        rng = np.random.default_rng(seed)
        burst = rng.normal(0.0, 0.025 * intensity, tail_len).astype(np.float32)
        saturated[-tail_len:] = saturated[-tail_len:] * 0.65 + burst
    return saturated.astype(np.float32)


def apply_scratch_fx(
    samples_f32: np.ndarray,
    sample_rate: int = 16000,
    intensity: float = 0.45,
    chunk_index: int = 0,
) -> np.ndarray:
    n = len(samples_f32)
    if n < int(sample_rate * 0.09):
        return samples_f32.copy()

    out = samples_f32.copy()
    t = np.linspace(0.0, n / float(sample_rate), n, dtype=np.float32)
    wobble = 1.0 + 0.025 * intensity * np.sin(2.0 * np.pi * 6.5 * t + chunk_index)
    idx_curve = np.cumsum(wobble)
    idx_curve = idx_curve * ((n - 1) / max(1.0, float(idx_curve[-1])))
    out = np.interp(idx_curve, np.arange(n, dtype=np.float32), out).astype(np.float32)

    if chunk_index % 2 == 1:
        scrub_len = min(n // 5, int(sample_rate * 0.032))
        if scrub_len > 16:
            head = out[:scrub_len]
            rev = head[::-1] * 0.65
            win = np.hanning(len(rev)).astype(np.float32)
            out = np.concatenate([head * 0.85, rev * win, out[scrub_len:]]).astype(np.float32)
    return out


def apply_corrupt_fx(
    samples_f32: np.ndarray,
    sample_rate: int = 16000,
    intensity: float = 0.45,
    chunk_index: int = 0,
) -> np.ndarray:
    n = len(samples_f32)
    if n < 64:
        return samples_f32.copy()

    out = samples_f32.copy()
    step = max(1, int(round(1 + 3 * intensity)))
    if step > 1:
        held = out[::step]
        out = np.repeat(held, step)[:n]

    levels = max(16.0, 96.0 - 64.0 * intensity)
    out = np.round(out * levels) / levels

    t = np.arange(len(out), dtype=np.float32) / float(sample_rate)
    ring_hz = 48.0 + float((chunk_index * 29) % 45)
    ring = (1.0 - 0.24 * intensity) + (0.24 * intensity) * np.sin(2.0 * np.pi * ring_hz * t)
    out = (out * ring).astype(np.float32)

    if chunk_index % 3 == 1 and len(out) > int(sample_rate * 0.08):
        stutter_len = int(sample_rate * 0.026)
        stutter = out[:stutter_len] * np.hanning(stutter_len).astype(np.float32)
        out = np.concatenate([stutter, stutter, out]).astype(np.float32)

    return np.clip(out, -0.95, 0.95).astype(np.float32)


def process_chunk_fx(
    samples_f32: np.ndarray,
    fx_mode: str = "clean",
    sample_rate: int = 16000,
    intensity: float = 0.45,
    chunk_index: int = 0,
) -> np.ndarray:
    mode = (fx_mode or "clean").strip().lower()
    if mode == "clean" or len(samples_f32) == 0:
        return samples_f32
    if mode == "retro_8bit":
        return apply_retro_8bit_fx(samples_f32, sample_rate=sample_rate, intensity=intensity)
    if mode == "radio":
        return apply_radio_fx(samples_f32, sample_rate=sample_rate, intensity=intensity)
    if mode == "scratch":
        return apply_scratch_fx(
            samples_f32, sample_rate=sample_rate, intensity=intensity, chunk_index=chunk_index
        )
    if mode == "corrupt":
        return apply_corrupt_fx(
            samples_f32, sample_rate=sample_rate, intensity=intensity, chunk_index=chunk_index
        )
    return samples_f32
