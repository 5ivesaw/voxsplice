"""
dsp.py — High-Speed Digital Signal Processing (DSP), ITU-T G.711 8-Bit μ-Law Codec
with Zero-Phase Butterworth Reconstruction Filter, Voice Activity Detection (VAD),
Smart Valley Utterance Splitter, and Pitch/Timbre Analysis.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.signal import butter as _scipy_butter
    from scipy.signal import filtfilt as _scipy_filtfilt
    from scipy.signal import resample_poly as _scipy_resample_poly
except ImportError:  # pragma: no cover
    _scipy_butter = None
    _scipy_filtfilt = None
    _scipy_resample_poly = None

# =============================================================================
# 1. Precomputed 64 KB Lookup Tables for ITU-T G.711 8-bit μ-Law (Zero CPU Cost)
# =============================================================================

_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def _build_ulaw_tables() -> Tuple[np.ndarray, np.ndarray]:
    u_vals = np.arange(256, dtype=np.uint8)
    u_inv = ~u_vals
    sign = u_inv & 0x80
    exponent = (u_inv >> 4) & 0x07
    mantissa = u_inv & 0x0F
    sample = ((mantissa.astype(np.int32) << 3) + _ULAW_BIAS) << exponent
    sample -= _ULAW_BIAS
    decode_lut = np.where(sign != 0, -sample, sample).astype(np.int16)

    raw_u16 = np.arange(65536, dtype=np.uint16)
    pcm_i16 = raw_u16.view(np.int16).astype(np.int32)

    sign_bit = np.where(pcm_i16 < 0, 0x80, 0x00).astype(np.uint8)
    mag = np.minimum(np.abs(pcm_i16), _ULAW_CLIP) + _ULAW_BIAS

    shifted = mag >> 7
    exp = np.zeros_like(shifted, dtype=np.int32)
    for e in range(1, 8):
        exp = np.where(shifted >= (1 << e), e, exp)

    mant = (mag >> (exp + 3)) & 0x0F
    ulaw_byte = ~(sign_bit | (exp.astype(np.uint8) << 4) | mant.astype(np.uint8)) & 0xFF
    encode_lut = ulaw_byte.astype(np.uint8)
    return encode_lut, decode_lut


_ULAW_ENCODE_LUT, _ULAW_DECODE_LUT = _build_ulaw_tables()


def encode_ulaw(pcm_int16: np.ndarray) -> bytes:
    arr = np.asarray(pcm_int16, dtype=np.int16)
    u16_idx = arr.view(np.uint16)
    return _ULAW_ENCODE_LUT[u16_idx].tobytes()


def decode_ulaw(ulaw_bytes: bytes | np.ndarray) -> np.ndarray:
    if isinstance(ulaw_bytes, (bytes, bytearray, memoryview)):
        idx = np.frombuffer(ulaw_bytes, dtype=np.uint8)
    else:
        idx = np.asarray(ulaw_bytes, dtype=np.uint8)
    return _ULAW_DECODE_LUT[idx].copy()


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float32(pcm_int16: np.ndarray) -> np.ndarray:
    return np.asarray(pcm_int16, dtype=np.float32) / 32768.0


def apply_ulaw_reconstruction_filter(
    samples_f32: np.ndarray, sample_rate: int = 16000, cutoff_hz: float = 3850.0
) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth reconstruction low-pass filter that eliminates
    high-frequency 8-bit μ-law staircase quantization harmonics above `cutoff_hz`.
    Preserves crisp vocal formants and plosive bursts while preventing upper-band quantization hiss.
    """
    if len(samples_f32) < 32 or sample_rate <= int(cutoff_hz * 2.05):
        return samples_f32
    if _scipy_butter is not None and _scipy_filtfilt is not None:
        nyq = 0.5 * float(sample_rate)
        norm_cut = min(0.95, max(0.10, cutoff_hz / nyq))
        b, a = _scipy_butter(4, norm_cut, btype="low")
        return _scipy_filtfilt(b, a, samples_f32).astype(np.float32)
    return np.convolve(samples_f32, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")


# =============================================================================
# 2. Polyphase Anti-Aliased Resampling & Vocal Bandpass Filter
# =============================================================================


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(samples) == 0:
        return samples.copy()

    orig_dtype = samples.dtype
    x = samples.astype(np.float32)

    if _scipy_resample_poly is not None and len(x) > 8:
        import math

        g = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        if up <= 64 and down <= 64:
            out = _scipy_resample_poly(x, up, down)
            if orig_dtype == np.int16:
                return np.clip(np.round(out), -32768, 32767).astype(np.int16)
            return out.astype(orig_dtype)

    if target_sr < orig_sr and len(x) > 4:
        x = np.convolve(x, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")

    duration = len(x) / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))
    orig_idx = np.linspace(0.0, len(x) - 1, num=len(x), dtype=np.float32)
    target_idx = np.linspace(0.0, len(x) - 1, num=target_len, dtype=np.float32)
    out = np.interp(target_idx, orig_idx, x)

    if orig_dtype == np.int16:
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)
    return out.astype(orig_dtype)


def vocal_bandpass_filter(
    samples_f32: np.ndarray,
    sample_rate: int = 16000,
    low_hz: float = 85.0,
    high_hz: float = 5600.0,
) -> np.ndarray:
    """
    Full-voice preservation bandpass filter (85 Hz – 5,600 Hz default):
    Removes sub-85 Hz DC/hum rumble and ultra-high hiss while preserving the speaker's
    fundamental frequency f0 (90–260 Hz), first formant F1, and sibilant/plosive clarity.
    """
    n = len(samples_f32)
    if n < 32:
        return samples_f32.copy()

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(samples_f32)

    mask = np.zeros_like(freqs, dtype=np.float32)
    passband = (freqs >= low_hz) & (freqs <= high_hz)
    mask[passband] = 1.0

    taper_low = 35.0
    low_taper = (freqs >= max(15.0, low_hz - taper_low)) & (freqs < low_hz)
    if np.any(low_taper):
        mask[low_taper] = 0.5 * (
            1.0 - np.cos(np.pi * (freqs[low_taper] - (low_hz - taper_low)) / taper_low)
        )

    taper_high = 250.0
    high_taper = (freqs > high_hz) & (freqs <= high_hz + taper_high)
    if np.any(high_taper):
        mask[high_taper] = 0.5 * (
            1.0 + np.cos(np.pi * (freqs[high_taper] - high_hz) / taper_high)
        )

    filtered = np.fft.irfft(spectrum * mask, n=n)
    return filtered.astype(np.float32)


# =============================================================================
# 3. Acoustic Metrics: RMS, ZCR, Spectral Flatness, VAD & Valley Splitter
# =============================================================================


def compute_rms(samples_f32: np.ndarray) -> float:
    if len(samples_f32) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples_f32, dtype=np.float64))))


def compute_zcr(samples_f32: np.ndarray) -> float:
    if len(samples_f32) < 4:
        return 0.0
    signs = np.signbit(samples_f32)
    return float(np.mean(signs[:-1] != signs[1:]))


def compute_spectral_flatness(samples_f32: np.ndarray, sample_rate: int = 16000) -> float:
    n = len(samples_f32)
    if n < 64:
        return 0.0

    windowed = samples_f32 * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    band = (freqs >= 300.0) & (freqs <= 3400.0)
    power = np.square(mag[band], dtype=np.float64) + 1e-12
    if len(power) == 0:
        return 0.0

    geom_mean = float(np.exp(np.mean(np.log(power))))
    arith_mean = float(np.mean(power))
    if arith_mean <= 1e-12:
        return 0.0
    return float(np.clip(geom_mean / arith_mean, 0.0, 1.0))


def is_human_speech_buffer(
    samples_f32: np.ndarray,
    sample_rate: int = 16000,
    min_rms: float = 0.007,
    max_flatness: float = 0.45,
) -> bool:
    if len(samples_f32) < int(sample_rate * 0.07):
        return False
    rms = compute_rms(samples_f32)
    if rms < min_rms:
        return False
    flat = compute_spectral_flatness(samples_f32, sample_rate=sample_rate)
    if flat > max_flatness:
        return False

    n = len(samples_f32)
    mag2 = np.square(np.abs(np.fft.rfft(samples_f32)), dtype=np.float64)
    total_e = float(np.sum(mag2)) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    vocal_e = float(np.sum(mag2[(freqs >= 180.0) & (freqs <= 3800.0)]))
    return (vocal_e / total_e) >= 0.28


def split_stream_into_utterances(
    pcm16: np.ndarray,
    sample_rate: int = 16000,
    pause_ms: float = 180.0,
    max_utt_s: float = 4.5,
) -> List[Tuple[int, int]]:
    frame_len = max(64, int(sample_rate * 0.020))
    n_frames = len(pcm16) // frame_len
    if n_frames < 5:
        return [(0, len(pcm16))] if len(pcm16) > 0 else []

    frames = pcm16[: n_frames * frame_len].reshape(n_frames, frame_len).astype(np.float32)
    rms_frames = np.sqrt(np.mean(frames * frames, axis=1))
    noise_floor = float(np.percentile(rms_frames, 10))
    peak_rms = float(np.percentile(rms_frames, 90))
    thresh = min(max(160.0, peak_rms * 0.25), max(160.0, noise_floor * 2.1))

    pause_frames = max(4, int(round(pause_ms / 20.0)))
    max_frames = max(50, int(round(max_utt_s / 0.020)))
    valley_search_frames = min(75, max_frames // 3)

    utterances: List[Tuple[int, int]] = []
    i = 0
    while i < n_frames:
        if rms_frames[i] < thresh:
            i += 1
            continue
        start_f = max(0, i - 4)
        j = i + 1
        silence_cnt = 0
        while j < n_frames:
            if rms_frames[j] < thresh:
                silence_cnt += 1
                if silence_cnt >= pause_frames:
                    break
            else:
                silence_cnt = 0
            if (j - start_f) >= max_frames:
                win_lo = max(start_f + 20, j - valley_search_frames)
                window = rms_frames[win_lo : j + 1]
                best_rel = int(np.argmin(window))
                j = win_lo + best_rel
                break
            j += 1
        end_f = min(n_frames, j + 3)
        if (end_f - start_f) >= 12:
            utterances.append((start_f * frame_len, min(len(pcm16), end_f * frame_len)))
        i = max(i + 1, end_f)

    if not utterances and len(pcm16) > 0:
        utterances.append((0, len(pcm16)))
    return utterances


def snap_to_zero_crossing(samples_f32: np.ndarray, target_idx: int, radius: int = 128) -> int:
    n = len(samples_f32)
    if n < 4:
        return max(0, min(n, target_idx))
    lo = max(0, target_idx - radius)
    hi = min(n - 1, target_idx + radius)
    if hi <= lo + 1:
        return max(0, min(n, target_idx))

    seg = samples_f32[lo:hi]
    signs = np.signbit(seg)
    crossings = np.where(signs[:-1] != signs[1:])[0]
    if len(crossings) == 0:
        return lo + int(np.argmin(np.abs(seg)))
    best_c = min(crossings, key=lambda c: abs((lo + int(c)) - target_idx))
    return lo + int(best_c)


def apply_micro_edge_fade(
    samples_f32: np.ndarray, sample_rate: int = 16000, fade_ms: float = 2.0
) -> np.ndarray:
    if len(samples_f32) < 16:
        return samples_f32.copy()
    out = samples_f32 - float(np.mean(samples_f32))
    fade_len = min(len(out) // 6, max(6, int(round((fade_ms / 1000.0) * sample_rate))))
    t = np.linspace(0.0, np.pi, fade_len, dtype=np.float32)
    win_in = 0.5 * (1.0 - np.cos(t))
    win_out = 1.0 - win_in
    out[:fade_len] *= win_in
    out[-fade_len:] *= win_out
    return out.astype(np.float32)


def estimate_pitch_and_centroid(
    samples_f32: np.ndarray, sample_rate: int = 16000
) -> Tuple[float, float]:
    n = len(samples_f32)
    if n < int(sample_rate * 0.025):
        return 140.0, 1200.0

    windowed = samples_f32 * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mag_sum = float(np.sum(mag))
    centroid_hz = float(np.sum(freqs * mag) / mag_sum) if mag_sum > 1e-9 else 1200.0

    frame_len = min(n, int(sample_rate * 0.045))
    hop = max(frame_len // 2, int(sample_rate * 0.020))
    min_lag = max(2, int(sample_rate / 400.0))
    max_lag = min(frame_len // 2 - 1, int(sample_rate / 65.0))

    if max_lag <= min_lag:
        return 140.0, round(centroid_hz, 2)

    pitches: List[float] = []
    for start in range(0, max(1, n - frame_len + 1), hop):
        frame = samples_f32[start : start + frame_len]
        frame = frame - float(np.mean(frame))
        energy = float(np.sum(frame * frame))
        if energy < 1e-5:
            continue

        fft_size = 1 << (2 * frame_len - 1).bit_length()
        f_spec = np.fft.rfft(frame, n=fft_size)
        acf = np.fft.irfft(f_spec * np.conj(f_spec), n=fft_size)[:frame_len]
        if acf[0] <= 1e-8:
            continue
        acf_norm = acf / acf[0]

        search_region = acf_norm[min_lag : max_lag + 1]
        if len(search_region) == 0:
            continue
        best_rel = int(np.argmax(search_region))
        peak_val = float(search_region[best_rel])
        best_lag = min_lag + best_rel

        if peak_val >= 0.28:
            if 0 < best_lag < len(acf_norm) - 1:
                y0, y1, y2 = acf_norm[best_lag - 1], acf_norm[best_lag], acf_norm[best_lag + 1]
                denom = 2.0 * (2.0 * y1 - y0 - y2)
                if abs(denom) > 1e-7:
                    delta = float(np.clip((y0 - y2) / denom, -0.5, 0.5))
                    best_lag_f = best_lag + delta
                else:
                    best_lag_f = float(best_lag)
            else:
                best_lag_f = float(best_lag)

            f0 = sample_rate / max(1.0, best_lag_f)
            if 65.0 <= f0 <= 400.0:
                pitches.append(f0)

    pitch_hz = float(np.median(pitches)) if pitches else 140.0
    return round(pitch_hz, 2), round(centroid_hz, 2)
