"""
config.py — Cross-platform configuration, TOML persistence, and runtime settings for VoxSplice.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


APP_NAME = "voxsplice"
LEGACY_APP_NAME = "5mic"
APP_VERSION = "2.4.0"

AUDIO_FORMAT_SPECS: Dict[str, Dict[str, Any]] = {
    "ulaw_12k": {
        "sample_rate": 12000,
        "bytes_per_sample": 1,
        "encoding": "ulaw",
        "kbps": 12.0,
        "mb_per_min": 0.72,
        "label": "12 kHz 8-bit ITU-T G.711 μ-Law (12 KB/s — Crisp + Subtle Retro Grit)",
    },
    "ulaw_8k": {
        "sample_rate": 8000,
        "bytes_per_sample": 1,
        "encoding": "ulaw",
        "kbps": 8.0,
        "mb_per_min": 0.48,
        "label": "8 kHz 8-bit ITU-T G.711 μ-Law (8 KB/s — Ultra-Compact Walkie-Talkie)",
    },
    "pcm_16k": {
        "sample_rate": 16000,
        "bytes_per_sample": 2,
        "encoding": "pcm16",
        "kbps": 32.0,
        "mb_per_min": 1.92,
        "label": "16 kHz 16-bit Linear PCM (32 KB/s — Uncompressed Studio Fidelity)",
    },
}

VOICE_MODES: Dict[str, Dict[str, str]] = {
    "natural": {
        "label": "Natural Flow",
        "description": "Minimizes pitch and spectral centroid jumps across seams via Viterbi DP planning.",
    },
    "low_pitch": {
        "label": "Low Pitch (< 130 Hz)",
        "description": "Prioritizes naturally deep-voiced clips without artificial pitch-shifting DSP.",
    },
    "high_pitch": {
        "label": "High Pitch (> 180 Hz)",
        "description": "Prioritizes naturally bright or high-pitched clips from captured speakers.",
    },
    "victim_lock": {
        "label": "Victim Lock (Single Speaker)",
        "description": "Locks onto the (f0, spectral_centroid) cluster of a target speaker in the lobby.",
    },
    "chaos": {
        "label": "Frankenstein Chaos",
        "description": "Maximizes speaker, pitch, and timbre transitions between every adjacent phrase chunk.",
    },
    "random": {
        "label": "Randomized Selection",
        "description": "Samples stochastically among top matching clips for unpredictable cadence.",
    },
}

FX_MODES: Dict[str, Dict[str, str]] = {
    "clean": {
        "label": "Clean Splice",
        "description": "Phase-aligned equal-power Hann crossfade with RMS loudness normalization.",
    },
    "retro_8bit": {
        "label": "Retro 8-Bit μ-Law Crunch",
        "description": "Authentic logarithmic 8-bit quantization with gentle 3.2 kHz rolloff.",
    },
    "radio": {
        "label": "Walkie-Talkie Radio",
        "description": "350–3,100 Hz comms bandpass with soft-knee overdrive and carrier squelch tail.",
    },
    "scratch": {
        "label": "Tape / Vinyl Micro-Scratch",
        "description": "Applies brief tape-shuttle pitch scrub at phrase boundaries.",
    },
    "corrupt": {
        "label": "Skinwalker Corruptify",
        "description": "Glitch micro-stutter, ring modulation, and sample-hold decimation.",
    },
}


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent.parent


def get_user_data_dir() -> Path:
    env_override = os.environ.get("VOXSPLICE_DATA_DIR") or os.environ.get("FIVEMIC_DATA_DIR")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (Path.home() / "AppData" / "Roaming")
        path = base / APP_NAME
    elif system == "Darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        xdg_data = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg_data) if xdg_data else (Path.home() / ".local" / "share")
        path = base / APP_NAME

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_config_dir() -> Path:
    env_override = os.environ.get("VOXSPLICE_CONFIG_DIR") or os.environ.get("FIVEMIC_CONFIG_DIR")
    if env_override:
        path = Path(env_override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    system = platform.system()
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (Path.home() / "AppData" / "Roaming")
        path = base / APP_NAME
    elif system == "Darwin":
        path = Path.home() / "Library" / "Preferences" / APP_NAME
    else:
        xdg_cfg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg_cfg) if xdg_cfg else (Path.home() / ".config")
        path = base / APP_NAME

    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class VoxSpliceConfig:
    language: str = "en-us"
    stt_model_id: str = "auto"
    embedding_model_id: str = "cmu-arpabet-hybrid"
    tokenizer_id: str = "cmu-arpabet-g2p"
    vosk_model_path: str = "models/default-en"

    # Audio & Storage Settings
    audio_format: str = "ulaw_12k"
    capture_sample_rate: int = 16000
    max_arena_mb: int = 200
    max_slots_per_phrase: int = 5
    use_starter_pack_fallback: bool = True
    enable_subword_splicing: bool = True

    # Acoustic Slicer & Bleed Detection Settings
    silence_gap_ms: float = 25.0
    seam_window_ms: float = 20.0
    seam_rms_threshold: float = 0.28
    max_bleed_chain_words: int = 3
    always_extract_single_words: bool = True
    min_word_confidence: float = 0.65
    enable_bandpass_filter: bool = True
    bandpass_low_hz: float = 300.0
    bandpass_high_hz: float = 3400.0
    max_spectral_flatness: float = 0.42
    min_word_rms: float = 0.008
    enable_vad_gate: bool = True

    # Stitcher & Voice Modes
    voice_mode: str = "natural"
    fx_mode: str = "clean"
    fx_intensity: float = 0.45
    semantic_similarity_cutoff: float = 0.56
    low_pitch_cutoff_hz: float = 130.0
    high_pitch_cutoff_hz: float = 180.0
    target_rms: float = 0.14
    max_gain_boost: float = 3.5
    crossfade_ms: float = 15.0
    locked_victim_id: Optional[str] = None

    # Routing & Feedback Safety
    virtual_mic_name: str = "MimicMic"
    half_duplex_cooldown_ms: float = 260.0
    monitor_self: bool = True
    udp_host: str = "127.0.0.1"
    udp_port: int = 9999
    api_port: int = 18420

    # UI Overlay Settings
    pill_always_on_top: bool = True
    pill_opacity: float = 0.96
    pill_width: int = 580
    pill_height: int = 76
    launch_overlay_with_tui: bool = False

    @property
    def arena_sample_rate(self) -> int:
        spec = AUDIO_FORMAT_SPECS.get(self.audio_format, AUDIO_FORMAT_SPECS["ulaw_12k"])
        return int(spec["sample_rate"])

    @property
    def arena_encoding(self) -> str:
        spec = AUDIO_FORMAT_SPECS.get(self.audio_format, AUDIO_FORMAT_SPECS["ulaw_12k"])
        return str(spec["encoding"])

    @property
    def max_arena_bytes(self) -> int:
        return int(self.max_arena_mb) * 1024 * 1024

    def resolve_arena_path(self) -> Path:
        return get_user_data_dir() / "audio_arena.pcm"

    def resolve_db_path(self) -> Path:
        return get_user_data_dir() / "clips.db"

    def resolve_models_dir(self) -> Path:
        d = get_user_data_dir() / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve_asset_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute() and p.exists():
            return p
        candidates = [
            get_base_dir() / rel_path,
            get_user_data_dir() / rel_path,
            Path.cwd() / rel_path,
        ]
        for c in candidates:
            if c.exists():
                return c
        return get_base_dir() / rel_path

    def to_toml_string(self) -> str:
        lines = [
            f"# VoxSplice v{APP_VERSION} Configuration",
            "",
            "[general]",
            f'language = "{self.language}"',
            f'stt_model_id = "{self.stt_model_id}"',
            f'embedding_model_id = "{self.embedding_model_id}"',
            f'tokenizer_id = "{self.tokenizer_id}"',
            f'vosk_model_path = "{self.vosk_model_path}"',
            "",
            "[storage]",
            f'audio_format = "{self.audio_format}"',
            f"capture_sample_rate = {self.capture_sample_rate}",
            f"max_arena_mb = {self.max_arena_mb}",
            f"max_slots_per_phrase = {self.max_slots_per_phrase}",
            f"use_starter_pack_fallback = {'true' if self.use_starter_pack_fallback else 'false'}",
            f"enable_subword_splicing = {'true' if self.enable_subword_splicing else 'false'}",
            "",
            "[slicer]",
            f"silence_gap_ms = {self.silence_gap_ms}",
            f"seam_window_ms = {self.seam_window_ms}",
            f"seam_rms_threshold = {self.seam_rms_threshold}",
            f"max_bleed_chain_words = {self.max_bleed_chain_words}",
            f"always_extract_single_words = {'true' if self.always_extract_single_words else 'false'}",
            f"min_word_confidence = {self.min_word_confidence}",
            f"enable_bandpass_filter = {'true' if self.enable_bandpass_filter else 'false'}",
            f"bandpass_low_hz = {self.bandpass_low_hz}",
            f"bandpass_high_hz = {self.bandpass_high_hz}",
            f"max_spectral_flatness = {self.max_spectral_flatness}",
            f"min_word_rms = {self.min_word_rms}",
            f"enable_vad_gate = {'true' if self.enable_vad_gate else 'false'}",
            "",
            "[stitcher]",
            f'voice_mode = "{self.voice_mode}"',
            f'fx_mode = "{self.fx_mode}"',
            f"fx_intensity = {self.fx_intensity}",
            f"semantic_similarity_cutoff = {self.semantic_similarity_cutoff}",
            f"low_pitch_cutoff_hz = {self.low_pitch_cutoff_hz}",
            f"high_pitch_cutoff_hz = {self.high_pitch_cutoff_hz}",
            f"target_rms = {self.target_rms}",
            f"max_gain_boost = {self.max_gain_boost}",
            f"crossfade_ms = {self.crossfade_ms}",
            "",
            "[routing]",
            f'virtual_mic_name = "{self.virtual_mic_name}"',
            f"half_duplex_cooldown_ms = {self.half_duplex_cooldown_ms}",
            f"monitor_self = {'true' if self.monitor_self else 'false'}",
            f'udp_host = "{self.udp_host}"',
            f"udp_port = {self.udp_port}",
            f"api_port = {self.api_port}",
            "",
            "[ui]",
            f"pill_always_on_top = {'true' if self.pill_always_on_top else 'false'}",
            f"pill_opacity = {self.pill_opacity}",
            f"pill_width = {self.pill_width}",
            f"pill_height = {self.pill_height}",
            f"launch_overlay_with_tui = {'true' if self.launch_overlay_with_tui else 'false'}",
            "",
        ]
        return "\n".join(lines)

    def save(self, path: Optional[Path] = None) -> Path:
        target = path or (get_user_config_dir() / "config.toml")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_toml_string(), encoding="utf-8")
        return target

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "VoxSpliceConfig":
        cfg = cls()
        target = path or (get_user_config_dir() / "config.toml")
        if not target.exists():
            default_path = get_base_dir() / "config.default.toml"
            if default_path.exists():
                target = default_path
            else:
                return cfg

        if tomllib is None:
            return cfg

        try:
            raw = tomllib.loads(target.read_text(encoding="utf-8"))
            for section in ("general", "storage", "slicer", "stitcher", "routing", "ui"):
                sub = raw.get(section, {})
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)
        except Exception:
            pass
        return cfg
