"""
stt_backends.py — Pluggable Multi-Backend Speech-to-Text & Word-Timestamp Engine.

Supports:
1. `pocketsphinx-en-us`: Native C library (`libpocketsphinx.so.3` + `libsphinxbase.so.3`) bound via
   Python `ctypes`. Zero external pip package required on Debian/Ubuntu/Pop!_OS; emits 10ms frame-accurate
   word timestamps and log-lattice posterior word probabilities (`ps_seg_prob`). Automatically pads 80ms
   of silence around each utterance so delta/double-delta MFCC context frames (`1s_c_d_dd`) never clip
   initial or final consonants. Also supports JSGF grammar-guided forced alignment (`ps_set_jsgf_string`).
2. `vosk-small-en` / `vosk-lgraph-en`: AlphaCephei Kaldi TDNN-F recognizer (`vosk` Python package).
3. `whisper-tiny-en-onnx` / `whisper-base-en-onnx`: CTranslate2 (`faster-whisper`) or ONNX Runtime INT8
   encoder-decoder with cross-attention word timestamps.
"""

from __future__ import annotations

import ctypes
import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import VoxSpliceConfig
from .dsp import (
    int16_to_float32,
    is_human_speech_buffer,
    resample_audio,
    split_stream_into_utterances,
)
from .model_hub import ModelHubManager


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    conf: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PocketSphinxCBackend:
    """
    Direct `ctypes` binding to `libpocketsphinx.so.3` and `libsphinxbase.so.3`.
    Provides real-time C-level speech recognition, frame-level word boundaries, and
    posterior word confidence probabilities without requiring pip compilation.
    """

    def __init__(
        self,
        hmm_dir: str = "/usr/share/pocketsphinx/model/en-us/en-us",
        lm_path: str = "/usr/share/pocketsphinx/model/en-us/en-us.lm.bin",
        dict_path: str = "/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict",
    ) -> None:
        self.hmm_dir = hmm_dir
        self.lm_path = lm_path
        self.dict_path = dict_path
        self._lock = threading.RLock()
        self.available = False
        self.status_message = "PocketSphinx uninitialized"
        self.sb: Any = None
        self.ps: Any = None
        self.config_ptr: Any = None
        self.decoder_ptr: Any = None
        self.lmath_ptr: Any = None
        self._init_c_library()

    def _init_c_library(self) -> None:
        sb_candidates = [
            "/usr/lib/x86_64-linux-gnu/libsphinxbase.so.3",
            "/usr/lib64/libsphinxbase.so.3",
            "/usr/lib/libsphinxbase.so.3",
        ]
        ps_candidates = [
            "/usr/lib/x86_64-linux-gnu/libpocketsphinx.so.3",
            "/usr/lib64/libpocketsphinx.so.3",
            "/usr/lib/libpocketsphinx.so.3",
        ]
        sb_path = next((p for p in sb_candidates if Path(p).exists()), None)
        ps_path = next((p for p in ps_candidates if Path(p).exists()), None)

        if not sb_path or not ps_path or not Path(self.hmm_dir).exists():
            self.available = False
            self.status_message = "libpocketsphinx.so.3 or en-us model not found on system"
            return

        try:
            self.sb = ctypes.CDLL(sb_path, mode=ctypes.RTLD_GLOBAL)
            self.ps = ctypes.CDLL(ps_path, mode=ctypes.RTLD_GLOBAL)

            self.ps.ps_args.argtypes = []
            self.ps.ps_args.restype = ctypes.c_void_p

            self.sb.cmd_ln_parse_r.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int32,
            ]
            self.sb.cmd_ln_parse_r.restype = ctypes.c_void_p

            self.ps.ps_init.argtypes = [ctypes.c_void_p]
            self.ps.ps_init.restype = ctypes.c_void_p
            self.ps.ps_free.argtypes = [ctypes.c_void_p]
            self.ps.ps_free.restype = ctypes.c_int

            self.ps.ps_start_utt.argtypes = [ctypes.c_void_p]
            self.ps.ps_start_utt.restype = ctypes.c_int
            self.ps.ps_process_raw.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.ps.ps_process_raw.restype = ctypes.c_int
            self.ps.ps_end_utt.argtypes = [ctypes.c_void_p]
            self.ps.ps_end_utt.restype = ctypes.c_int

            self.ps.ps_get_hyp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
            self.ps.ps_get_hyp.restype = ctypes.c_char_p

            self.ps.ps_seg_iter.argtypes = [ctypes.c_void_p]
            self.ps.ps_seg_iter.restype = ctypes.c_void_p
            self.ps.ps_seg_next.argtypes = [ctypes.c_void_p]
            self.ps.ps_seg_next.restype = ctypes.c_void_p
            self.ps.ps_seg_word.argtypes = [ctypes.c_void_p]
            self.ps.ps_seg_word.restype = ctypes.c_char_p
            self.ps.ps_seg_frames.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            self.ps.ps_seg_prob.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]
            self.ps.ps_seg_prob.restype = ctypes.c_int32

            self.ps.ps_get_logmath.argtypes = [ctypes.c_void_p]
            self.ps.ps_get_logmath.restype = ctypes.c_void_p
            self.sb.logmath_exp.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.sb.logmath_exp.restype = ctypes.c_double

            self.ps.ps_set_jsgf_string.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
            self.ps.ps_set_jsgf_string.restype = ctypes.c_int
            self.ps.ps_set_search.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.ps.ps_set_search.restype = ctypes.c_int
            self.ps.ps_unset_search.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.ps.ps_unset_search.restype = ctypes.c_int
            self.ps.ps_lookup_word.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.ps.ps_lookup_word.restype = ctypes.c_char_p

            argv_list = [
                b"pocketsphinx",
                b"-hmm",
                self.hmm_dir.encode("utf-8"),
                b"-lm",
                self.lm_path.encode("utf-8"),
                b"-dict",
                self.dict_path.encode("utf-8"),
                b"-frate",
                b"100",
                b"-logfn",
                b"/dev/null",
            ]
            ArgvType = ctypes.c_char_p * len(argv_list)
            self.config_ptr = self.sb.cmd_ln_parse_r(
                None, self.ps.ps_args(), len(argv_list), ArgvType(*argv_list), 1
            )
            if not self.config_ptr:
                self.status_message = "PocketSphinx cmd_ln_parse_r returned NULL"
                return

            self.decoder_ptr = self.ps.ps_init(self.config_ptr)
            if not self.decoder_ptr:
                self.status_message = "PocketSphinx ps_init returned NULL"
                return

            self.lmath_ptr = self.ps.ps_get_logmath(self.decoder_ptr)
            self.available = True
            self.status_message = "PocketSphinx EN-US (Native C PTM + 135k Dict) active"
        except Exception as exc:
            self.available = False
            self.status_message = f"PocketSphinx init error: {exc}"

    def _decode_chunk_16k(self, chunk_16k: np.ndarray, time_offset_s: float = 0.0) -> List[WordTimestamp]:
        if not self.available or len(chunk_16k) < 400:
            return []

        # Pad 80ms (1280 samples) of silence before and after so 1s_c_d_dd delta/double-delta
        # context windows never clip the first or last phoneme of an utterance
        pad_samples = 1280
        pad_s = pad_samples / 16000.0
        chunk_dur_s = len(chunk_16k) / 16000.0
        zeros = np.zeros(pad_samples, dtype=np.int16)
        padded = np.ascontiguousarray(np.concatenate([zeros, chunk_16k, zeros]), dtype=np.int16)

        out: List[WordTimestamp] = []

        with self._lock:
            self.ps.ps_start_utt(self.decoder_ptr)
            self.ps.ps_process_raw(
                self.decoder_ptr,
                padded.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                len(padded),
                0,
                1,
            )
            self.ps.ps_end_utt(self.decoder_ptr)

            seg = self.ps.ps_seg_iter(self.decoder_ptr)
            while seg:
                raw_w = self.ps.ps_seg_word(seg)
                if raw_w:
                    w_str = raw_w.decode("utf-8", errors="ignore")
                    if not w_str.startswith("<") and not w_str.startswith("["):
                        clean_w = re.sub(r"\(\d+\)$", "", w_str).strip().lower()
                        sf = ctypes.c_int()
                        ef = ctypes.c_int()
                        self.ps.ps_seg_frames(seg, ctypes.byref(sf), ctypes.byref(ef))
                        ascr = ctypes.c_int32()
                        lscr = ctypes.c_int32()
                        lback = ctypes.c_int32()
                        lp = self.ps.ps_seg_prob(
                            seg, ctypes.byref(ascr), ctypes.byref(lscr), ctypes.byref(lback)
                        )
                        prob = float(self.sb.logmath_exp(self.lmath_ptr, lp))
                        prob = float(np.clip(prob, 0.01, 1.0))

                        rel_start = max(0.0, sf.value * 0.01 - pad_s)
                        rel_end = min(chunk_dur_s, max(rel_start + 0.03, (ef.value + 1) * 0.01 - pad_s))
                        start_s = time_offset_s + rel_start
                        end_s = time_offset_s + rel_end
                        if clean_w:
                            out.append(
                                WordTimestamp(
                                    word=clean_w,
                                    start=round(start_s, 4),
                                    end=round(end_s, 4),
                                    conf=round(prob, 4),
                                )
                            )
                seg = self.ps.ps_seg_next(seg)
        return out

    def transcribe_pcm16(
        self,
        pcm16: np.ndarray,
        sample_rate: int = 16000,
        split_utterances: bool = True,
    ) -> List[WordTimestamp]:
        if not self.available or len(pcm16) == 0:
            return []

        if sample_rate != 16000:
            samples_16k = resample_audio(pcm16, sample_rate, 16000)
        else:
            samples_16k = np.asarray(pcm16, dtype=np.int16)

        if split_utterances and len(samples_16k) > 16000 * 5:
            spans = split_stream_into_utterances(samples_16k, sample_rate=16000)
            all_words: List[WordTimestamp] = []
            for s0, s1 in spans:
                offset_s = s0 / 16000.0
                all_words.extend(self._decode_chunk_16k(samples_16k[s0:s1], time_offset_s=offset_s))
            return all_words

        return self._decode_chunk_16k(samples_16k, time_offset_s=0.0)

    def align_known_phrase_pcm16(
        self,
        pcm16: np.ndarray,
        known_text: str,
        sample_rate: int = 16000,
    ) -> List[WordTimestamp]:
        if not self.available or len(pcm16) == 0:
            return []

        if sample_rate != 16000:
            samples_16k = resample_audio(pcm16, sample_rate, 16000)
        else:
            samples_16k = np.asarray(pcm16, dtype=np.int16)

        tokens = [
            w for w in re.sub(r"[^a-z0-9\s']", " ", known_text.lower()).split() if w
        ]
        if not tokens:
            return self.transcribe_pcm16(samples_16k, sample_rate=16000, split_utterances=False)

        with self._lock:
            in_dict = []
            for t in tokens:
                looked = self.ps.ps_lookup_word(self.decoder_ptr, t.encode("utf-8"))
                if looked:
                    in_dict.append(t)

            if len(in_dict) != len(tokens):
                return self.transcribe_pcm16(samples_16k, sample_rate=16000, split_utterances=False)

            seq_rule = " ".join(in_dict)
            jsgf = f"#JSGF V1.0;\ngrammar align;\npublic <utt> = {seq_rule};\n"
            rc = self.ps.ps_set_jsgf_string(
                self.decoder_ptr, b"align_search", jsgf.encode("utf-8")
            )
            if rc != 0:
                return self.transcribe_pcm16(samples_16k, sample_rate=16000, split_utterances=False)

            try:
                self.ps.ps_set_search(self.decoder_ptr, b"align_search")
                words = self._decode_chunk_16k(samples_16k, time_offset_s=0.0)
                return words
            finally:
                self.ps.ps_set_search(self.decoder_ptr, b"_default")
                self.ps.ps_unset_search(self.decoder_ptr, b"align_search")


class MultiBackendSTTExtractor:
    """
    Unified Speech-to-Text & Word Timestamp Extractor supporting:
    - `pocketsphinx-en-us` (Native C `libpocketsphinx.so.3`)
    - `vosk-small-en` / `vosk-lgraph-en` (`vosk`)
    - `whisper-tiny-en-onnx` / `whisper-base-en-onnx` (`faster-whisper` / ONNX)
    """

    def __init__(self, config: Optional[VoxSpliceConfig] = None) -> None:
        self.config = config or VoxSpliceConfig.load()
        self.hub = ModelHubManager(self.config)
        self.active_model_id: str = self.config.stt_model_id
        self.backend_type: str = "none"
        self.pocketsphinx_backend: Optional[PocketSphinxCBackend] = None
        self.vosk_model: Any = None
        self.vosk_recognizer: Any = None
        self.whisper_model: Any = None
        self.available: bool = False
        self.status_message: str = "Initializing STT..."
        self._load_best_backend()

    def _resolve_vosk_dir(self, model_id: str) -> Optional[Path]:
        models_dir = self.config.resolve_models_dir()
        candidates = [
            models_dir / model_id,
            models_dir / "vosk-model-small-en-us-0.15",
            models_dir / "vosk-model-en-us-0.22-lgraph",
            models_dir / "default-en",
            self.config.resolve_asset_path(self.config.vosk_model_path),
            Path.home() / ".local" / "share" / "5mic" / "models" / "vosk-model-small-en-us-0.15",
            Path.home() / ".local" / "lib" / "5mic" / "models" / "default-en",
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                if (c / "am").exists() or (c / "conf").exists():
                    return c
                for sub in c.iterdir():
                    if sub.is_dir() and (sub / "am").exists():
                        return sub
        return None

    def _load_best_backend(self) -> None:
        requested = self.config.stt_model_id

        if requested in ("whisper-tiny-en-onnx", "whisper-base-en-onnx", "whisper-tiny-en", "whisper-base-en"):
            try:
                from faster_whisper import WhisperModel  # type: ignore

                hf_id = "base.en" if "base" in requested else "tiny.en"
                self.whisper_model = WhisperModel(
                    hf_id,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.config.resolve_models_dir()),
                )
                self.backend_type = "faster_whisper"
                self.active_model_id = requested
                self.available = True
                self.status_message = f"Faster-Whisper ({hf_id} INT8 CPU) active"
                return
            except Exception:
                pass

        if requested in ("auto", "vosk-small-en", "vosk-lgraph-en"):
            try:
                import vosk  # type: ignore

                vosk.SetLogLevel(-1)
                v_dir = self._resolve_vosk_dir(requested)
                if v_dir is not None:
                    self.vosk_model = vosk.Model(str(v_dir))
                    self.vosk_recognizer = vosk.KaldiRecognizer(
                        self.vosk_model, float(self.config.capture_sample_rate)
                    )
                    self.vosk_recognizer.SetWords(True)
                    self.backend_type = "vosk"
                    self.active_model_id = requested if requested != "auto" else "vosk-small-en"
                    self.available = True
                    self.status_message = f"Vosk ({v_dir.name}) active"
                    return
            except Exception:
                pass

        ps_backend = PocketSphinxCBackend()
        if ps_backend.available:
            self.pocketsphinx_backend = ps_backend
            self.backend_type = "pocketsphinx"
            self.active_model_id = "pocketsphinx-en-us"
            self.available = True
            self.status_message = ps_backend.status_message
            return

        try:
            import vosk  # type: ignore

            vosk.SetLogLevel(-1)
            self.vosk_model = vosk.Model(lang="en-us")
            self.vosk_recognizer = vosk.KaldiRecognizer(
                self.vosk_model, float(self.config.capture_sample_rate)
            )
            self.vosk_recognizer.SetWords(True)
            self.backend_type = "vosk"
            self.active_model_id = "vosk-small-en"
            self.available = True
            self.status_message = "Vosk (en-us) active"
            return
        except Exception as exc:
            self.available = False
            self.status_message = f"No offline STT backend loaded ({exc})"

    def download_and_switch_model(
        self,
        model_id: str,
        variant_id: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str]:
        ok, msg, _path = self.hub.download_model_variant(
            model_id,
            variant_id=variant_id,
            progress_cb=(lambda s, _p: progress_cb(s)) if progress_cb else None,
        )
        if not ok:
            return False, msg
        self.config.stt_model_id = model_id
        self._load_best_backend()
        if self.available:
            self.config.save()
            return True, f"Activated {model_id} ({self.status_message})"
        return False, f"Model files downloaded, but runtime initialization failed: {self.status_message}"

    def process_utterance_pcm16(
        self,
        pcm16: np.ndarray,
        sample_rate: Optional[int] = None,
        known_text_hint: Optional[str] = None,
    ) -> List[WordTimestamp]:
        if not self.available or len(pcm16) == 0:
            return []

        sr = sample_rate or self.config.capture_sample_rate
        f32 = int16_to_float32(pcm16)

        if self.config.enable_vad_gate and not is_human_speech_buffer(
            f32,
            sample_rate=sr,
            min_rms=self.config.min_word_rms,
            max_flatness=self.config.max_spectral_flatness + 0.08,
        ):
            return []

        if self.backend_type == "pocketsphinx" and self.pocketsphinx_backend is not None:
            if known_text_hint:
                return self.pocketsphinx_backend.align_known_phrase_pcm16(
                    pcm16, known_text_hint, sample_rate=sr
                )
            return self.pocketsphinx_backend.transcribe_pcm16(pcm16, sample_rate=sr)

        if self.backend_type == "faster_whisper" and self.whisper_model is not None:
            try:
                f32_16k = resample_audio(f32, sr, 16000) if sr != 16000 else f32
                segments, _info = self.whisper_model.transcribe(
                    f32_16k,
                    language="en",
                    word_timestamps=True,
                    vad_filter=True,
                    beam_size=1,
                )
                out_w: List[WordTimestamp] = []
                for seg in segments:
                    for w_obj in getattr(seg, "words", []) or []:
                        raw_w = str(w_obj.word).strip().lower().strip(".,!?;:\"'")
                        if not raw_w:
                            continue
                        out_w.append(
                            WordTimestamp(
                                word=raw_w,
                                start=round(max(0.0, float(w_obj.start)), 4),
                                end=round(max(float(w_obj.start) + 0.04, float(w_obj.end)), 4),
                                conf=round(float(getattr(w_obj, "probability", 0.9)), 4),
                            )
                        )
                return out_w
            except Exception:
                return []

        if self.backend_type == "vosk" and self.vosk_recognizer is not None:
            pcm_16k = resample_audio(pcm16, sr, self.config.capture_sample_rate) if sr != self.config.capture_sample_rate else pcm16
            raw_bytes = np.asarray(pcm_16k, dtype=np.int16).tobytes()
            self.vosk_recognizer.Reset()
            self.vosk_recognizer.AcceptWaveform(raw_bytes)
            res_json = self.vosk_recognizer.FinalResult()
            try:
                data = json.loads(res_json)
            except Exception:
                return []
            out: List[WordTimestamp] = []
            for item in data.get("result", []):
                w = str(item.get("word", "")).strip().lower()
                if not w:
                    continue
                out.append(
                    WordTimestamp(
                        word=w,
                        start=round(float(item.get("start", 0.0)), 4),
                        end=round(float(item.get("end", 0.0)), 4),
                        conf=round(float(item.get("conf", 1.0)), 4),
                    )
                )
            return out

        return []
