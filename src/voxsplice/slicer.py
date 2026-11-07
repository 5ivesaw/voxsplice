"""
slicer.py — Acoustic Boundary & Coarticulation Inspector, Chain-Lock Phrase Slicer,
and Multi-Granularity Word/Sub-Phrase Extractor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import VoxSpliceConfig
from .dsp import (
    apply_micro_edge_fade,
    apply_ulaw_reconstruction_filter,
    compute_rms,
    compute_spectral_flatness,
    compute_zcr,
    decode_ulaw,
    encode_ulaw,
    estimate_pitch_and_centroid,
    float32_to_int16,
    int16_to_float32,
    resample_audio,
    snap_to_zero_crossing,
    vocal_bandpass_filter,
)
from .phonetics import PhoneticLexicon
from .stt_backends import WordTimestamp

_SHORT_FUNCTION_WORDS: Set[str] = {
    "a", "i", "u", "to", "in", "it", "is", "be", "on", "at", "or", "of", "by", "as", "an",
    "so", "if", "we", "he", "me", "my", "no", "up", "go", "do", "us", "am", "the", "and",
    "for", "are", "but", "not", "you", "all", "can", "was", "one", "out", "who", "get",
    "has", "him", "his", "how", "its", "let", "now", "see", "two", "way", "did", "say",
    "she", "our", "out", "there", "where", "here", "under", "over", "after", "until",
    "this", "that", "from", "with", "have", "will", "what", "when", "your", "they", "them",
    "i'm", "im", "i'll", "you'd",
}


@dataclass
class SeamAnalysis:
    left_word: str
    right_word: str
    seam_time_s: float
    silence_gap_ms: float
    seam_rms: float
    words_rms: float
    seam_rms_ratio: float
    zcr_shift: float
    is_bleeding: bool
    optimal_cut_sample: int
    coarticulation_note: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlicedClip:
    text: str
    word_count: int
    start_s: float
    end_s: float
    encoded_bytes: bytes
    audio_format: str
    sample_rate: int
    pitch_hz: float
    rms_energy: float
    spectral_centroid: float
    spectral_flatness: float
    seam_rms_ratio: float
    is_locked_phrase: bool
    priority: int = 1
    speaker_tag: str = "lobby"
    confidence: float = 1.0
    phonemes: str = ""
    word_offsets_json: str = "[]"


class CoarticulationSlicer:
    def __init__(self, config: Optional[VoxSpliceConfig] = None) -> None:
        self.config = config or VoxSpliceConfig.load()
        self.lexicon = PhoneticLexicon.get_default()

    def _describe_phoneme_transition(self, w_left: str, w_right: str, is_bleeding: bool) -> str:
        p_left = self.lexicon.get_unstressed_phones(w_left)
        p_right = self.lexicon.get_unstressed_phones(w_right)
        tail_p = f"/{p_left[-1]}/" if p_left else "/?/"
        head_p = f"/{p_right[0]}/" if p_right else "/?/"
        if is_bleeding:
            return f"Continuous voicing across {tail_p} -> {head_p} (coarticulated lock)"
        return f"Clean acoustic boundary between {tail_p} -> {head_p} (safe cut)"

    def analyze_seam(
        self,
        samples_f32: np.ndarray,
        sample_rate: int,
        w_left: WordTimestamp,
        w_right: WordTimestamp,
    ) -> SeamAnalysis:
        n = len(samples_f32)
        gap_s = w_right.start - w_left.end
        gap_ms = gap_s * 1000.0
        seam_time_s = 0.5 * (w_left.end + w_right.start)
        seam_center_idx = int(np.clip(round(seam_time_s * sample_rate), 0, max(0, n - 1)))

        l_start = int(np.clip(round(w_left.start * sample_rate), 0, n))
        l_end = int(np.clip(round(w_left.end * sample_rate), l_start + 1, n))
        r_start = int(np.clip(round(w_right.start * sample_rate), 0, n))
        r_end = int(np.clip(round(w_right.end * sample_rate), r_start + 1, n))

        left_seg = samples_f32[l_start:l_end]
        right_seg = samples_f32[r_start:r_end]
        rms_left = compute_rms(left_seg)
        rms_right = compute_rms(right_seg)
        words_rms = max(1e-6, 0.5 * (rms_left + rms_right))

        zcr_left = compute_zcr(left_seg[-min(len(left_seg), int(sample_rate * 0.03)) :])
        zcr_right = compute_zcr(right_seg[: min(len(right_seg), int(sample_rate * 0.03))])
        zcr_shift = abs(zcr_right - zcr_left)

        half_win = max(1, int(round((self.config.seam_window_ms * 0.5 / 1000.0) * sample_rate)))
        win_start = max(0, seam_center_idx - half_win)
        win_end = min(n, seam_center_idx + half_win)
        seam_slice = samples_f32[win_start:win_end]
        seam_rms = compute_rms(seam_slice)
        seam_rms_ratio = float(seam_rms / words_rms)

        if len(seam_slice) >= 8:
            micro = max(2, int(sample_rate * 0.002))
            sq = np.convolve(np.square(seam_slice), np.ones(micro) / micro, mode="valid")
            min_offset = int(np.argmin(sq)) + micro // 2
            raw_cut = min(n - 1, win_start + min_offset)
            optimal_cut_sample = snap_to_zero_crossing(samples_f32, raw_cut, radius=64)
        else:
            optimal_cut_sample = snap_to_zero_crossing(samples_f32, seam_center_idx, radius=64)

        if gap_ms > self.config.silence_gap_ms:
            note = self._describe_phoneme_transition(w_left.word, w_right.word, False)
            return SeamAnalysis(
                left_word=w_left.word,
                right_word=w_right.word,
                seam_time_s=round(seam_time_s, 4),
                silence_gap_ms=round(gap_ms, 2),
                seam_rms=round(seam_rms, 5),
                words_rms=round(words_rms, 5),
                seam_rms_ratio=round(seam_rms_ratio, 4),
                zcr_shift=round(zcr_shift, 4),
                is_bleeding=False,
                optimal_cut_sample=optimal_cut_sample,
                coarticulation_note=note,
                reason=f"CLEAN • Silence Gap ({gap_ms:.1f}ms > {self.config.silence_gap_ms:.0f}ms)",
            )

        if seam_rms_ratio < self.config.seam_rms_threshold:
            note = self._describe_phoneme_transition(w_left.word, w_right.word, False)
            return SeamAnalysis(
                left_word=w_left.word,
                right_word=w_right.word,
                seam_time_s=round(seam_time_s, 4),
                silence_gap_ms=round(gap_ms, 2),
                seam_rms=round(seam_rms, 5),
                words_rms=round(words_rms, 5),
                seam_rms_ratio=round(seam_rms_ratio, 4),
                zcr_shift=round(zcr_shift, 4),
                is_bleeding=False,
                optimal_cut_sample=optimal_cut_sample,
                coarticulation_note=note,
                reason=f"CLEAN • Acoustic Dip ({seam_rms_ratio:.2f} < {self.config.seam_rms_threshold:.2f})",
            )

        note = self._describe_phoneme_transition(w_left.word, w_right.word, True)
        return SeamAnalysis(
            left_word=w_left.word,
            right_word=w_right.word,
            seam_time_s=round(seam_time_s, 4),
            silence_gap_ms=round(gap_ms, 2),
            seam_rms=round(seam_rms, 5),
            words_rms=round(words_rms, 5),
            seam_rms_ratio=round(seam_rms_ratio, 4),
            zcr_shift=round(zcr_shift, 4),
            is_bleeding=True,
            optimal_cut_sample=optimal_cut_sample,
            coarticulation_note=note,
            reason=f"BLEEDING • Continuous Voicing ({seam_rms_ratio:.2f} >= {self.config.seam_rms_threshold:.2f})",
        )

    def encode_clip_samples(self, samples_f32: np.ndarray, orig_sr: int) -> bytes:
        target_sr = self.config.arena_sample_rate
        resampled = resample_audio(samples_f32, orig_sr, target_sr)
        pcm16 = float32_to_int16(resampled)
        if self.config.arena_encoding == "ulaw":
            return encode_ulaw(pcm16)
        return pcm16.tobytes()

    def decode_clip_bytes(self, raw_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
        if self.config.arena_encoding == "ulaw":
            pcm16 = decode_ulaw(raw_bytes)
            f32 = int16_to_float32(pcm16)
            out = resample_audio(f32, self.config.arena_sample_rate, target_sr)
            return apply_ulaw_reconstruction_filter(out, sample_rate=target_sr)
        pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
        f32 = int16_to_float32(pcm16)
        return resample_audio(f32, self.config.arena_sample_rate, target_sr)

    def _build_clip(
        self,
        words: Sequence[WordTimestamp],
        samples_f32: np.ndarray,
        sample_rate: int,
        cut_start_sample: int,
        cut_end_sample: int,
        seam_rms_ratio: float,
        is_locked_phrase: bool,
        speaker_tag: str = "lobby",
        word_sample_spans: Optional[Sequence[Tuple[str, int, int]]] = None,
    ) -> Optional[SlicedClip]:
        if cut_end_sample - cut_start_sample < int(sample_rate * 0.032):
            return None

        s_idx = snap_to_zero_crossing(samples_f32, cut_start_sample, radius=64)
        e_idx = snap_to_zero_crossing(samples_f32, cut_end_sample, radius=64)
        if e_idx - s_idx < int(sample_rate * 0.032):
            return None

        raw_slice = samples_f32[s_idx:e_idx]
        flatness = compute_spectral_flatness(raw_slice, sample_rate=sample_rate)
        if flatness > self.config.max_spectral_flatness:
            return None

        if self.config.enable_bandpass_filter:
            proc_slice = vocal_bandpass_filter(
                raw_slice,
                sample_rate=sample_rate,
                low_hz=85.0,
                high_hz=5600.0,
            )
        else:
            proc_slice = raw_slice

        proc_slice = apply_micro_edge_fade(proc_slice, sample_rate=sample_rate, fade_ms=2.0)
        rms = compute_rms(proc_slice)
        if rms < self.config.min_word_rms:
            return None

        pitch_hz, centroid_hz = estimate_pitch_and_centroid(proc_slice, sample_rate=sample_rate)
        encoded = self.encode_clip_samples(proc_slice, orig_sr=sample_rate)
        text = " ".join(w.word.strip().lower() for w in words if w.word.strip())
        if not text:
            return None

        avg_conf = float(np.mean([w.conf for w in words])) if words else 1.0
        phones_str = " ".join(self.lexicon.phrase_phones(text))

        offsets_list = []
        total_samples = max(1, e_idx - s_idx)
        if word_sample_spans:
            for w_txt, ws, we in word_sample_spans:
                r0 = max(0.0, min(1.0, (ws - s_idx) / float(total_samples)))
                r1 = max(r0, min(1.0, (we - s_idx) / float(total_samples)))
                offsets_list.append({"w": w_txt, "r0": round(r0, 4), "r1": round(r1, 4)})

        return SlicedClip(
            text=text,
            word_count=len(words),
            start_s=round(s_idx / float(sample_rate), 4),
            end_s=round(e_idx / float(sample_rate), 4),
            encoded_bytes=encoded,
            audio_format=self.config.audio_format,
            sample_rate=self.config.arena_sample_rate,
            pitch_hz=pitch_hz,
            rms_energy=round(rms, 5),
            spectral_centroid=centroid_hz,
            spectral_flatness=round(flatness, 4),
            seam_rms_ratio=round(seam_rms_ratio, 4),
            is_locked_phrase=is_locked_phrase,
            priority=1,
            speaker_tag=speaker_tag,
            confidence=round(avg_conf, 3),
            phonemes=phones_str,
            word_offsets_json=json.dumps(offsets_list),
        )

    def slice_utterance(
        self,
        pcm_input: np.ndarray,
        words: Sequence[WordTimestamp],
        sample_rate: int = 16000,
        speaker_tag: str = "lobby",
    ) -> Tuple[List[SlicedClip], List[SeamAnalysis]]:
        if len(words) == 0 or len(pcm_input) == 0:
            return [], []

        if pcm_input.dtype == np.int16:
            samples_f32 = int16_to_float32(pcm_input)
        else:
            samples_f32 = np.asarray(pcm_input, dtype=np.float32)

        min_conf = float(self.config.min_word_confidence)
        short_conf = max(0.46, min_conf - 0.16)

        confident_runs: List[List[WordTimestamp]] = []
        current_run: List[WordTimestamp] = []

        for w in words:
            clean_w = w.word.strip().lower()
            if not clean_w or clean_w in ("[unk]", "<unk>"):
                if current_run:
                    confident_runs.append(current_run)
                    current_run = []
                continue
            req_conf = short_conf if (clean_w in _SHORT_FUNCTION_WORDS and min_conf <= 0.70) else min_conf
            if float(w.conf) < req_conf:
                if current_run:
                    confident_runs.append(current_run)
                    current_run = []
                continue
            current_run.append(w)

        if current_run:
            confident_runs.append(current_run)

        all_clips: List[SlicedClip] = []
        all_seams: List[SeamAnalysis] = []

        for run in confident_runs:
            run_clips, run_seams = self._slice_confident_run(
                samples_f32, run, sample_rate=sample_rate, speaker_tag=speaker_tag
            )
            all_clips.extend(run_clips)
            all_seams.extend(run_seams)

        return all_clips, all_seams

    def _slice_confident_run(
        self,
        samples_f32: np.ndarray,
        words: Sequence[WordTimestamp],
        sample_rate: int,
        speaker_tag: str,
    ) -> Tuple[List[SlicedClip], List[SeamAnalysis]]:
        n_samples = len(samples_f32)
        n_words = len(words)
        if n_words == 0:
            return [], []

        seams: List[SeamAnalysis] = []
        for i in range(n_words - 1):
            seams.append(self.analyze_seam(samples_f32, sample_rate, words[i], words[i + 1]))

        pre_margin = int(round(0.015 * sample_rate))
        post_margin = int(round(0.025 * sample_rate))

        word_start_samples: List[int] = []
        word_end_samples: List[int] = []
        for i, w in enumerate(words):
            if i == 0 or seams[i - 1].silence_gap_ms > self.config.silence_gap_ms:
                s_idx = int(np.clip(round(w.start * sample_rate) - pre_margin, 0, n_samples))
            else:
                s_idx = seams[i - 1].optimal_cut_sample

            if i == n_words - 1 or seams[i].silence_gap_ms > self.config.silence_gap_ms:
                e_idx = int(np.clip(round(w.end * sample_rate) + post_margin, s_idx + 1, n_samples))
            else:
                e_idx = max(s_idx + 1, seams[i].optimal_cut_sample)

            word_start_samples.append(s_idx)
            word_end_samples.append(e_idx)

        clips: List[SlicedClip] = []
        emitted_spans = set()
        emitted_texts = set()
        i = 0
        max_chain = max(2, int(self.config.max_bleed_chain_words))

        # 1. Primary Chain-Lock Pass
        while i < n_words:
            j = i
            while j < n_words - 1 and seams[j].is_bleeding and (j - i + 1) < max_chain:
                j += 1

            if j > i and j < n_words - 1 and seams[j].is_bleeding:
                best_k = min(range(i, j + 1), key=lambda k: seams[k].seam_rms_ratio)
                j = best_k

            chain_words = words[i : j + 1]
            c_start = word_start_samples[i]
            c_end = word_end_samples[j]

            left_ratio = seams[i - 1].seam_rms_ratio if i > 0 else 0.05
            right_ratio = seams[j].seam_rms_ratio if j < n_words - 1 else 0.05
            edge_ratio = 0.5 * (left_ratio + right_ratio)

            spans_meta = [
                (words[k].word.strip().lower(), word_start_samples[k], word_end_samples[k])
                for k in range(i, j + 1)
            ]

            is_locked = len(chain_words) > 1
            clip = self._build_clip(
                words=chain_words,
                samples_f32=samples_f32,
                sample_rate=sample_rate,
                cut_start_sample=c_start,
                cut_end_sample=c_end,
                seam_rms_ratio=edge_ratio,
                is_locked_phrase=is_locked,
                speaker_tag=speaker_tag,
                word_sample_spans=spans_meta,
            )
            if clip is not None:
                clips.append(clip)
                emitted_spans.add((i, j))
                emitted_texts.add(clip.text)

            i = j + 1

        # 2. Sliding Connected Phrase Pass (2-word and 3-word phrases across contiguous connected words)
        for span in (2, 3):
            for s_i in range(n_words - span + 1):
                s_j = s_i + span - 1
                if (s_i, s_j) in emitted_spans:
                    continue
                if all(seams[k].silence_gap_ms < 90.0 for k in range(s_i, s_j)):
                    sub_words = words[s_i : s_j + 1]
                    txt = " ".join(w.word.strip().lower() for w in sub_words if w.word.strip())
                    if txt in emitted_texts:
                        continue
                    left_r = seams[s_i - 1].seam_rms_ratio if s_i > 0 else 0.05
                    right_r = seams[s_j].seam_rms_ratio if s_j < n_words - 1 else 0.05
                    spans_meta = [
                        (words[k].word.strip().lower(), word_start_samples[k], word_end_samples[k])
                        for k in range(s_i, s_j + 1)
                    ]
                    sub_clip = self._build_clip(
                        words=sub_words,
                        samples_f32=samples_f32,
                        sample_rate=sample_rate,
                        cut_start_sample=word_start_samples[s_i],
                        cut_end_sample=word_end_samples[s_j],
                        seam_rms_ratio=0.5 * (left_r + right_r),
                        is_locked_phrase=True,
                        speaker_tag=speaker_tag,
                        word_sample_spans=spans_meta,
                    )
                    if sub_clip is not None:
                        clips.append(sub_clip)
                        emitted_spans.add((s_i, s_j))
                        emitted_texts.add(txt)

        # 3. Single-Word Vocabulary Guarantee Pass (`always_extract_single_words = True`)
        if self.config.always_extract_single_words:
            for idx in range(n_words):
                if (idx, idx) in emitted_spans:
                    continue
                w = words[idx]
                if (w.end - w.start) < 0.032:
                    continue
                left_r = seams[idx - 1].seam_rms_ratio if idx > 0 else 0.05
                right_r = seams[idx].seam_rms_ratio if idx < n_words - 1 else 0.05
                single_clip = self._build_clip(
                    words=[w],
                    samples_f32=samples_f32,
                    sample_rate=sample_rate,
                    cut_start_sample=word_start_samples[idx],
                    cut_end_sample=word_end_samples[idx],
                    seam_rms_ratio=0.5 * (left_r + right_r),
                    is_locked_phrase=False,
                    speaker_tag=speaker_tag,
                    word_sample_spans=[(w.word.strip().lower(), word_start_samples[idx], word_end_samples[idx])],
                )
                if single_clip is not None:
                    clips.append(single_clip)
                    emitted_spans.add((idx, idx))

        return clips, seams
