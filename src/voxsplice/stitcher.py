"""
stitcher.py — Viterbi Dynamic Programming Sentence Cover Planner, 5-Tier Hybrid Matcher
(Exact, Surgical Sub-Phrase, Sub-Word Syllable Splicing, Compositional Semantic/Phonetic, Starter Fallback),
and Phase-Aligned Equal-Power Hann Crossfader.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .arena_db import ArenaDatabase, ClipRecord
from .config import VoxSpliceConfig
from .dsp import compute_rms, float32_to_int16
from .fx_rack import process_chunk_fx
from .phonetics import PhoneticLexicon, SyllableSliceSpec

_SYNONYM_CLUSTERS: List[Tuple[str, ...]] = [
    ("hello", "hey", "hi", "yo", "sup", "greetings", "welcome"),
    ("human", "person", "guy", "bro", "man", "dude", "player", "user", "somebody", "someone", "people", "everybody"),
    ("real", "actual", "true", "genuine", "legit", "alive", "normal", "good", "really"),
    ("behind", "back", "near", "close", "around", "beside", "outside", "inside", "under"),
    ("distant", "remote", "far", "away", "external", "foreign"),
    ("look", "see", "watch", "peep", "check", "stare", "view", "find"),
    ("run", "running", "go", "going", "move", "walk", "stay", "standing", "rush", "escape", "flee", "leave", "left", "enter", "try", "trying"),
    ("work", "function", "operate", "working", "perform", "act"),
    ("stop", "wait", "hold", "freeze", "halt", "chill", "pause", "stay", "avoid"),
    ("help", "save", "assist", "rescue", "please", "need", "want", "require", "must"),
    ("yes", "yeah", "yep", "yup", "sure", "okay", "ok", "good"),
    ("no", "nah", "nope", "never", "not", "dont", "cant", "wont", "avoid", "without"),
    ("what", "huh", "why", "how", "where", "who", "when", "which", "that"),
    ("scared", "afraid", "terrified", "creepy", "weird", "strange", "bad", "dark"),
    ("monster", "mimic", "skinwalker", "ghost", "demon", "creature", "entity", "thing", "it"),
    ("friend", "friends", "buddy", "pal", "teammate", "mate", "homie", "brother", "team", "guys"),
    ("drop", "give", "throw", "put", "hand", "pass", "take", "bring", "get", "open"),
    ("weapon", "sword", "gun", "knife", "item", "loot", "flashlight", "gear", "option", "lights"),
    ("dead", "died", "killed", "gone", "lost", "down", "alone"),
    ("hear", "heard", "listen", "sound", "noise", "voice", "talking", "speaking", "saying", "call", "dial"),
    ("me", "my", "mine", "myself", "i", "im", "i'm", "i'll"),
    ("you", "your", "yours", "yourself", "u", "ur", "you'd"),
    ("is", "am", "are", "was", "were", "be", "being", "been", "its", "have", "will", "did"),
    ("the", "this", "that", "these", "those", "a", "an", "all", "our"),
    ("place", "location", "position", "spot", "site", "area", "zone", "room", "building", "here", "there"),
    ("information", "info", "data", "details", "knowledge", "facts", "story"),
    ("technical", "tech", "system", "digital", "configuration", "setup", "demonstration"),
    ("connect", "connected", "connection", "link", "linked", "joined", "online"),
    ("internet", "web", "network", "net", "server"),
    ("learn", "know", "understand", "discover", "study", "read", "share"),
]


class StaticEmbeddingBrain:
    def __init__(self, config: Optional[VoxSpliceConfig] = None) -> None:
        self.config = config or VoxSpliceConfig.load()
        self.dim: int = 256
        self.lexicon = PhoneticLexicon.get_default()
        self.model2vec_model: Any = None
        self.using_model2vec: bool = False
        self._cache: Dict[str, np.ndarray] = {}
        self._synonym_basis: Dict[str, np.ndarray] = {}
        self._synonym_lookup: Dict[str, Set[str]] = {}
        self._init_synonym_basis()
        self._try_load_model2vec()

    def _init_synonym_basis(self) -> None:
        rng = np.random.default_rng(seed=5005)
        for cluster in _SYNONYM_CLUSTERS:
            cluster_set = set(cluster)
            cluster_vec = rng.standard_normal(self.dim).astype(np.float32)
            cluster_vec /= max(1e-8, float(np.linalg.norm(cluster_vec)))
            for word in cluster:
                self._synonym_lookup.setdefault(word, set()).update(cluster_set)
                word_jitter = rng.standard_normal(self.dim).astype(np.float32) * 0.10
                v = cluster_vec * 0.92 + word_jitter
                v /= max(1e-8, float(np.linalg.norm(v)))
                if word in self._synonym_basis:
                    merged = self._synonym_basis[word] + v
                    merged /= max(1e-8, float(np.linalg.norm(merged)))
                    self._synonym_basis[word] = merged
                else:
                    self._synonym_basis[word] = v

    def _try_load_model2vec(self) -> None:
        if os.environ.get("VOXSPLICE_OFFLINE_EMBEDDINGS") == "1" or os.environ.get("FIVEMIC_OFFLINE_EMBEDDINGS") == "1":
            self.using_model2vec = False
            return
        try:
            from model2vec import StaticModel  # type: ignore

            local_path = self.config.resolve_models_dir() / self.config.embedding_model_id
            if local_path.exists():
                self.model2vec_model = StaticModel.from_pretrained(str(local_path))
                self.using_model2vec = True
        except Exception:
            self.using_model2vec = False

    def _fast_hybrid_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = [w for w in re.sub(r"[^a-z0-9\s']", "", text.lower()).split() if w]
        if not words:
            return vec

        for w in words:
            w_vec = np.zeros(self.dim, dtype=np.float32)
            stem = self.lexicon.morphological_stem(w)

            if w in self._synonym_basis:
                w_vec += 0.72 * self._synonym_basis[w]
            elif stem in self._synonym_basis:
                w_vec += 0.65 * self._synonym_basis[stem]

            h_stem = zlib.crc32(f"stem:{stem}".encode("utf-8"))
            w_vec[h_stem % self.dim] += 0.25
            w_vec[(h_stem >> 8) % self.dim] -= 0.25

            phones = self.lexicon.get_unstressed_phones(w)
            if phones:
                p_str = " ".join(phones)
                h_phon = zlib.crc32(f"arpabet:{p_str}".encode("utf-8"))
                for shift in (0, 5, 11, 17):
                    p_idx = (h_phon >> shift) % self.dim
                    sign = 1.0 if ((h_phon >> (shift + 2)) & 1) == 0 else -1.0
                    w_vec[p_idx] += 0.16 * sign

            norm = float(np.linalg.norm(w_vec))
            if norm > 1e-8:
                w_vec /= norm
            vec += w_vec

        total_norm = float(np.linalg.norm(vec))
        if total_norm > 1e-8:
            vec /= total_norm
        return vec

    def encode(self, text: str) -> np.ndarray:
        norm_text = " ".join(text.strip().lower().split())
        if norm_text in self._cache:
            return self._cache[norm_text]

        if self.using_model2vec and self.model2vec_model is not None:
            try:
                raw_v = self.model2vec_model.encode([norm_text])[0].astype(np.float32)
                n = float(np.linalg.norm(raw_v))
                if n > 1e-8:
                    raw_v /= n
                self._cache[norm_text] = raw_v
                return raw_v
            except Exception:
                pass

        vec = self._fast_hybrid_vector(norm_text)
        if len(self._cache) < 25000:
            self._cache[norm_text] = vec
        return vec

    def are_synonyms_or_inflections(self, w1: str, w2: str) -> Tuple[bool, float]:
        a = w1.strip().lower().replace("'", "")
        b = w2.strip().lower().replace("'", "")
        if a == b:
            return True, 1.0
        stem_a = self.lexicon.morphological_stem(a)
        stem_b = self.lexicon.morphological_stem(b)
        if stem_a and stem_a == stem_b and len(stem_a) >= 3:
            return True, 0.94
        if a in self._synonym_lookup and b in self._synonym_lookup[a]:
            return True, 0.86
        if stem_a in self._synonym_lookup and stem_b in self._synonym_lookup[stem_a]:
            return True, 0.82
        return False, 0.0

    def similarity(self, text_a: str, text_b: str) -> float:
        a = text_a.strip().lower().replace("'", "")
        b = text_b.strip().lower().replace("'", "")
        if a == b:
            return 1.0

        is_syn, syn_score = self.are_synonyms_or_inflections(a, b)
        phon_sim = self.lexicon.phonetic_similarity(a, b)
        vec_sim = float(np.dot(self.encode(a), self.encode(b)))

        eff_phon = phon_sim * 0.95 if phon_sim >= 0.76 else phon_sim * 0.62
        best = max(syn_score, eff_phon, vec_sim)
        if is_syn and phon_sim > 0.45:
            best = min(0.99, best + 0.05)
        return round(float(np.clip(best, 0.0, 1.0)), 4)

    def compositional_phrase_similarity(self, phrase_a: str, phrase_b: str) -> float:
        wa = phrase_a.strip().lower().split()
        wb = phrase_b.strip().lower().split()
        if len(wa) != len(wb) or len(wa) < 2:
            return 0.0
        scores: List[float] = []
        for tok_a, tok_b in zip(wa, wb):
            s = self.similarity(tok_a, tok_b)
            if s < 0.78:
                return 0.0
            scores.append(s)
        return round(float(sum(scores) / len(scores)), 4)


@dataclass
class SelectedChunk:
    target_text: str
    matched_text: str
    clip_id: int
    tier: str
    similarity: float
    mode_score: float
    pitch_hz: float
    spectral_centroid: float
    rms_energy: float
    applied_gain: float
    duration_ms: float
    speaker_tag: str
    is_locked_phrase: bool
    rel_start: float = 0.0
    rel_end: float = 1.0
    subword_parts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StitchResult:
    request_text: str
    assembled_text: str
    voice_mode: str
    samples_f32: np.ndarray
    samples_int16: np.ndarray
    sample_rate: int
    chunks: List[SelectedChunk] = field(default_factory=list)
    search_ms: float = 0.0
    dsp_stitch_ms: float = 0.0
    total_ms: float = 0.0
    fx_mode: str = "clean"


@dataclass
class _CandidateMatch:
    span: int
    target_text: str
    matched_text: str
    primary_clip: ClipRecord
    tier: str
    similarity: float
    rel_start: float = 0.0
    rel_end: float = 1.0
    subword_clips: List[Tuple[ClipRecord, SyllableSliceSpec]] = field(default_factory=list)


class FrankensteinStitcher:
    def __init__(
        self,
        db: ArenaDatabase,
        config: Optional[VoxSpliceConfig] = None,
        brain: Optional[StaticEmbeddingBrain] = None,
    ) -> None:
        self.db = db
        self.config = config or db.config
        self.brain = brain or StaticEmbeddingBrain(self.config)
        self.lexicon = PhoneticLexicon.get_default()
        self._rng = np.random.default_rng()

    def _score_candidate_for_mode(
        self,
        clip: ClipRecord,
        base_similarity: float,
        span: int,
        mode: str,
        prev_clip: Optional[ClipRecord],
        victim_ref: Optional[ClipRecord],
        used_clip_ids: Set[int],
    ) -> float:
        priority_bonus = 0.24 if clip.priority > 0 else 0.0
        clean_bonus = max(0.0, 0.10 * (1.0 - min(1.0, clip.seam_rms_ratio)))
        # Exact Tier 1 match gets a decisive +0.55 tier priority so mode bonuses choose AMONG exact
        # clips first and never replace an available exact word with a synonym!
        exact_tier_bonus = 0.55 if base_similarity >= 0.99 else 0.0
        phrase_bonus = (0.30 * (span - 1)) if (span > 1 and base_similarity >= 0.99) else (0.14 * (span - 1) if span > 1 else 0.0)
        score = base_similarity + exact_tier_bonus + priority_bonus + clean_bonus + phrase_bonus

        if clip.id in used_clip_ids:
            score *= 0.35

        mode_norm = mode.strip().lower()
        if mode_norm == "normal":
            mode_norm = "natural"

        if mode_norm == "low_pitch":
            if clip.pitch_hz <= self.config.low_pitch_cutoff_hz:
                depth_bonus = min(
                    0.32, (self.config.low_pitch_cutoff_hz - clip.pitch_hz) / 85.0 + 0.16
                )
                score += depth_bonus
            else:
                penalty = min(0.28, (clip.pitch_hz - self.config.low_pitch_cutoff_hz) / 160.0)
                score -= penalty

        elif mode_norm == "high_pitch":
            if clip.pitch_hz >= self.config.high_pitch_cutoff_hz:
                high_bonus = min(
                    0.32, (clip.pitch_hz - self.config.high_pitch_cutoff_hz) / 110.0 + 0.16
                )
                score += high_bonus
            else:
                penalty = min(0.28, (self.config.high_pitch_cutoff_hz - clip.pitch_hz) / 140.0)
                score -= penalty

        elif mode_norm == "victim_lock" and victim_ref is not None:
            if self.config.locked_victim_id and clip.speaker_tag == self.config.locked_victim_id:
                score += 0.35
            pitch_dist = abs(clip.pitch_hz - victim_ref.pitch_hz) / 55.0
            cent_dist = abs(clip.spectral_centroid - victim_ref.spectral_centroid) / 480.0
            timbre_dist = math.sqrt(pitch_dist * pitch_dist + cent_dist * cent_dist)
            score += 0.30 * math.exp(-timbre_dist)

        elif mode_norm == "chaos" and prev_clip is not None:
            pitch_jump = min(1.2, abs(clip.pitch_hz - prev_clip.pitch_hz) / 55.0)
            cent_jump = min(1.2, abs(clip.spectral_centroid - prev_clip.spectral_centroid) / 450.0)
            spk_switch = 0.16 if clip.speaker_tag != prev_clip.speaker_tag else 0.0
            speaker_diff_bonus = 0.16 * pitch_jump + 0.12 * cent_jump + spk_switch
            if clip.id == prev_clip.id:
                score *= 0.25
            else:
                score += speaker_diff_bonus

        elif mode_norm == "natural" and prev_clip is not None:
            pitch_diff = abs(clip.pitch_hz - prev_clip.pitch_hz)
            cent_diff = abs(clip.spectral_centroid - prev_clip.spectral_centroid)
            continuity = math.exp(-(pitch_diff / 85.0) - (cent_diff / 800.0))
            score += 0.12 * continuity
            if clip.speaker_tag == prev_clip.speaker_tag and clip.priority > 0:
                score += 0.06

        elif mode_norm == "random":
            score += float(self._rng.uniform(-0.08, 0.14))

        return round(float(score), 5)

    def _build_surgical_index(
        self, live_clips: Sequence[ClipRecord]
    ) -> Dict[str, List[Tuple[ClipRecord, float, float]]]:
        surgical: Dict[str, List[Tuple[ClipRecord, float, float]]] = {}
        for c in live_clips:
            if c.word_count <= 1:
                continue
            try:
                offsets = json.loads(c.word_offsets_json or "[]")
            except Exception:
                offsets = []
            if offsets and len(offsets) == c.word_count:
                for item in offsets:
                    w = str(item.get("w", "")).strip().lower().replace("'", "")
                    r0 = float(item.get("r0", 0.0))
                    r1 = float(item.get("r1", 1.0))
                    if w and (r1 - r0) >= 0.12:
                        surgical.setdefault(w, []).append((c, r0, r1))
            else:
                words = [w.replace("'", "") for w in c.text.split()]
                weights = [max(1, len(self.lexicon.get_phones(w))) for w in words]
                tot = float(sum(weights)) or 1.0
                acc = 0.0
                for w, wt in zip(words, weights):
                    r0 = acc / tot
                    acc += wt
                    r1 = acc / tot
                    if (r1 - r0) >= 0.12:
                        surgical.setdefault(w, []).append((c, round(r0, 4), round(r1, 4)))
        return surgical

    def _plan_sentence_viterbi(
        self,
        tokens: Sequence[str],
        all_clips: Sequence[ClipRecord],
        mode: str,
    ) -> List[Tuple[_CandidateMatch, float]]:
        n = len(tokens)
        if n == 0 or not all_clips:
            return []

        by_text: Dict[str, List[ClipRecord]] = {}
        live_clips: List[ClipRecord] = []
        live_by_wc: Dict[int, List[ClipRecord]] = {}
        live_single_by_word: Dict[str, ClipRecord] = {}

        contraction_aliases = {
            "im": "i am", "i am": "im",
            "ill": "i will", "i will": "ill",
            "ive": "i have", "i have": "ive",
            "youre": "you are", "you are": "youre",
            "youd": "you would", "you would": "youd",
            "theyre": "they are", "they are": "theyre",
            "were": "we are",
            "theres": "there is", "there is": "theres",
            "thats": "that is", "that is": "thats",
            "whats": "what is", "what is": "whats",
            "dont": "do not", "do not": "dont",
            "cant": "can not", "can not": "cant",
            "wont": "will not", "will not": "wont",
        }
        for c in all_clips:
            norm_key = " ".join(c.text.lower().replace("'", "").split())
            by_text.setdefault(norm_key, []).append(c)
            for k_src, k_dst in contraction_aliases.items():
                if f" {norm_key} ".find(f" {k_src} ") != -1:
                    alt_key = f" {norm_key} ".replace(f" {k_src} ", f" {k_dst} ").strip()
                    by_text.setdefault(alt_key, []).append(c)
            if c.priority > 0:
                live_clips.append(c)
                live_by_wc.setdefault(c.word_count, []).append(c)
                if c.word_count == 1 and norm_key not in live_single_by_word:
                    live_single_by_word[norm_key] = c

        surgical_index = self._build_surgical_index(live_clips)

        victim_ref: Optional[ClipRecord] = None
        if mode == "victim_lock":
            if self.config.locked_victim_id:
                for c in live_clips:
                    if c.speaker_tag == self.config.locked_victim_id:
                        victim_ref = c
                        break
            if victim_ref is None and live_clips:
                victim_ref = live_clips[0]

        max_span = max(4, int(self.config.max_bleed_chain_words) + 1)

        dp: List[Optional[Tuple[float, List[Tuple[_CandidateMatch, float]], Set[int], Optional[ClipRecord]]]] = [
            None
        ] * (n + 1)
        dp[0] = (0.0, [], set(), None)

        for i in range(n):
            state_i = dp[i]
            if state_i is None:
                continue
            score_so_far, path_so_far, used_so_far, prev_clip = state_i

            for span in range(1, min(max_span, n - i) + 1):
                j = i + span
                phrase = " ".join(tokens[i:j])
                candidates_for_span: List[_CandidateMatch] = []

                # Tier 1: Exact Live Phrase or Word Match (`priority > 0`)
                exact_live = [c for c in by_text.get(phrase, []) if c.priority > 0]
                for c in exact_live[:5]:
                    candidates_for_span.append(
                        _CandidateMatch(
                            span=span,
                            target_text=phrase,
                            matched_text=c.text,
                            primary_clip=c,
                            tier="Tier 1 (Exact Live)",
                            similarity=1.0,
                        )
                    )

                # Tier 2: Surgical Sub-Phrase Word Extraction (`span == 1`)
                if span == 1 and not exact_live and phrase in surgical_index:
                    for c, r0, r1 in surgical_index[phrase][:2]:
                        candidates_for_span.append(
                            _CandidateMatch(
                                span=1,
                                target_text=phrase,
                                matched_text=phrase,
                                primary_clip=c,
                                tier="Tier 2 (Surgical Sub-Phrase)",
                                similarity=0.96,
                                rel_start=r0,
                                rel_end=r1,
                            )
                        )

                # Tier 3: Sub-Word Syllable / Prefix-Suffix Splicing (`span == 1`)
                if (
                    span == 1
                    and not candidates_for_span
                    and self.config.enable_subword_splicing
                    and live_single_by_word
                ):
                    splice_plan = self.lexicon.find_subword_splice_plan(
                        phrase, list(live_single_by_word.keys())
                    )
                    if splice_plan:
                        sub_pairs: List[Tuple[ClipRecord, SyllableSliceSpec]] = []
                        valid_plan = True
                        for spec in splice_plan:
                            src_c = live_single_by_word.get(spec.source_word)
                            if src_c is None:
                                valid_plan = False
                                break
                            sub_pairs.append((src_c, spec))
                        if valid_plan and sub_pairs:
                            label = "+".join(
                                f"{sp.source_word}[{sp.part_type}]" for _c, sp in sub_pairs
                            )
                            candidates_for_span.append(
                                _CandidateMatch(
                                    span=1,
                                    target_text=phrase,
                                    matched_text=f"{phrase} ({label})",
                                    primary_clip=sub_pairs[0][0],
                                    tier="Tier 3 (Sub-Word Syllable Splice)",
                                    similarity=0.89,
                                    subword_clips=sub_pairs,
                                )
                            )

                # Tier 4A: Compositional Multi-Word Phrase Synonym Match (`span == 2`)
                # Only trigger if at least one of the two words in `tokens[i:j]` is NOT available as an exact live word!
                has_missing_word = any(
                    not any(c.priority > 0 for c in by_text.get(tok, [])) for tok in tokens[i:j]
                )
                if not candidates_for_span and span == 2 and has_missing_word and live_clips:
                    pool_2 = [c for c in live_by_wc.get(2, []) if c.id not in used_so_far]
                    scored_2: List[Tuple[float, ClipRecord]] = []
                    for c in pool_2:
                        comp_sim = self.brain.compositional_phrase_similarity(phrase, c.text)
                        if comp_sim >= 0.80:
                            scored_2.append((comp_sim, c))
                    scored_2.sort(key=lambda x: x[0], reverse=True)
                    for sim, c in scored_2[:2]:
                        candidates_for_span.append(
                            _CandidateMatch(
                                span=2,
                                target_text=phrase,
                                matched_text=c.text,
                                primary_clip=c,
                                tier="Tier 4 (Compositional Phrase Synonym)",
                                similarity=sim,
                            )
                        )

                # Tier 4B: 1-Word Semantic Synonym, Morphological Stem & Phonetic Match (`span == 1`)
                if not candidates_for_span and span == 1 and live_clips:
                    pool = [c for c in live_by_wc.get(1, []) if c.id not in used_so_far]
                    scored_live: List[Tuple[float, ClipRecord]] = []
                    for c in pool:
                        sim = self.brain.similarity(phrase, c.text)
                        if sim >= self.config.semantic_similarity_cutoff:
                            scored_live.append((sim, c))
                    scored_live.sort(key=lambda x: x[0], reverse=True)
                    for sim, c in scored_live[:3]:
                        candidates_for_span.append(
                            _CandidateMatch(
                                span=1,
                                target_text=phrase,
                                matched_text=c.text,
                                primary_clip=c,
                                tier="Tier 4 (Semantic/Phonetic Live)",
                                similarity=sim,
                            )
                        )

                # Tier 5: Articulated Real-Speech Starter Pack (`priority == 0`) or Phonetic Fallback
                if not candidates_for_span and self.config.use_starter_pack_fallback:
                    starter_exact = [c for c in by_text.get(phrase, []) if c.priority == 0]
                    for c in starter_exact[:2]:
                        candidates_for_span.append(
                            _CandidateMatch(
                                span=span,
                                target_text=phrase,
                                matched_text=c.text,
                                primary_clip=c,
                                tier="Tier 5 (Starter Pack)",
                                similarity=0.88,
                            )
                        )

                if not candidates_for_span and span == 1:
                    one_word_live = [c for c in live_by_wc.get(1, []) if c.id not in used_so_far]
                    one_word_all = [c for c in all_clips if c.word_count == 1 and c.id not in used_so_far]
                    fallback_pool = one_word_live or one_word_all or list(all_clips)
                    if fallback_pool:
                        best_fb = max(
                            fallback_pool,
                            key=lambda c: self.brain.similarity(phrase, c.text),
                        )
                        fb_sim = self.brain.similarity(phrase, best_fb.text)
                        candidates_for_span.append(
                            _CandidateMatch(
                                span=1,
                                target_text=phrase,
                                matched_text=best_fb.text,
                                primary_clip=best_fb,
                                tier="Tier 4 (Phonetic Live)" if best_fb.priority > 0 else "Tier 5 (Starter Fallback)",
                                similarity=fb_sim,
                            )
                        )

                for cand in candidates_for_span:
                    step_score = self._score_candidate_for_mode(
                        clip=cand.primary_clip,
                        base_similarity=cand.similarity,
                        span=span,
                        mode=mode,
                        prev_clip=prev_clip,
                        victim_ref=victim_ref,
                        used_clip_ids=used_so_far,
                    )
                    total_j = score_so_far + step_score * span
                    if dp[j] is None or total_j > dp[j][0]:
                        new_used = set(used_so_far)
                        if cand.rel_start == 0.0 and cand.rel_end == 1.0 and not cand.subword_clips:
                            new_used.add(cand.primary_clip.id)
                        dp[j] = (
                            total_j,
                            path_so_far + [(cand, step_score)],
                            new_used,
                            cand.primary_clip,
                        )

        final_state = dp[n]
        if final_state is None:
            return []
        return final_state[1]

    def normalize_clip_gain(self, samples_f32: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(samples_f32) == 0:
            return samples_f32, 1.0

        centered = samples_f32 - float(np.mean(samples_f32))
        clip_rms = compute_rms(centered)
        if clip_rms < 1e-5:
            return centered.astype(np.float32), 1.0

        raw_ratio = self.config.target_rms / clip_rms
        gain = float(np.clip(raw_ratio**0.55, 0.35, self.config.max_gain_boost))
        scaled = centered * gain

        peak = float(np.max(np.abs(scaled)))
        if peak > 0.90:
            scaled = scaled * (0.90 / peak)
            gain = gain * (0.90 / peak)
        return scaled.astype(np.float32), round(gain, 2)

    def _phase_aligned_seam_join(
        self, left: np.ndarray, right: np.ndarray, fade_samples: int
    ) -> np.ndarray:
        cur_fade = min(fade_samples, len(left) // 6, len(right) // 6)
        if cur_fade < 8:
            return np.concatenate([left, right])

        tail = left[-cur_fade:]
        max_shift = min(cur_fade // 3, len(right) - cur_fade - 1, 32)
        best_shift = 0
        if max_shift > 4:
            best_corr = -1e18
            for sh in range(0, max_shift, 2):
                cand = right[sh : sh + cur_fade]
                corr = float(np.dot(tail, cand))
                if corr > best_corr:
                    best_corr = corr
                    best_shift = sh

        head = right[best_shift : best_shift + cur_fade]
        rest = right[best_shift + cur_fade :]

        theta = np.linspace(0.0, 0.5 * np.pi, cur_fade, dtype=np.float32)
        fade_out = np.cos(theta) ** 2
        fade_in = np.sin(theta) ** 2

        seam = tail * fade_out + head * fade_in
        return np.concatenate([left[:-cur_fade], seam, rest])

    def crossfade_chunks(
        self, chunks_f32: Sequence[np.ndarray], sample_rate: int = 16000
    ) -> np.ndarray:
        valid = [c for c in chunks_f32 if len(c) > 0]
        if not valid:
            return np.zeros(0, dtype=np.float32)
        if len(valid) == 1:
            out = valid[0].copy()
            edge = min(len(out) // 6, int(sample_rate * 0.003))
            if edge > 2:
                out[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)
                out[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
            return out

        fade_samples = max(24, min(int(sample_rate * 0.006), int(round((self.config.crossfade_ms / 1000.0) * sample_rate))))
        output = valid[0].copy()

        edge = min(len(output) // 6, int(sample_rate * 0.003))
        if edge > 2:
            output[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)

        for nxt in valid[1:]:
            output = self._phase_aligned_seam_join(output, nxt, fade_samples)

        if edge > 2 and len(output) > edge:
            output[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)

        return output.astype(np.float32)

    def suggest_playable_sentences(self, count: int = 5) -> List[str]:
        live_clips = self.db.get_all_clips(include_starter=False)
        if not live_clips:
            return [
                "hello my name is the real human bro",
                "behind you watch out right now",
                "who is that player over here",
                "stop talking and listen to me",
                "i am a real human in the room",
            ][:count]

        all_texts = list(dict.fromkeys([c.text for c in live_clips]))
        if len(all_texts) == 1:
            return [all_texts[0]] * min(count, 1)

        rng = np.random.default_rng(seed=int(time.time()) % 10000 + len(all_texts))
        suggestions: List[str] = []
        for idx in range(count * 3):
            if len(suggestions) >= count:
                break
            target_len = 3 if (idx % 2 == 0) else 4
            pool = list(all_texts)
            rng.shuffle(pool)
            chosen: List[str] = []
            total_w = 0
            for piece in pool:
                wc = len(piece.split())
                if total_w + wc <= 7:
                    chosen.append(piece)
                    total_w += wc
                if len(chosen) >= target_len or total_w >= 6:
                    break
            if chosen:
                sent = " ".join(chosen)
                if sent not in suggestions:
                    suggestions.append(sent)

        return suggestions[:count]

    def preview_sentence_plan(
        self, text: str, voice_mode: Optional[str] = None
    ) -> List[SelectedChunk]:
        mode = voice_mode or self.config.voice_mode
        tokens = [
            w for w in re.sub(r"[^a-z0-9\s']", " ", text.lower()).replace("'", "").split() if w
        ]
        if not tokens:
            return []

        all_clips = self.db.get_all_clips(include_starter=self.config.use_starter_pack_fallback)
        if not all_clips:
            return []

        planned = self._plan_sentence_viterbi(tokens, all_clips, mode)
        plan: List[SelectedChunk] = []
        for cand, m_score in planned:
            clip = cand.primary_clip
            dur_ms = (clip.pcm_length / float(max(1, clip.sample_rate))) * 1000.0
            if clip.audio_format == "pcm_16k":
                dur_ms *= 0.5
            dur_ms *= max(0.15, cand.rel_end - cand.rel_start)
            plan.append(
                SelectedChunk(
                    target_text=cand.target_text,
                    matched_text=cand.matched_text,
                    clip_id=clip.id,
                    tier=cand.tier,
                    similarity=cand.similarity,
                    mode_score=m_score,
                    pitch_hz=clip.pitch_hz,
                    spectral_centroid=clip.spectral_centroid,
                    rms_energy=clip.rms_energy,
                    applied_gain=round(
                        min(
                            self.config.max_gain_boost,
                            (self.config.target_rms / max(1e-4, clip.rms_energy)) ** 0.55,
                        ),
                        2,
                    ),
                    duration_ms=round(dur_ms, 1),
                    speaker_tag=clip.speaker_tag,
                    is_locked_phrase=bool(clip.is_locked_phrase),
                    rel_start=cand.rel_start,
                    rel_end=cand.rel_end,
                    subword_parts=[
                        {
                            "source_word": sp.source_word,
                            "part_type": sp.part_type,
                            "rel_start": sp.rel_start,
                            "rel_end": sp.rel_end,
                        }
                        for _c, sp in cand.subword_clips
                    ],
                )
            )
        return plan

    def stitch_sentence(
        self,
        text: str,
        voice_mode: Optional[str] = None,
        fx_mode: Optional[str] = None,
        output_sample_rate: int = 16000,
        speaker_filter: Optional[str] = None,
    ) -> StitchResult:
        t0 = time.perf_counter()
        mode = voice_mode or self.config.voice_mode
        active_fx = fx_mode or self.config.fx_mode
        tokens = [
            w for w in re.sub(r"[^a-z0-9\s']", " ", text.lower()).replace("'", "").split() if w
        ]
        if not tokens:
            empty = np.zeros(0, dtype=np.float32)
            return StitchResult(
                request_text=text,
                assembled_text="",
                voice_mode=mode,
                samples_f32=empty,
                samples_int16=np.zeros(0, dtype=np.int16),
                sample_rate=output_sample_rate,
                fx_mode=active_fx,
            )

        all_clips = self.db.get_all_clips(
            include_starter=self.config.use_starter_pack_fallback,
            speaker_tag=speaker_filter,
        )
        if not all_clips and speaker_filter:
            all_clips = self.db.get_all_clips(include_starter=self.config.use_starter_pack_fallback)

        planned = self._plan_sentence_viterbi(tokens, all_clips, mode)
        t_search = time.perf_counter()

        decoded_chunks: List[np.ndarray] = []
        selected_meta: List[SelectedChunk] = []

        for idx, (cand, m_score) in enumerate(planned):
            clip = cand.primary_clip
            if cand.subword_clips:
                sub_f32_parts: List[np.ndarray] = []
                for sub_c, spec in cand.subword_clips:
                    part_f32 = self.db.decode_clip_subrange_to_float32(
                        sub_c, spec.rel_start, spec.rel_end, target_sr=output_sample_rate
                    )
                    norm_part, _g = self.normalize_clip_gain(part_f32)
                    sub_f32_parts.append(norm_part)
                if len(sub_f32_parts) == 1:
                    raw_f32 = sub_f32_parts[0]
                else:
                    sub_fade = max(16, int(round(0.006 * output_sample_rate)))
                    raw_f32 = sub_f32_parts[0]
                    for nxt_p in sub_f32_parts[1:]:
                        raw_f32 = self._phase_aligned_seam_join(raw_f32, nxt_p, sub_fade)
                norm_f32, gain = self.normalize_clip_gain(raw_f32)
            elif cand.rel_start > 0.0 or cand.rel_end < 1.0:
                raw_f32 = self.db.decode_clip_subrange_to_float32(
                    clip, cand.rel_start, cand.rel_end, target_sr=output_sample_rate
                )
                norm_f32, gain = self.normalize_clip_gain(raw_f32)
            else:
                raw_f32 = self.db.decode_clip_to_float32(clip, target_sr=output_sample_rate)
                norm_f32, gain = self.normalize_clip_gain(raw_f32)

            fx_f32 = process_chunk_fx(
                norm_f32,
                fx_mode=active_fx,
                sample_rate=output_sample_rate,
                intensity=self.config.fx_intensity,
                chunk_index=idx,
            )
            decoded_chunks.append(fx_f32)
            dur_ms = (len(fx_f32) / float(output_sample_rate)) * 1000.0
            clean_matched = cand.matched_text.split(" (")[0]
            selected_meta.append(
                SelectedChunk(
                    target_text=cand.target_text,
                    matched_text=clean_matched,
                    clip_id=clip.id,
                    tier=cand.tier,
                    similarity=cand.similarity,
                    mode_score=m_score,
                    pitch_hz=clip.pitch_hz,
                    spectral_centroid=clip.spectral_centroid,
                    rms_energy=clip.rms_energy,
                    applied_gain=gain,
                    duration_ms=round(dur_ms, 1),
                    speaker_tag=clip.speaker_tag,
                    is_locked_phrase=bool(clip.is_locked_phrase),
                    rel_start=cand.rel_start,
                    rel_end=cand.rel_end,
                    subword_parts=[
                        {
                            "source_word": sp.source_word,
                            "part_type": sp.part_type,
                            "rel_start": sp.rel_start,
                            "rel_end": sp.rel_end,
                        }
                        for _c, sp in cand.subword_clips
                    ],
                )
            )

        stitched_f32 = self.crossfade_chunks(decoded_chunks, sample_rate=output_sample_rate)
        stitched_i16 = float32_to_int16(stitched_f32)
        t_end = time.perf_counter()

        search_ms = (t_search - t0) * 1000.0
        dsp_ms = (t_end - t_search) * 1000.0
        total_ms = (t_end - t0) * 1000.0
        assembled = " ".join(c.matched_text for c in selected_meta)

        return StitchResult(
            request_text=text,
            assembled_text=assembled,
            voice_mode=mode,
            samples_f32=stitched_f32,
            samples_int16=stitched_i16,
            sample_rate=output_sample_rate,
            chunks=selected_meta,
            search_ms=round(search_ms, 2),
            dsp_stitch_ms=round(dsp_ms, 2),
            total_ms=round(total_ms, 2),
            fx_mode=active_fx,
        )
