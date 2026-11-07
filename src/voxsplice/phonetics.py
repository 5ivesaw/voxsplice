"""
phonetics.py — CMU 135k ARPAbet Lexicon, Grapheme-to-Phoneme (G2P), Articulatory Phoneme Distance,
and Sub-Word Syllable Splicing Engine.

Enables VoxSplice to:
1. Map any English word to its exact ARPAbet phoneme sequence (via `/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict`
   or built-in rule-based G2P fallback).
2. Compute articulatory phoneme similarity so missing words match homophones, inflections, or
   acoustically similar words from the target speaker.
3. Estimate proportional acoustic durations of phonemes and syllables within a recorded word so
   missing words can be synthesized by splicing prefixes/suffixes ("halves of words") from existing recordings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}
_STOPS = {"P", "B", "T", "D", "K", "G"}
_FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
_AFFRICATES = {"CH", "JH"}
_NASALS = {"M", "N", "NG"}
_LIQUIDS_GLIDES = {"L", "R", "W", "Y"}

_ARTICULATORY_GROUP: Dict[str, str] = {
    "IY": "v_front_high", "IH": "v_front_high", "EY": "v_front_mid", "EH": "v_front_mid", "AE": "v_front_low",
    "AA": "v_back_low", "AO": "v_back_low", "AH": "v_central", "ER": "v_central",
    "OW": "v_back_mid", "UH": "v_back_high", "UW": "v_back_high",
    "AY": "v_diph", "AW": "v_diph", "OY": "v_diph",
    "P": "stop_labial", "B": "stop_labial",
    "T": "stop_alveolar", "D": "stop_alveolar",
    "K": "stop_velar", "G": "stop_velar",
    "F": "fric_labial", "V": "fric_labial",
    "TH": "fric_dental", "DH": "fric_dental",
    "S": "fric_sibilant", "Z": "fric_sibilant", "SH": "fric_sibilant", "ZH": "fric_sibilant",
    "CH": "affricate", "JH": "affricate",
    "M": "nasal_labial", "N": "nasal_alveolar", "NG": "nasal_velar",
    "L": "liquid", "R": "liquid", "W": "glide", "Y": "glide", "HH": "aspirate",
}


def strip_stress(phone: str) -> str:
    return re.sub(r"[012]$", "", phone.upper())


def phone_duration_weight(phone: str) -> float:
    base = strip_stress(phone)
    if base in _VOWELS:
        if phone.endswith("1"):
            return 1.45
        if phone.endswith("2"):
            return 1.25
        return 1.10
    if base in _FRICATIVES or base in _AFFRICATES:
        return 0.85
    if base in _NASALS or base in _LIQUIDS_GLIDES:
        return 0.80
    if base in _STOPS:
        return 0.55
    return 0.75


def _rule_based_g2p(word: str) -> List[str]:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return []

    digraphs = [
        ("tion", ["SH", "AH0", "N"]),
        ("sion", ["ZH", "AH0", "N"]),
        ("ough", ["AO1"]),
        ("igh", ["AY1"]),
        ("tch", ["CH"]),
        ("sh", ["SH"]),
        ("ch", ["CH"]),
        ("th", ["TH"]),
        ("ph", ["F"]),
        ("wh", ["W"]),
        ("ck", ["K"]),
        ("ng", ["NG"]),
        ("qu", ["K", "W"]),
        ("ee", ["IY1"]),
        ("ea", ["IY1"]),
        ("oo", ["UW1"]),
        ("ou", ["AW1"]),
        ("ow", ["OW1"]),
        ("ai", ["EY1"]),
        ("ay", ["EY1"]),
        ("oi", ["OY1"]),
        ("oy", ["OY1"]),
        ("ar", ["AA1", "R"]),
        ("er", ["ER0"]),
        ("ir", ["ER1"]),
        ("or", ["AO1", "R"]),
        ("ur", ["ER1"]),
    ]
    single_map = {
        "a": ["AE1"],
        "b": ["B"],
        "c": ["K"],
        "d": ["D"],
        "e": ["EH1"],
        "f": ["F"],
        "g": ["G"],
        "h": ["HH"],
        "i": ["IH1"],
        "j": ["JH"],
        "k": ["K"],
        "l": ["L"],
        "m": ["M"],
        "n": ["N"],
        "o": ["AA1"],
        "p": ["P"],
        "q": ["K"],
        "r": ["R"],
        "s": ["S"],
        "t": ["T"],
        "u": ["AH1"],
        "v": ["V"],
        "w": ["W"],
        "x": ["K", "S"],
        "y": ["Y"],
        "z": ["Z"],
    }

    phones: List[str] = []
    i = 0
    n = len(w)
    silent_e = n >= 4 and w.endswith("e") and w[-2] not in "aeiouy" and any(c in "aeiouy" for c in w[:-2])
    limit = n - 1 if silent_e else n

    while i < limit:
        matched = False
        for dg, phs in digraphs:
            if w.startswith(dg, i):
                phones.extend(phs)
                i += len(dg)
                matched = True
                break
        if matched:
            continue
        ch = w[i]
        if i + 1 < limit and w[i + 1] == ch and ch not in "aeiou":
            i += 1
        phones.extend(single_map.get(ch, []))
        i += 1

    return phones or ["AH0"]


@dataclass
class SyllableSliceSpec:
    source_word: str
    phones: List[str]
    rel_start: float = 0.0
    rel_end: float = 1.0
    part_type: str = "full"


class PhoneticLexicon:
    _instance: Optional["PhoneticLexicon"] = None

    def __init__(self, dict_path: Optional[Path] = None) -> None:
        self.entries: Dict[str, List[str]] = {}
        self._load_dictionary(dict_path)

    @classmethod
    def get_default(cls) -> "PhoneticLexicon":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_dictionary(self, custom_path: Optional[Path] = None) -> None:
        candidates = []
        if custom_path:
            candidates.append(custom_path)
        candidates.extend(
            [
                Path("/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict"),
                Path.home() / ".local" / "share" / "voxsplice" / "models" / "cmu-arpabet-g2p" / "cmudict-en-us.dict",
            ]
        )
        for path in candidates:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith(";;;"):
                                continue
                            parts = line.split()
                            if len(parts) < 2:
                                continue
                            word = parts[0].lower()
                            if "(" in word:
                                base_w = word.split("(", 1)[0]
                                if base_w in self.entries:
                                    continue
                                word = base_w
                            self.entries[word] = parts[1:]
                    break
                except Exception:
                    pass

        shorthand = {
            "im": ["AY1", "M"],
            "u": ["Y", "UW1"],
            "ur": ["Y", "AO1", "R"],
            "r": ["AA1", "R"],
            "bro": ["B", "R", "OW1"],
            "bruh": ["B", "R", "AH1"],
            "nah": ["N", "AA1"],
            "yep": ["Y", "EH1", "P"],
            "nope": ["N", "OW1", "P"],
            "sup": ["S", "AH1", "P"],
            "ive": ["AY1", "V"],
            "dont": ["D", "OW1", "N", "T"],
            "cant": ["K", "AE1", "N", "T"],
            "wont": ["W", "OW1", "N", "T"],
            "didnt": ["D", "IH1", "D", "AH0", "N", "T"],
            "youre": ["Y", "UH1", "R"],
            "theyre": ["DH", "EH1", "R"],
            "thats": ["DH", "AE1", "T", "S"],
            "whats": ["W", "AH1", "T", "S"],
        }
        for k, v in shorthand.items():
            self.entries.setdefault(k, v)

    def get_phones(self, word: str) -> List[str]:
        clean = re.sub(r"[^a-z']", "", word.lower().strip())
        if not clean:
            return []
        if clean in self.entries:
            return self.entries[clean]
        no_apos = clean.replace("'", "")
        if no_apos in self.entries:
            return self.entries[no_apos]
        return _rule_based_g2p(no_apos)

    def get_unstressed_phones(self, word: str) -> List[str]:
        return [strip_stress(p) for p in self.get_phones(word)]

    def phrase_phones(self, phrase: str) -> List[str]:
        out: List[str] = []
        for w in phrase.lower().split():
            out.extend(self.get_unstressed_phones(w))
        return out

    @staticmethod
    def _phone_sub_cost(p1: str, p2: str) -> float:
        b1, b2 = strip_stress(p1), strip_stress(p2)
        if b1 == b2:
            return 0.0
        g1 = _ARTICULATORY_GROUP.get(b1, "")
        g2 = _ARTICULATORY_GROUP.get(b2, "")
        if g1 and g1 == g2:
            return 0.28
        if (b1 in _VOWELS and b2 in _VOWELS) or (b1 in _STOPS and b2 in _STOPS) or (b1 in _FRICATIVES and b2 in _FRICATIVES):
            return 0.55
        return 1.0

    def phonetic_similarity(self, text_a: str, text_b: str) -> float:
        pa = self.phrase_phones(text_a)
        pb = self.phrase_phones(text_b)
        if not pa or not pb:
            return 0.0
        if pa == pb:
            return 1.0

        m, n = len(pa), len(pb)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = float(i) * 0.85
        for j in range(1, n + 1):
            dp[0][j] = float(j) * 0.85

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                sub = self._phone_sub_cost(pa[i - 1], pb[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j] + 0.85,
                    dp[i][j - 1] + 0.85,
                    dp[i - 1][j - 1] + sub,
                )

        max_len = float(max(m, n))
        sim = max(0.0, 1.0 - (dp[m][n] / max_len))
        if pa[0] == pb[0]:
            sim = min(1.0, sim + 0.06)
        if len(pa) >= 2 and len(pb) >= 2 and pa[-2:] == pb[-2:]:
            sim = min(1.0, sim + 0.08)
        return round(sim, 4)

    @staticmethod
    def morphological_stem(word: str) -> str:
        w = re.sub(r"[^a-z]", "", word.lower())
        if len(w) <= 3:
            return w
        for suffix in ("ing", "ed", "es", "er", "ly", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                stem = w[: -len(suffix)]
                if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                    stem = stem[:-1]
                return stem
        return w

    def compute_phone_time_intervals(self, word: str) -> List[Tuple[str, float, float]]:
        phones = self.get_phones(word)
        if not phones:
            return []
        weights = [phone_duration_weight(p) for p in phones]
        total_w = sum(weights) or 1.0
        intervals: List[Tuple[str, float, float]] = []
        acc = 0.0
        for p, w in zip(phones, weights):
            r0 = acc / total_w
            acc += w
            r1 = acc / total_w
            intervals.append((strip_stress(p), round(r0, 4), round(r1, 4)))
        return intervals

    def find_subword_splice_plan(
        self,
        target_word: str,
        available_words: Sequence[str],
    ) -> Optional[List[SyllableSliceSpec]]:
        target_phones = self.get_unstressed_phones(target_word)
        if len(target_phones) < 2 or not available_words:
            return None

        unique_avail = list(dict.fromkeys(w.lower().strip() for w in available_words if len(w.strip()) >= 2))
        avail_intervals = {w: self.compute_phone_time_intervals(w) for w in unique_avail}
        avail_phones = {w: [x[0] for x in ivs] for w, ivs in avail_intervals.items() if ivs}

        for w, phs in avail_phones.items():
            if w == target_word or len(phs) <= len(target_phones):
                continue
            ivs = avail_intervals[w]
            k = len(target_phones)
            if phs[:k] == target_phones:
                return [
                    SyllableSliceSpec(
                        source_word=w,
                        phones=target_phones,
                        rel_start=0.0,
                        rel_end=min(1.0, ivs[k - 1][2] + 0.04),
                        part_type="prefix",
                    )
                ]
            if phs[-k:] == target_phones:
                return [
                    SyllableSliceSpec(
                        source_word=w,
                        phones=target_phones,
                        rel_start=max(0.0, ivs[-k][1] - 0.04),
                        rel_end=1.0,
                        part_type="suffix",
                    )
                ]

        if len(target_phones) >= 3:
            for split_idx in range(2, len(target_phones)):
                left_target = target_phones[:split_idx]
                right_target = target_phones[split_idx:]

                best_left: Optional[SyllableSliceSpec] = None
                best_right: Optional[SyllableSliceSpec] = None

                for w, phs in avail_phones.items():
                    ivs = avail_intervals[w]
                    if phs == left_target:
                        best_left = SyllableSliceSpec(w, left_target, 0.0, 1.0, "full")
                        break
                    elif len(phs) > len(left_target) and phs[: len(left_target)] == left_target:
                        best_left = SyllableSliceSpec(
                            w, left_target, 0.0, min(1.0, ivs[len(left_target) - 1][2] + 0.03), "prefix"
                        )

                if best_left is None:
                    continue

                for w, phs in avail_phones.items():
                    ivs = avail_intervals[w]
                    if phs == right_target:
                        best_right = SyllableSliceSpec(w, right_target, 0.0, 1.0, "full")
                        break
                    elif len(phs) > len(right_target) and phs[-len(right_target) :] == right_target:
                        best_right = SyllableSliceSpec(
                            w, right_target, max(0.0, ivs[-len(right_target)][1] - 0.03), 1.0, "suffix"
                        )

                if best_left is not None and best_right is not None:
                    return [best_left, best_right]

        return None
