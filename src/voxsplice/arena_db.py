"""
arena_db.py — Two-File Arena Storage System (`audio_arena.pcm` + `clips.db` in SQLite WAL Mode).
"""

from __future__ import annotations

import mmap
import sqlite3
import struct
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import VoxSpliceConfig
from .dsp import (
    apply_micro_edge_fade,
    apply_ulaw_reconstruction_filter,
    decode_ulaw,
    int16_to_float32,
    resample_audio,
    snap_to_zero_crossing,
)
from .phonetics import PhoneticLexicon
from .slicer import SlicedClip


@dataclass
class ClipRecord:
    id: int
    text: str
    word_count: int
    pcm_offset: int
    pcm_length: int
    pitch_hz: float
    rms_energy: float
    spectral_centroid: float
    created_at: float
    seam_rms_ratio: float = 0.10
    is_locked_phrase: int = 0
    priority: int = 1
    pinned: int = 0
    speaker_tag: str = "lobby"
    audio_format: str = "ulaw_12k"
    sample_rate: int = 12000
    confidence: float = 1.0
    phonemes: str = ""
    word_offsets_json: str = "[]"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArenaDatabase:
    def __init__(
        self,
        config: Optional[VoxSpliceConfig] = None,
        arena_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self.config = config or VoxSpliceConfig.load()
        self.arena_path = Path(arena_path) if arena_path else self.config.resolve_arena_path()
        self.db_path = Path(db_path) if db_path else self.config.resolve_db_path()

        self.arena_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._mmap: Optional[mmap.mmap] = None
        self._mmap_size: int = 0

        if not self.arena_path.exists():
            self.arena_path.write_bytes(b"")

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._refresh_mmap()

        self.total_ingested: int = 0
        self.total_dedup_pruned: int = 0
        self.total_ring_evicted: int = 0

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    pcm_offset INTEGER NOT NULL,
                    pcm_length INTEGER NOT NULL,
                    pitch_hz REAL NOT NULL,
                    rms_energy REAL NOT NULL,
                    spectral_centroid REAL NOT NULL,
                    created_at REAL NOT NULL,
                    seam_rms_ratio REAL DEFAULT 0.10,
                    is_locked_phrase INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 1,
                    pinned INTEGER DEFAULT 0,
                    speaker_tag TEXT DEFAULT 'lobby',
                    audio_format TEXT DEFAULT 'ulaw_12k',
                    sample_rate INTEGER DEFAULT 12000,
                    confidence REAL DEFAULT 1.0,
                    phonemes TEXT DEFAULT '',
                    word_offsets_json TEXT DEFAULT '[]'
                );
                """
            )
            cols = {r[1] for r in cur.execute("PRAGMA table_info(clips)").fetchall()}
            if "confidence" not in cols:
                cur.execute("ALTER TABLE clips ADD COLUMN confidence REAL DEFAULT 1.0;")
            if "phonemes" not in cols:
                cur.execute("ALTER TABLE clips ADD COLUMN phonemes TEXT DEFAULT '';")
            if "word_offsets_json" not in cols:
                cur.execute("ALTER TABLE clips ADD COLUMN word_offsets_json TEXT DEFAULT '[]';")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_text ON clips(text);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_priority ON clips(priority, pinned);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_created ON clips(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_clips_speaker ON clips(speaker_tag);")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS free_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pcm_offset INTEGER NOT NULL,
                    pcm_length INTEGER NOT NULL
                );
                """
            )
            self._conn.commit()

    def _refresh_mmap(self) -> None:
        with self._lock:
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None

            if not self.arena_path.exists():
                self._mmap_size = 0
                return

            fsize = self.arena_path.stat().st_size
            self._mmap_size = fsize
            if fsize > 0:
                f = open(self.arena_path, "r+b")
                try:
                    self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                finally:
                    f.close()

    def _write_arena_bytes(self, data: bytes, offset: Optional[int] = None) -> Tuple[int, int]:
        length = len(data)
        if length == 0:
            return 0, 0

        with self._lock:
            if offset is not None:
                with open(self.arena_path, "r+b") as f:
                    f.seek(offset)
                    f.write(data)
                    f.flush()
                self._refresh_mmap()
                return offset, length

            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, pcm_offset, pcm_length
                FROM free_blocks
                WHERE pcm_length >= ?
                ORDER BY pcm_length ASC
                LIMIT 1
                """,
                (length,),
            )
            row = cur.fetchone()
            if row is not None:
                fb_id = int(row["id"])
                fb_offset = int(row["pcm_offset"])
                fb_len = int(row["pcm_length"])
                with open(self.arena_path, "r+b") as f:
                    f.seek(fb_offset)
                    f.write(data)
                    f.flush()

                rem = fb_len - length
                if rem >= 64:
                    cur.execute(
                        "UPDATE free_blocks SET pcm_offset = ?, pcm_length = ? WHERE id = ?",
                        (fb_offset + length, rem, fb_id),
                    )
                else:
                    cur.execute("DELETE FROM free_blocks WHERE id = ?", (fb_id,))
                self._conn.commit()
                self._refresh_mmap()
                return fb_offset, length

            with open(self.arena_path, "ab") as f:
                write_offset = f.tell()
                f.write(data)
                f.flush()
            self._refresh_mmap()
            return write_offset, length

    def _enforce_arena_ring_cap(self, incoming_len: int) -> None:
        max_bytes = self.config.max_arena_bytes
        with self._lock:
            cur_size = self.arena_path.stat().st_size if self.arena_path.exists() else 0
            if cur_size + incoming_len <= max_bytes:
                return

            bytes_needed = (cur_size + incoming_len) - max_bytes + max(4096, incoming_len * 2)
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, pcm_offset, pcm_length
                FROM clips
                WHERE pinned = 0 AND priority > 0
                ORDER BY created_at ASC
                LIMIT 64
                """
            )
            victims = cur.fetchall()
            freed = 0
            for v in victims:
                cid = int(v["id"])
                coff = int(v["pcm_offset"])
                clen = int(v["pcm_length"])
                cur.execute("DELETE FROM clips WHERE id = ?", (cid,))
                cur.execute(
                    "INSERT INTO free_blocks (pcm_offset, pcm_length) VALUES (?, ?)",
                    (coff, clen),
                )
                freed += clen
                self.total_ring_evicted += 1
                if freed >= bytes_needed:
                    break
            self._conn.commit()

    def insert_clip(self, clip: SlicedClip, pinned: int = 0) -> Optional[int]:
        text = clip.text.strip().lower()
        if not text or len(clip.encoded_bytes) == 0:
            return None

        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT id, pcm_offset, pcm_length, pitch_hz, seam_rms_ratio, priority, pinned
                FROM clips
                WHERE text = ?
                ORDER BY priority DESC, seam_rms_ratio ASC
                """,
                (text,),
            )
            existing = cur.fetchall()

            if clip.priority > 0:
                starter_rows = [r for r in existing if int(r["priority"]) == 0 and int(r["pinned"]) == 0]
                for sr in starter_rows:
                    cur.execute("DELETE FROM clips WHERE id = ?", (int(sr["id"]),))
                    cur.execute(
                        "INSERT INTO free_blocks (pcm_offset, pcm_length) VALUES (?, ?)",
                        (int(sr["pcm_offset"]), int(sr["pcm_length"])),
                    )
                if starter_rows:
                    existing = [r for r in existing if int(r["priority"]) > 0 or int(r["pinned"]) > 0]

            unpinned = [r for r in existing if int(r["pinned"]) == 0]
            if len(existing) >= self.config.max_slots_per_phrase and unpinned:
                worst = max(unpinned, key=lambda r: float(r["seam_rms_ratio"]))
                worst_ratio = float(worst["seam_rms_ratio"])
                pitches = [float(r["pitch_hz"]) for r in existing]
                gives_new_pitch = (clip.pitch_hz < min(pitches) - 12.0) or (clip.pitch_hz > max(pitches) + 12.0)

                if clip.seam_rms_ratio >= worst_ratio and not gives_new_pitch:
                    self.total_dedup_pruned += 1
                    return None

                worst_id = int(worst["id"])
                worst_off = int(worst["pcm_offset"])
                worst_len = int(worst["pcm_length"])

                if len(clip.encoded_bytes) <= worst_len:
                    off, length = self._write_arena_bytes(clip.encoded_bytes, offset=worst_off)
                    rem = worst_len - length
                    if rem >= 32:
                        cur.execute(
                            "INSERT INTO free_blocks (pcm_offset, pcm_length) VALUES (?, ?)",
                            (worst_off + length, rem),
                        )
                else:
                    cur.execute(
                        "INSERT INTO free_blocks (pcm_offset, pcm_length) VALUES (?, ?)",
                        (worst_off, worst_len),
                    )
                    self._enforce_arena_ring_cap(len(clip.encoded_bytes))
                    off, length = self._write_arena_bytes(clip.encoded_bytes)

                cur.execute(
                    """
                    UPDATE clips
                    SET word_count = ?, pcm_offset = ?, pcm_length = ?,
                        pitch_hz = ?, rms_energy = ?, spectral_centroid = ?,
                        created_at = ?, seam_rms_ratio = ?, is_locked_phrase = ?,
                        priority = ?, pinned = ?, speaker_tag = ?,
                        audio_format = ?, sample_rate = ?, confidence = ?,
                        phonemes = ?, word_offsets_json = ?
                    WHERE id = ?
                    """,
                    (
                        clip.word_count,
                        off,
                        length,
                        clip.pitch_hz,
                        clip.rms_energy,
                        clip.spectral_centroid,
                        now,
                        clip.seam_rms_ratio,
                        1 if clip.is_locked_phrase else 0,
                        clip.priority,
                        pinned,
                        clip.speaker_tag,
                        clip.audio_format,
                        clip.sample_rate,
                        clip.confidence,
                        clip.phonemes,
                        clip.word_offsets_json,
                        worst_id,
                    ),
                )
                self._conn.commit()
                self.total_dedup_pruned += 1
                self.total_ingested += 1
                return worst_id

            self._enforce_arena_ring_cap(len(clip.encoded_bytes))
            off, length = self._write_arena_bytes(clip.encoded_bytes)
            cur.execute(
                """
                INSERT INTO clips (
                    text, word_count, pcm_offset, pcm_length,
                    pitch_hz, rms_energy, spectral_centroid, created_at,
                    seam_rms_ratio, is_locked_phrase, priority, pinned,
                    speaker_tag, audio_format, sample_rate, confidence,
                    phonemes, word_offsets_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    text,
                    clip.word_count,
                    off,
                    length,
                    clip.pitch_hz,
                    clip.rms_energy,
                    clip.spectral_centroid,
                    now,
                    clip.seam_rms_ratio,
                    1 if clip.is_locked_phrase else 0,
                    clip.priority,
                    pinned,
                    clip.speaker_tag,
                    clip.audio_format,
                    clip.sample_rate,
                    clip.confidence,
                    clip.phonemes,
                    clip.word_offsets_json,
                ),
            )
            new_id = int(cur.lastrowid or 0)
            self._conn.commit()
            self.total_ingested += 1
            return new_id

    def read_clip_bytes(self, clip: ClipRecord) -> bytes:
        with self._lock:
            if self._mmap is not None and (clip.pcm_offset + clip.pcm_length) <= self._mmap_size:
                return self._mmap[clip.pcm_offset : clip.pcm_offset + clip.pcm_length]
            if not self.arena_path.exists():
                return b""
            with open(self.arena_path, "rb") as f:
                f.seek(clip.pcm_offset)
                return f.read(clip.pcm_length)

    def decode_clip_to_float32(self, clip: ClipRecord, target_sr: int = 16000) -> np.ndarray:
        raw = self.read_clip_bytes(clip)
        if not raw:
            return np.zeros(0, dtype=np.float32)

        if clip.audio_format.startswith("ulaw"):
            pcm16 = decode_ulaw(raw)
            f32 = int16_to_float32(pcm16)
            out = resample_audio(f32, clip.sample_rate, target_sr)
            return apply_ulaw_reconstruction_filter(out, sample_rate=target_sr)

        pcm16 = np.frombuffer(raw, dtype=np.int16)
        f32 = int16_to_float32(pcm16)
        return resample_audio(f32, clip.sample_rate, target_sr)

    def decode_clip_subrange_to_float32(
        self,
        clip: ClipRecord,
        rel_start: float,
        rel_end: float,
        target_sr: int = 16000,
    ) -> np.ndarray:
        full_f32 = self.decode_clip_to_float32(clip, target_sr=target_sr)
        n = len(full_f32)
        if n < 32:
            return full_f32

        r0 = max(0.0, min(0.95, float(rel_start)))
        r1 = max(r0 + 0.05, min(1.0, float(rel_end)))
        s_idx = snap_to_zero_crossing(full_f32, int(round(r0 * n)), radius=64)
        e_idx = snap_to_zero_crossing(full_f32, int(round(r1 * n)), radius=64)
        if e_idx <= s_idx + 16:
            s_idx = int(round(r0 * n))
            e_idx = max(s_idx + 16, int(round(r1 * n)))

        sub = full_f32[s_idx:e_idx].copy()
        return apply_micro_edge_fade(sub, sample_rate=target_sr, fade_ms=2.0)

    def get_all_clips(
        self,
        include_starter: bool = True,
        speaker_tag: Optional[str] = None,
    ) -> List[ClipRecord]:
        with self._lock:
            cur = self._conn.cursor()
            query = "SELECT * FROM clips WHERE 1=1"
            params: List[Any] = []
            if not include_starter:
                query += " AND priority > 0"
            if speaker_tag:
                query += " AND speaker_tag = ?"
                params.append(speaker_tag)
            query += " ORDER BY priority DESC, created_at DESC"
            rows = cur.execute(query, params).fetchall()
            return [
                ClipRecord(
                    id=int(r["id"]),
                    text=str(r["text"]),
                    word_count=int(r["word_count"]),
                    pcm_offset=int(r["pcm_offset"]),
                    pcm_length=int(r["pcm_length"]),
                    pitch_hz=float(r["pitch_hz"]),
                    rms_energy=float(r["rms_energy"]),
                    spectral_centroid=float(r["spectral_centroid"]),
                    created_at=float(r["created_at"]),
                    seam_rms_ratio=float(r["seam_rms_ratio"]),
                    is_locked_phrase=int(r["is_locked_phrase"]),
                    priority=int(r["priority"]),
                    pinned=int(r["pinned"]),
                    speaker_tag=str(r["speaker_tag"]),
                    audio_format=str(r["audio_format"]),
                    sample_rate=int(r["sample_rate"]),
                    confidence=float(r["confidence"]) if "confidence" in r.keys() else 1.0,
                    phonemes=str(r["phonemes"]) if "phonemes" in r.keys() else "",
                    word_offsets_json=str(r["word_offsets_json"]) if "word_offsets_json" in r.keys() else "[]",
                )
                for r in rows
            ]

    def pin_speaker_profile(self, profile_name: str, recent_seconds: float = 300.0) -> int:
        cutoff = time.time() - recent_seconds
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE clips
                SET pinned = 1, speaker_tag = ?
                WHERE priority > 0 AND created_at >= ?
                """,
                (profile_name.strip(), cutoff),
            )
            updated = cur.rowcount
            self._conn.commit()
            return updated

    def clear_live_clips(self) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, pcm_offset, pcm_length FROM clips WHERE priority > 0 AND pinned = 0")
            rows = cur.fetchall()
            for r in rows:
                cur.execute(
                    "INSERT INTO free_blocks (pcm_offset, pcm_length) VALUES (?, ?)",
                    (int(r["pcm_offset"]), int(r["pcm_length"])),
                )
            cur.execute("DELETE FROM clips WHERE priority > 0 AND pinned = 0")
            removed = len(rows)
            self._conn.commit()
            return removed

    def list_speaker_profiles(self) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT speaker_tag,
                       COUNT(*) as clip_count,
                       AVG(pitch_hz) as avg_pitch_hz,
                       AVG(spectral_centroid) as avg_centroid_hz,
                       SUM(pinned) as pinned_count
                FROM clips
                WHERE priority > 0
                GROUP BY speaker_tag
                ORDER BY clip_count DESC
                """
            ).fetchall()
            return [
                {
                    "speaker_tag": str(r["speaker_tag"]),
                    "clip_count": int(r["clip_count"]),
                    "avg_pitch_hz": round(float(r["avg_pitch_hz"] or 0.0), 1),
                    "avg_centroid_hz": round(float(r["avg_centroid_hz"] or 0.0), 1),
                    "pinned_count": int(r["pinned_count"] or 0),
                }
                for r in rows
            ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            total_clips = int(cur.execute("SELECT COUNT(*) FROM clips").fetchone()[0])
            live_clips = int(cur.execute("SELECT COUNT(*) FROM clips WHERE priority > 0").fetchone()[0])
            starter_clips = int(cur.execute("SELECT COUNT(*) FROM clips WHERE priority = 0").fetchone()[0])
            pinned_clips = int(cur.execute("SELECT COUNT(*) FROM clips WHERE pinned = 1").fetchone()[0])
            locked_chains = int(
                cur.execute("SELECT COUNT(*) FROM clips WHERE is_locked_phrase = 1 AND priority > 0").fetchone()[0]
            )
            single_words = int(
                cur.execute("SELECT COUNT(*) FROM clips WHERE word_count = 1 AND priority > 0").fetchone()[0]
            )
            unique_phrases = int(
                cur.execute("SELECT COUNT(DISTINCT text) FROM clips WHERE priority > 0").fetchone()[0]
            )
            arena_bytes = self.arena_path.stat().st_size if self.arena_path.exists() else 0

            return {
                "total_clips": total_clips,
                "live_clips": live_clips,
                "starter_clips": starter_clips,
                "pinned_clips": pinned_clips,
                "locked_chains": locked_chains,
                "single_words": single_words,
                "unique_phrases": unique_phrases,
                "arena_bytes": arena_bytes,
                "arena_mb": round(arena_bytes / (1024 * 1024), 3),
                "max_arena_mb": self.config.max_arena_mb,
                "total_ingested": self.total_ingested,
                "total_dedup_pruned": self.total_dedup_pruned,
                "total_ring_evicted": self.total_ring_evicted,
                "arena_path": str(self.arena_path),
                "db_path": str(self.db_path),
            }

    def load_starter_pack_file(self, pack_path: Optional[Path] = None) -> int:
        path = pack_path or self.config.resolve_asset_path("assets/starter_pack.bin")
        if not path.exists():
            return 0

        with self._lock:
            cur = self._conn.cursor()
            existing_starter = int(cur.execute("SELECT COUNT(*) FROM clips WHERE priority = 0").fetchone()[0])
            if existing_starter > 0:
                return 0

        data = path.read_bytes()
        if len(data) < 16 or data[:8] not in (b"VOXPACK1", b"5MICPACK"):
            return 0

        _version, count = struct.unpack("<II", data[8:16])
        pos = 16
        loaded = 0
        lex = PhoneticLexicon.get_default()

        for _ in range(count):
            if pos + 2 > len(data):
                break
            (w_len,) = struct.unpack("<H", data[pos : pos + 2])
            pos += 2
            text = data[pos : pos + w_len].decode("utf-8", errors="ignore")
            pos += w_len

            if pos + 19 > len(data):
                break
            wc, sr, pitch, rms, centroid, data_len = struct.unpack("<BHfffI", data[pos : pos + 19])
            pos += 19
            audio_bytes = data[pos : pos + data_len]
            pos += data_len

            clip = SlicedClip(
                text=text,
                word_count=int(wc),
                start_s=0.0,
                end_s=len(audio_bytes) / float(max(1, sr)),
                encoded_bytes=audio_bytes,
                audio_format="ulaw_12k" if sr == 12000 else "ulaw_8k",
                sample_rate=int(sr),
                pitch_hz=round(float(pitch), 2),
                rms_energy=round(float(rms), 5),
                spectral_centroid=round(float(centroid), 2),
                spectral_flatness=0.12,
                seam_rms_ratio=0.04,
                is_locked_phrase=(int(wc) > 1),
                priority=0,
                speaker_tag="starter_pack",
                phonemes=" ".join(lex.phrase_phones(text)),
            )
            if self.insert_clip(clip, pinned=0) is not None:
                loaded += 1

        return loaded

    def close(self) -> None:
        with self._lock:
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None
            try:
                self._conn.close()
            except Exception:
                pass
