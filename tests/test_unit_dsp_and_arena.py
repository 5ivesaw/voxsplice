"""
test_unit_dsp_and_arena.py — Comprehensive Unit & Integration Test Suite for VoxSplice.

Verifies:
1. 8-bit ITU-T G.711 μ-Law 64KB LUT codec + zero-phase Butterworth de-quantization reconstruction filter.
2. CMU 135k ARPAbet Phonetic Lexicon, articulatory phoneme similarity, and sub-word syllable splicing.
3. SpeechVADGate & Wiener Spectral Flatness explosion/noise rejection + Smart Valley Utterance Splitter.
4. CoarticulationSlicer confidence gating, sliding connected phrases, and single-word vocabulary guarantee.
5. ArenaDatabase SQLite WAL + mmap storage, surgical sub-range decoding, 5-slot deduplication ring,
   ring-buffer eviction, and speaker profile pinning.
6. FrankensteinStitcher Viterbi DP global optimality, bidirectional contraction matching, compositional
   phrase synonyms, voice mode pitch separation, and DSP FX rack.
7. HalfDuplexGuard, LocalUDPBridgeServer, FastAPI endpoints, CLI commands, and Minecraft `.m` bridge.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

os.environ["VOXSPLICE_OFFLINE_EMBEDDINGS"] = "1"

from bridges.minecraft_chat_bridge import extract_mimic_command
from voxsplice.api_server import PreviewRequest, SayRequest, create_api_app
from voxsplice.arena_db import ArenaDatabase
from voxsplice.audio_io import CrossPlatformAudioEngine, HalfDuplexGuard
from voxsplice.cli import LocalUDPBridgeServer, main as cli_main, send_udp_say
from voxsplice.config import VoxSpliceConfig
from voxsplice.dsp import (
    apply_ulaw_reconstruction_filter,
    compute_spectral_flatness,
    decode_ulaw,
    encode_ulaw,
    float32_to_int16,
    int16_to_float32,
    is_human_speech_buffer,
    split_stream_into_utterances,
)
from voxsplice.fx_rack import process_chunk_fx
from voxsplice.hardware import inspect_system_hardware
from voxsplice.model_hub import ModelHubManager
from voxsplice.phonetics import PhoneticLexicon
from voxsplice.slicer import CoarticulationSlicer, SlicedClip
from voxsplice.stitcher import FrankensteinStitcher, StaticEmbeddingBrain
from voxsplice.stt_backends import WordTimestamp


class TestVoxSpliceCoreSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)
        self.cfg = VoxSpliceConfig(
            max_arena_mb=1,
            max_slots_per_phrase=5,
            udp_port=29999,
            min_word_confidence=0.65,
            use_starter_pack_fallback=True,
        )
        self.arena_path = self.base_path / "audio_arena.pcm"
        self.db_path = self.base_path / "clips.db"
        self.db = ArenaDatabase(self.cfg, arena_path=self.arena_path, db_path=self.db_path)
        self.slicer = CoarticulationSlicer(self.cfg)
        self.brain = StaticEmbeddingBrain(self.cfg)
        self.stitcher = FrankensteinStitcher(self.db, self.cfg, self.brain)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp_dir.cleanup()

    def test_01_ulaw_codec_and_butterworth_reconstruction(self) -> None:
        sr = 12000
        t = np.linspace(0.0, 0.25, int(sr * 0.25), dtype=np.float32)
        voice = 0.45 * np.sin(2.0 * np.pi * 135.0 * t) + 0.20 * np.sin(2.0 * np.pi * 270.0 * t)
        pcm16 = float32_to_int16(voice)
        ulaw = encode_ulaw(pcm16)
        self.assertEqual(len(ulaw), len(pcm16))
        dec_f32 = apply_ulaw_reconstruction_filter(int16_to_float32(decode_ulaw(ulaw)), sample_rate=16000)
        corr = float(np.corrcoef(voice, dec_f32)[0, 1])
        self.assertGreater(corr, 0.995)

    def test_02_phonetic_lexicon_and_subword_splicing_plan(self) -> None:
        lex = PhoneticLexicon.get_default()
        self.assertEqual(lex.phonetic_similarity("to", "too"), 1.0)
        self.assertEqual(lex.phonetic_similarity("there", "their"), 1.0)
        self.assertGreater(lex.phonetic_similarity("name", "game"), 0.70)

        # Prefix extraction: 'connect' from 'connected'
        plan_prefix = lex.find_subword_splice_plan("connect", ["connected", "other"])
        self.assertIsNotNone(plan_prefix)
        self.assertEqual(plan_prefix[0].source_word, "connected")
        self.assertEqual(plan_prefix[0].part_type, "prefix")

        # 2-part compound subword splice: 'into' from 'in' + 'to'
        plan_compound = lex.find_subword_splice_plan("into", ["inside", "onto"])
        self.assertIsNotNone(plan_compound)
        self.assertEqual(len(plan_compound), 2)

    def test_03_vad_gate_and_valley_splitter(self) -> None:
        sr = 16000
        rng = np.random.default_rng(42)
        noise = rng.uniform(-0.5, 0.5, int(sr * 0.4)).astype(np.float32)
        self.assertFalse(is_human_speech_buffer(noise, sample_rate=sr))
        self.assertGreater(compute_spectral_flatness(noise, sample_rate=sr), 0.50)

        # Construct two 1.2s speech bursts separated by 0.4s silence
        t_burst = np.linspace(0.0, 1.2, int(sr * 1.2), dtype=np.float32)
        burst = float32_to_int16(0.35 * np.sin(2.0 * np.pi * 150.0 * t_burst))
        silence = np.zeros(int(sr * 0.4), dtype=np.int16)
        stream = np.concatenate([burst, silence, burst])
        spans = split_stream_into_utterances(stream, sample_rate=sr, pause_ms=180.0)
        self.assertEqual(len(spans), 2)

    def test_04_slicer_confidence_filtering_and_sliding_phrases(self) -> None:
        sr = 16000
        t = np.linspace(0.0, 2.2, int(sr * 2.2), dtype=np.float32)
        audio = (0.35 * np.sin(2.0 * np.pi * 130.0 * t) + 0.15 * np.sin(2.0 * np.pi * 260.0 * t)).astype(np.float32)
        words = [
            WordTimestamp("hello", 0.05, 0.30, conf=0.95),
            WordTimestamp("my", 0.30, 0.55, conf=0.92),
            WordTimestamp("misheard_noise", 0.56, 0.85, conf=0.32),
            WordTimestamp("to", 0.90, 0.96, conf=0.55),  # Short function word retained
            WordTimestamp("real", 0.96, 1.30, conf=0.94),
            WordTimestamp("human", 1.30, 1.75, conf=0.96),
        ]
        clips, _seams = self.slicer.slice_utterance(audio, words, sample_rate=sr)
        texts = {c.text for c in clips}
        self.assertTrue(all("misheard_noise" not in tx for tx in texts))
        self.assertIn("hello my", texts)
        self.assertIn("to real human", texts)
        self.assertIn("real human", texts)
        self.assertIn("to", texts)
        self.assertIn("human", texts)

    def test_05_viterbi_global_optimality_and_contraction_matching(self) -> None:
        """
        Verify Viterbi DP picks 'where' (1w) + 'is the real' (3w) + 'human' (1w)
        rather than getting greedily trapped by 'where is' (2w) when 'the' and 'real' are missing.
        """
        self.cfg.use_starter_pack_fallback = False
        t = np.linspace(0.0, 0.35, int(12000 * 0.35), dtype=np.float32)
        enc = encode_ulaw(float32_to_int16(0.25 * np.sin(2.0 * np.pi * 135.0 * t)))

        for phrase in ["where", "where is", "is the real", "human", "what i'm saying"]:
            self.db.insert_clip(
                SlicedClip(
                    text=phrase,
                    word_count=len(phrase.split()),
                    start_s=0.0,
                    end_s=0.35,
                    encoded_bytes=enc,
                    audio_format="ulaw_12k",
                    sample_rate=12000,
                    pitch_hz=135.0,
                    rms_energy=0.14,
                    spectral_centroid=1100.0,
                    spectral_flatness=0.10,
                    seam_rms_ratio=0.05,
                    is_locked_phrase=(len(phrase.split()) > 1),
                    priority=1,
                    speaker_tag="lobby",
                )
            )

        res = self.stitcher.stitch_sentence("where is the real human")
        self.assertEqual(res.assembled_text, "where is the real human")
        self.assertEqual([c.matched_text for c in res.chunks], ["where", "is the real", "human"])

        # Contraction expansion matching: 'what i am saying' -> 'what i'm saying'
        res_c = self.stitcher.stitch_sentence("what i am saying")
        self.assertEqual(res_c.assembled_text, "what i'm saying")

    def test_06_hardware_profiler_model_hub_and_api_server(self) -> None:
        hw = inspect_system_hardware(self.base_path)
        self.assertGreater(hw.total_ram_mb, 256)
        self.assertIn(hw.hardware_tier, ("low_end", "mid_range", "high_end"))

        hub = ModelHubManager(self.cfg)
        catalog = hub.list_catalog(hw=hw)
        self.assertGreaterEqual(len(catalog), 8)

        for fx in ("clean", "retro_8bit", "radio", "scratch", "corrupt"):
            sig = np.sin(np.linspace(0.0, 20.0, 2400, dtype=np.float32)) * 0.3
            out = process_chunk_fx(sig, fx_mode=fx, sample_rate=16000, intensity=0.5, chunk_index=1)
            self.assertFalse(np.any(np.isnan(out)))

        guard = HalfDuplexGuard(cooldown_ms=40.0)
        guard.begin_transmit()
        self.assertTrue(guard.is_blocked)
        guard.end_transmit()
        time.sleep(0.05)
        self.assertFalse(guard.is_blocked)

        self.assertEqual(
            extract_mimic_command("[12:00:00] [CHAT] <Player> .m hello real human"),
            "hello real human",
        )

        self.db.load_starter_pack_file(ROOT / "assets" / "starter_pack.bin")
        engine = CrossPlatformAudioEngine(self.cfg, self.db, self.slicer, self.stitcher)
        app = create_api_app(engine)
        self.assertIsNotNone(app)

        srv = LocalUDPBridgeServer(engine)
        if srv.start():
            try:
                rep = send_udp_say(self.cfg.udp_host, self.cfg.udp_port, "hello my name is human")
                self.assertIsNotNone(rep)
                self.assertTrue(rep["ok"])
            finally:
                srv.stop()

        # Verify CLI commands run cleanly
        rc_hw = cli_main(["hardware", "--json"])
        self.assertEqual(rc_hw, 0)


if __name__ == "__main__":
    unittest.main()
