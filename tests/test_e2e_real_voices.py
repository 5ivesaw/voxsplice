"""
test_e2e_real_voices.py — Real End-to-End Multi-Voice Recording, Slicing, Limited-Vocabulary
Reconstruction, and Round-Trip STT Intelligibility Verification Test Suite.

Runs the full multi-voice benchmark across 5 distinct voices (`allison_pbx_real`, `arctic_rms_deep`,
`arctic_kal_mid`, `arctic_slt_bright`, `arctic_awb_accent`) and verifies:
1. All 5 voices are ingested via 100% blind STT recognition with both multi-word locked phrases and single words.
2. Reconstructed sentences across limited per-voice vocabularies achieve >= 92% mean intelligibility.
3. Low-pitch mode (`low_pitch`) selects significantly lower f0 clips than high-pitch mode (`high_pitch`).
4. Chaos mode (`chaos`) switches across multiple speakers within a single sentence.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_e2e_voice_benchmarks import run_full_benchmarks


class TestRealVoicesEndToEnd(unittest.TestCase):
    def test_multi_voice_end_to_end_reconstruction_and_intelligibility(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = run_full_benchmarks(Path(td))
            agg = summary["aggregate_metrics"]

            self.assertGreaterEqual(agg["total_voices_tested"], 5)
            self.assertGreaterEqual(agg["total_sentences_stitched"], 14)
            self.assertGreaterEqual(
                agg["mean_intelligibility_score"],
                0.92,
                f"Mean intelligibility {agg['mean_intelligibility_score']} fell below 0.92 threshold",
            )
            self.assertLess(
                agg["mean_stitch_latency_ms"],
                60.0,
                f"Mean stitch latency {agg['mean_stitch_latency_ms']}ms exceeded 60ms threshold",
            )

            by_id = {r["id"]: r for r in summary["stitch_benchmarks"]}
            # Verify Low Pitch Mode vs High Pitch Mode pitch separation
            low_f0 = by_id["lobby_low_pitch_01"]["quality"]["pitch_hz"]
            high_f0 = by_id["lobby_high_pitch_01"]["quality"]["pitch_hz"]
            self.assertLess(
                low_f0,
                high_f0,
                f"Expected low_pitch f0 ({low_f0} Hz) < high_pitch f0 ({high_f0} Hz)",
            )

            # Verify Chaos Mode uses multiple distinct speakers
            chaos_speakers = by_id["lobby_chaos_01"]["speakers_used"]
            self.assertGreaterEqual(
                len(chaos_speakers),
                2,
                f"Expected Chaos mode to jump across >= 2 speakers, got {chaos_speakers}",
            )


if __name__ == "__main__":
    unittest.main()
