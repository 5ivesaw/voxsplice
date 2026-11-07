"""
run_e2e_voice_benchmarks.py — Real End-to-End Multi-Voice Recording, Slicing, Limited-Vocabulary
Sentence Reconstruction, Acoustic Seam Inspection, and Round-Trip STT Intelligibility Benchmark.

Executes REAL tests on this machine across 5 distinct voices using 100% blind, unassisted
PocketSphinx C continuous speech recognition (`known_hint = None`):
1. `allison_pbx_real` (Female Human Voice, 73.35s real recording `/usr/share/doc/python3-aiortc/examples/server/demo-instruct.wav`)
2. `arctic_rms_deep`  (US Deep Male Voice, f0 ~ 104 Hz, speaking gaming & proximity lobby dialogue)
3. `arctic_kal_mid`   (US Mid Male Voice, f0 ~ 131 Hz, speaking tactical team callouts)
4. `arctic_slt_bright`(US Female Voice, f0 ~ 186 Hz, speaking survival & exploration dialogue)
5. `arctic_awb_accent`(Scottish Accented Male Voice, f0 ~ 149 Hz, speaking casual lobby banter)
"""

from __future__ import annotations

import ctypes
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from voxsplice.arena_db import ArenaDatabase
from voxsplice.audio_io import CrossPlatformAudioEngine
from voxsplice.config import VoxSpliceConfig
from voxsplice.dsp import compute_rms, estimate_pitch_and_centroid
from voxsplice.hardware import inspect_system_hardware
from voxsplice.phonetics import PhoneticLexicon
from voxsplice.slicer import CoarticulationSlicer
from voxsplice.stitcher import FrankensteinStitcher
from voxsplice.stt_backends import PocketSphinxCBackend


def generate_multi_voice_corpus(corpus_dir: Path) -> Dict[str, Dict[str, Any]]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    flite = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libflite.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libflite_usenglish.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libflite_cmulex.so.1", mode=ctypes.RTLD_GLOBAL)
    flite.flite_init()
    flite.flite_text_to_speech.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p]
    flite.flite_text_to_speech.restype = ctypes.c_float

    voice_specs = {
        "arctic_rms_deep": (
            "/usr/lib/x86_64-linux-gnu/libflite_cmu_us_rms.so.1",
            "register_cmu_us_rms",
            (
                "Hello my name is the real human player standing right behind you in the room. "
                "Drop the weapon and walk away from that door right now. "
                "We need to stay together if we want to survive this monster in the dark."
            ),
        ),
        "arctic_kal_mid": (
            "/usr/lib/x86_64-linux-gnu/libflite_cmu_us_kal16.so.1",
            "register_cmu_us_kal16",
            (
                "Hey guys look over there under the outside window. "
                "Someone is trying to open the door without making a sound. "
                "Watch the back and hold your position until our team arrives."
            ),
        ),
        "arctic_slt_bright": (
            "/usr/lib/x86_64-linux-gnu/libflite_cmu_us_slt.so.1",
            "register_cmu_us_slt",
            (
                "I heard a strange noise inside the other building. "
                "Please help me find the key before the lights go out again. "
                "Are you sure that person is really one of our friends in this game."
            ),
        ),
        "arctic_awb_accent": (
            "/usr/lib/x86_64-linux-gnu/libflite_cmu_us_awb.so.1",
            "register_cmu_us_awb",
            (
                "Where did everybody go after we left the first room. "
                "Stop running around and listen to what I am saying to you. "
                "There is no way I am going back down there alone."
            ),
        ),
    }

    manifest: Dict[str, Dict[str, Any]] = {}

    real_pbx = Path("/usr/share/doc/python3-aiortc/examples/server/demo-instruct.wav")
    if real_pbx.exists():
        manifest["allison_pbx_real"] = {
            "wav_path": str(real_pbx),
            "description": "Real Human Recording (Allison Smith Asterisk PBX, 8kHz mono, 73.35s)",
            "source_transcript": (
                "If you like to learn more technical information about that period... "
                "in order to work you will need to be connected to the internet... "
                "when you have connected to a remote location configuration..."
            ),
            "slice_start_s": 0.0,
            "slice_duration_s": 42.0,
            "known_hint": None,
        }

    for tag, (lib_path, reg_name, paragraph) in voice_specs.items():
        vlib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        reg_fn = getattr(vlib, reg_name)
        reg_fn.restype = ctypes.c_void_p
        v_ptr = reg_fn(None)
        wav_out = corpus_dir / f"{tag}.wav"
        dur_s = float(
            flite.flite_text_to_speech(
                paragraph.encode("utf-8"), v_ptr, str(wav_out).encode("utf-8")
            )
        )
        manifest[tag] = {
            "wav_path": str(wav_out),
            "description": f"CMU Arctic Voice ({tag}, 16kHz mono, {dur_s:.2f}s)",
            "source_transcript": paragraph,
            "slice_start_s": 0.0,
            "slice_duration_s": dur_s,
            "known_hint": None,  # 100% blind, unassisted STT recognition!
        }

    return manifest


def evaluate_audio_quality_and_intelligibility(
    ps: PocketSphinxCBackend,
    lexicon: PhoneticLexicon,
    samples_f32: np.ndarray,
    samples_i16: np.ndarray,
    sample_rate: int,
    expected_assembled_text: str,
) -> Dict[str, Any]:
    if len(samples_f32) < 64:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "pitch_hz": 0.0,
            "centroid_hz": 0.0,
            "seam_click_ratio": 0.0,
            "roundtrip_transcript": "",
            "phonetic_intelligibility": 0.0,
            "word_overlap_recall": 0.0,
            "intelligibility_score": 0.0,
        }

    rms = compute_rms(samples_f32)
    peak = float(np.max(np.abs(samples_f32)))
    pitch_hz, centroid_hz = estimate_pitch_and_centroid(samples_f32, sample_rate=sample_rate)

    diffs = np.abs(np.diff(samples_f32))
    p999_diff = float(np.percentile(diffs, 99.9))
    median_diff = max(1e-6, float(np.percentile(diffs, 90.0)))
    click_ratio = round(p999_diff / median_diff, 3)

    rt_open_words = [
        w.word
        for w in ps.transcribe_pcm16(samples_i16, sample_rate=sample_rate, split_utterances=False)
    ]
    rt_open_text = " ".join(rt_open_words)

    exp_tokens = [w.replace("'", "") for w in expected_assembled_text.lower().split() if w]
    open_clean = [w.replace("'", "") for w in rt_open_words]
    open_set = set(open_clean)
    matched_tokens = sum(
        1
        for tok in exp_tokens
        if tok in open_set or any(lexicon.phonetic_similarity(tok, rw) >= 0.72 for rw in open_clean)
    )
    word_recall = round(matched_tokens / float(max(1, len(exp_tokens))), 4)
    phon_sim = lexicon.phonetic_similarity(expected_assembled_text, rt_open_text)
    combined_intel = round(max(word_recall, phon_sim), 4)

    return {
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "pitch_hz": pitch_hz,
        "centroid_hz": centroid_hz,
        "seam_click_ratio": click_ratio,
        "roundtrip_transcript": rt_open_text,
        "phonetic_intelligibility": phon_sim,
        "word_overlap_recall": word_recall,
        "intelligibility_score": combined_intel,
    }


def run_full_benchmarks(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = output_dir / "voice_corpus"
    stitched_dir = output_dir / "stitched_samples"
    stitched_dir.mkdir(parents=True, exist_ok=True)

    t_start_all = time.perf_counter()
    hw = inspect_system_hardware(output_dir)
    ps = PocketSphinxCBackend()
    lexicon = PhoneticLexicon.get_default()

    corpus = generate_multi_voice_corpus(corpus_dir)

    cfg = VoxSpliceConfig(
        use_starter_pack_fallback=False,
        min_word_confidence=0.65,
        enable_subword_splicing=True,
        audio_format="ulaw_12k",
    )
    arena_path = output_dir / "benchmark_arena.pcm"
    db_path = output_dir / "benchmark_clips.db"
    arena_path.unlink(missing_ok=True)
    db_path.unlink(missing_ok=True)

    db = ArenaDatabase(cfg, arena_path=arena_path, db_path=db_path)
    slicer = CoarticulationSlicer(cfg)
    stitcher = FrankensteinStitcher(db, cfg)
    engine = CrossPlatformAudioEngine(cfg, db, slicer, stitcher)

    ingestion_report: Dict[str, Any] = {}
    for speaker_tag, meta in corpus.items():
        t0 = time.perf_counter()
        clips, seams, words = engine.ingest_wav_file(
            meta["wav_path"],
            speaker_tag=speaker_tag,
            max_duration_s=meta["slice_duration_s"],
            start_offset_s=meta["slice_start_s"],
            known_text_hint=meta["known_hint"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ingestion_report[speaker_tag] = {
            "description": meta["description"],
            "words_recognized": len(words),
            "clips_extracted": len(clips),
            "locked_phrases": sum(1 for c in clips if c.is_locked_phrase),
            "single_words": sum(1 for c in clips if not c.is_locked_phrase),
            "seams_inspected": len(seams),
            "ingest_time_ms": round(elapsed_ms, 2),
            "sample_vocabulary": sorted(set(c.text for c in clips))[:25],
        }

    test_cases = [
        {
            "id": "real_pbx_01",
            "speaker_filter": "allison_pbx_real",
            "voice_mode": "natural",
            "target_text": "you need to be connected to the internet in order to work",
            "scenario": "Real Human Voice (8kHz PBX): Recombining 2 separate clauses into 1 sentence",
        },
        {
            "id": "real_pbx_02",
            "speaker_filter": "allison_pbx_real",
            "voice_mode": "natural",
            "target_text": "you like to learn more technical information about",
            "scenario": "Real Human Voice (8kHz PBX): Connected coarticulated technical phrase reconstruction",
        },
        {
            "id": "real_pbx_03",
            "speaker_filter": "allison_pbx_real",
            "voice_mode": "natural",
            "target_text": "you need to connect to a distant place to function",
            "scenario": "Real Human Voice (8kHz PBX): Limited-vocabulary synonym & compositional phrase translation",
        },
        {
            "id": "rms_deep_01",
            "speaker_filter": "arctic_rms_deep",
            "voice_mode": "natural",
            "target_text": "hello my name is the real human standing right behind you",
            "scenario": "Deep Male Voice (RMS): Reconstructing proximity chat greeting from segmented phrases",
        },
        {
            "id": "rms_deep_02",
            "speaker_filter": "arctic_rms_deep",
            "voice_mode": "natural",
            "target_text": "we need to stay together in the room right now",
            "scenario": "Deep Male Voice (RMS): Novel tactical sentence spliced across 3 different sentences",
        },
        {
            "id": "rms_deep_03",
            "speaker_filter": "arctic_rms_deep",
            "voice_mode": "natural",
            "target_text": "move away from that monster in the dark",
            "scenario": "Deep Male Voice (RMS): Synonym substitution ('move'->'away'/'stay') + phrase splice",
        },
        {
            "id": "kal_mid_01",
            "speaker_filter": "arctic_kal_mid",
            "voice_mode": "natural",
            "target_text": "guys look over there someone is trying to open",
            "scenario": "Mid Male Voice (KAL16): Splicing callout header with middle clause",
        },
        {
            "id": "kal_mid_02",
            "speaker_filter": "arctic_kal_mid",
            "voice_mode": "natural",
            "target_text": "watch the back until our team arrives without making",
            "scenario": "Mid Male Voice (KAL16): Recombining clauses cleanly at zero-crossing seams",
        },
        {
            "id": "slt_bright_01",
            "speaker_filter": "arctic_slt_bright",
            "voice_mode": "natural",
            "target_text": "please help me find the lights inside the other building",
            "scenario": "Female Voice (SLT): Cross-sentence splicing across 2 distinct utterances",
        },
        {
            "id": "slt_bright_02",
            "speaker_filter": "arctic_slt_bright",
            "voice_mode": "natural",
            "target_text": "are you that person is really one of our friends in this game",
            "scenario": "Female Voice (SLT): Coarticulated question reconstruction",
        },
        {
            "id": "awb_accent_01",
            "speaker_filter": "arctic_awb_accent",
            "voice_mode": "natural",
            "target_text": "stop running around and listen to what i am saying to you",
            "scenario": "Scottish Male Voice (AWB): Connected phrase and contraction ('i'm') matching",
        },
        {
            "id": "lobby_chaos_01",
            "speaker_filter": None,
            "voice_mode": "chaos",
            "target_text": "hello my name is the real human behind you right now",
            "scenario": "Multi-Voice Frankenstein Chaos Mode: Jumps between speakers while preserving exact words",
        },
        {
            "id": "lobby_low_pitch_01",
            "speaker_filter": None,
            "voice_mode": "low_pitch",
            "target_text": "we need to stay together and listen to what i am saying",
            "scenario": "Multi-Voice Low Pitch Mode (< 130 Hz): Automatically selects deep-voiced clips",
        },
        {
            "id": "lobby_high_pitch_01",
            "speaker_filter": None,
            "voice_mode": "high_pitch",
            "target_text": "please help me find the real human in this game",
            "scenario": "Multi-Voice High Pitch Mode (> 180 Hz): Automatically selects high-pitched clips",
        },
    ]

    results_list: List[Dict[str, Any]] = []
    for tc in test_cases:
        out_wav = stitched_dir / f"{tc['id']}.wav"
        res = engine.say(
            tc["target_text"],
            voice_mode=tc["voice_mode"],
            fx_mode="clean",
            play_audio=False,
            speaker_filter=tc["speaker_filter"],
            save_wav_path=out_wav,
        )
        metrics = evaluate_audio_quality_and_intelligibility(
            ps=ps,
            lexicon=lexicon,
            samples_f32=res.samples_f32,
            samples_i16=res.samples_int16,
            sample_rate=res.sample_rate,
            expected_assembled_text=res.assembled_text,
        )
        speakers_used = sorted(set(c.speaker_tag for c in res.chunks))
        results_list.append(
            {
                "id": tc["id"],
                "scenario": tc["scenario"],
                "speaker_filter": tc["speaker_filter"] or "all_lobby_voices",
                "voice_mode": tc["voice_mode"],
                "target_text": tc["target_text"],
                "assembled_text": res.assembled_text,
                "chunk_count": len(res.chunks),
                "speakers_used": speakers_used,
                "search_ms": res.search_ms,
                "dsp_stitch_ms": res.dsp_stitch_ms,
                "total_ms": res.total_ms,
                "duration_s": round(len(res.samples_f32) / float(max(1, res.sample_rate)), 2),
                "wav_file": str(out_wav),
                "chunks": [c.to_dict() for c in res.chunks],
                "quality": metrics,
            }
        )

    db_stats = db.get_stats()
    sample_seams = [s.to_dict() for s in engine.last_seams[:12]]
    sample_clips = [c.to_dict() for c in db.get_all_clips(include_starter=False)[:40]]
    db.close()

    avg_latency_ms = round(float(np.mean([r["total_ms"] for r in results_list])), 2)
    avg_intel = round(
        float(np.mean([r["quality"]["intelligibility_score"] for r in results_list])), 4
    )

    summary = {
        "timestamp": time.time(),
        "hardware": hw.to_dict(),
        "arena_stats": db_stats,
        "ingestion_by_speaker": ingestion_report,
        "stitch_benchmarks": results_list,
        "aggregate_metrics": {
            "total_voices_tested": len(corpus),
            "total_sentences_stitched": len(results_list),
            "mean_stitch_latency_ms": avg_latency_ms,
            "mean_intelligibility_score": avg_intel,
            "total_benchmark_wall_s": round(time.perf_counter() - t_start_all, 2),
        },
        "sample_seams": sample_seams,
        "sample_clips": sample_clips,
    }

    report_path = output_dir / "e2e_benchmark_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    out_dir = Path(tempfile.gettempdir()) / "voxsplice_e2e_benchmarks"
    rep = run_full_benchmarks(out_dir)
    print("=" * 78)
    print(" VoxSplice — Real Multi-Voice End-to-End Reconstruction Benchmark Summary")
    print("=" * 78)
    for spk, info in rep["ingestion_by_speaker"].items():
        print(
            f" [Voice: {spk:<18s}] Words={info['words_recognized']:<3d} | "
            f"Clips={info['clips_extracted']:<3d} (Phrases={info['locked_phrases']}, Singles={info['single_words']}) | "
            f"Ingest={info['ingest_time_ms']:.1f}ms"
        )
    print("-" * 78)
    for r in rep["stitch_benchmarks"]:
        q = r["quality"]
        print(
            f" [{r['id']:<18s}] Mode={r['voice_mode']:<10s} | {r['total_ms']:5.2f}ms | "
            f"f0={q['pitch_hz']:5.1f}Hz | Intelligibility={q['intelligibility_score']*100:5.1f}%"
        )
        print(f"    Target    : \"{r['target_text']}\"")
        print(f"    Assembled : \"{r['assembled_text']}\" ({r['chunk_count']} chunks, speakers={r['speakers_used']})")
        print(f"    RoundTrip : \"{q['roundtrip_transcript']}\"")
    print("=" * 78)
    print(" Aggregate:", json.dumps(rep["aggregate_metrics"], indent=2))
