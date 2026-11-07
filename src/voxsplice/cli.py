"""
cli.py — Unified Command-Line Interface & Daemon Router for VoxSplice (`voxsplice`, `5mic`, `mimic`).

Provides full command-line access to every engine capability:
- `voxsplice`                       : Single-command interactive TUI + live capture + UDP bridge
- `voxsplice gui [--check]`         : Launch macOS-inspired Desktop Studio GUI
- `voxsplice pill`                  : Launch compact rounded pinnable Quick-Type overlay bar
- `voxsplice daemon`                : Run headless capture + UDP (9999) + HTTP API (18420) daemon
- `voxsplice say "<text>"`          : Stitch and speak sentence (supports `--output out.wav`, `--mode`, `--fx`)
- `voxsplice ingest <file.wav>`     : Transcribe, inspect seams, and slice a WAV recording into `clips.db`
- `voxsplice stats [--json]`        : Display SQLite & Arena storage telemetry
- `voxsplice clips`                 : List indexed clips in `clips.db` with pitch/speaker filters
- `voxsplice clear`                 : Clear unpinned live clips
- `voxsplice models`                : Browse, search Hugging Face Hub, or install STT/Embedding/Tokenizer models
- `voxsplice hardware [--json]`     : Inspect system CPU/RAM/GPU/OS and view intelligent model recommendations
- `voxsplice devices`               : List audio devices and virtual microphone routing status
- `voxsplice profiles`              : List or pin speaker profiles
- `voxsplice demo`                  : Play 5 auto-suggested sentences with 8-second intervals
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .arena_db import ArenaDatabase
from .audio_io import CrossPlatformAudioEngine
from .config import AUDIO_FORMAT_SPECS, FX_MODES, VOICE_MODES, VoxSpliceConfig
from .desktop_ui import launch_desktop_studio, start_local_api_background
from .hardware import inspect_system_hardware
from .model_hub import ModelHubManager
from .slicer import CoarticulationSlicer
from .stitcher import FrankensteinStitcher
from .stt_backends import MultiBackendSTTExtractor
from .tui_app import VoxSpliceTUI


class LocalUDPBridgeServer:
    """Zero-latency localhost UDP listener (`127.0.0.1:9999`) for game mods & CLI triggers."""

    def __init__(self, engine: CrossPlatformAudioEngine) -> None:
        self.engine = engine
        self.host = engine.config.udp_host
        self.port = int(engine.config.udp_port)
        self._sock: Optional[socket.socket] = None
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if self._running.is_set():
            return True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.settimeout(0.5)
            self._running.set()
            self._thread = threading.Thread(
                target=self._serve_loop, name="VoxSplice-UDPServer", daemon=True
            )
            self._thread.start()
            return True
        except Exception:
            return False

    def _serve_loop(self) -> None:
        while self._running.is_set() and self._sock is not None:
            try:
                data, addr = self._sock.recvfrom(8192)
            except socket.timeout:
                continue
            except Exception:
                break

            payload = data.decode("utf-8", errors="ignore").strip()
            if not payload:
                continue

            mode_override: Optional[str] = None
            fx_override: Optional[str] = None
            speaker_override: Optional[str] = None
            text = payload
            if payload.startswith("{"):
                try:
                    obj = json.loads(payload)
                    text = str(obj.get("text", "")).strip()
                    mode_override = obj.get("mode")
                    fx_override = obj.get("fx")
                    speaker_override = obj.get("speaker")
                except Exception:
                    pass

            if text:
                res = self.engine.say(
                    text,
                    voice_mode=mode_override,
                    fx_mode=fx_override,
                    speaker_filter=speaker_override,
                    play_audio=True,
                    blocking=False,
                )
                reply = json.dumps(
                    {
                        "ok": True,
                        "assembled_text": res.assembled_text,
                        "total_ms": res.total_ms,
                        "search_ms": res.search_ms,
                        "dsp_stitch_ms": res.dsp_stitch_ms,
                        "chunks": len(res.chunks),
                        "fx_mode": res.fx_mode,
                    }
                ).encode("utf-8")
                try:
                    self._sock.sendto(reply, addr)
                except Exception:
                    pass

    def stop(self) -> None:
        self._running.clear()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


def send_udp_say(
    host: str,
    port: int,
    text: str,
    mode: Optional[str] = None,
    fx: Optional[str] = None,
    speaker: Optional[str] = None,
    timeout_s: float = 0.45,
) -> Optional[Dict[str, Any]]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_s)
    payload = json.dumps({"text": text, "mode": mode, "fx": fx, "speaker": speaker}).encode("utf-8")
    try:
        sock.sendto(payload, (host, port))
        data, _ = sock.recvfrom(8192)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None
    finally:
        sock.close()


def build_engine(cfg: Optional[VoxSpliceConfig] = None, load_starter: bool = True) -> CrossPlatformAudioEngine:
    config = cfg or VoxSpliceConfig.load()
    db = ArenaDatabase(config)
    if load_starter and config.use_starter_pack_fallback:
        db.load_starter_pack_file()
    slicer = CoarticulationSlicer(config)
    stitcher = FrankensteinStitcher(db, config)
    stt = MultiBackendSTTExtractor(config)
    return CrossPlatformAudioEngine(config, db, slicer, stitcher, stt_extractor=stt)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxsplice",
        description="VoxSplice — Real-Time Concatenative Voice Reconstruction & Coarticulation Splicing Engine",
    )
    sub = parser.add_subparsers(dest="command")

    # tui
    p_tui = sub.add_parser("tui", help="Launch single-command interactive Terminal UI")
    p_tui.add_argument("--mode", choices=list(VOICE_MODES.keys()), default=None)
    p_tui.add_argument("--fx", choices=list(FX_MODES.keys()), default=None)

    # gui
    p_gui = sub.add_parser("gui", help="Launch macOS-inspired Desktop Studio GUI")
    p_gui.add_argument("--check", action="store_true", help="Verify local API & UI server startup and exit")

    # pill
    p_pill = sub.add_parser("pill", help="Launch compact rounded pinnable Quick-Type overlay bar")
    p_pill.add_argument("--check", action="store_true", help="Verify overlay server startup and exit")

    # daemon
    p_daemon = sub.add_parser("daemon", help="Run headless background capture, UDP (9999), and API (18420) daemon")
    p_daemon.add_argument("--mode", choices=list(VOICE_MODES.keys()), default=None)
    p_daemon.add_argument("--fx", choices=list(FX_MODES.keys()), default=None)
    p_daemon.add_argument("--format", choices=list(AUDIO_FORMAT_SPECS.keys()), default=None)
    p_daemon.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = infinite)")

    # say
    p_say = sub.add_parser("say", help="Stitch and speak a target sentence from recorded voice clips")
    p_say.add_argument("text", nargs="+", help="Target sentence to reconstruct and speak")
    p_say.add_argument("--mode", choices=list(VOICE_MODES.keys()), default=None)
    p_say.add_argument("--fx", choices=list(FX_MODES.keys()), default=None)
    p_say.add_argument("--speaker", type=str, default=None, help="Restrict clips to a specific speaker tag")
    p_say.add_argument("--output", "-o", type=str, default=None, help="Save stitched output to a .wav file")
    p_say.add_argument("--no-play", action="store_true", help="Do not transmit to audio device (useful with --output)")
    p_say.add_argument("--no-starter", action="store_true", help="Disable starter pack fallback (use live clips only)")
    p_say.add_argument("--json", action="store_true", help="Output detailed stitch telemetry as JSON")

    # ingest
    p_ing = sub.add_parser("ingest", help="Transcribe, analyze seams, and slice a WAV file into clips.db")
    p_ing.add_argument("wav_file", type=str, help="Path to input .wav audio recording")
    p_ing.add_argument("--speaker", type=str, default="lobby", help="Speaker tag to assign to extracted clips")
    p_ing.add_argument("--offset", type=float, default=0.0, help="Start offset in seconds")
    p_ing.add_argument("--duration", type=float, default=None, help="Max duration in seconds to ingest")
    p_ing.add_argument("--hint", type=str, default=None, help="Optional transcript hint for forced alignment")
    p_ing.add_argument("--json", action="store_true", help="Print JSON summary of extracted clips and seams")

    # stats
    p_stats = sub.add_parser("stats", help="Display live SQLite clips.db and Arena storage telemetry")
    p_stats.add_argument("--json", action="store_true", help="Output telemetry as JSON")

    # clips
    p_clips = sub.add_parser("clips", help="List indexed voice clips in clips.db")
    p_clips.add_argument("--starter", action="store_true", help="Include starter pack clips")
    p_clips.add_argument("--speaker", type=str, default=None, help="Filter by speaker tag")
    p_clips.add_argument("--pitch", choices=["all", "low", "high"], default="all", help="Filter by pitch band")
    p_clips.add_argument("--limit", type=int, default=40, help="Maximum clips to display")

    # clear
    sub.add_parser("clear", help="Clear all unpinned live clips from clips.db")

    # models
    p_mod = sub.add_parser("models", help="Explore, search, or download STT, Embedding, and Tokenizer models")
    p_mod.add_argument("action", nargs="?", choices=["list", "search", "install"], default="list")
    p_mod.add_argument("target", nargs="?", default=None, help="Search query or model_id to install")
    p_mod.add_argument("--category", choices=["stt", "embedding", "tokenizer", "vad"], default=None)
    p_mod.add_argument("--json", action="store_true")

    # Legacy alias `model`
    p_mod_alias = sub.add_parser("model", help="Alias for `models`: list or install a model")
    p_mod_alias.add_argument("model_id", nargs="?", default=None)

    # hardware
    p_hw = sub.add_parser("hardware", help="Inspect system hardware capabilities and view model recommendations")
    p_hw.add_argument("--json", action="store_true")

    # devices
    sub.add_parser("devices", help="List audio input/output devices and virtual mic routing status")

    # profiles
    p_prof = sub.add_parser("profiles", help="List or pin captured speaker profiles")
    p_prof.add_argument("--pin", type=str, default=None, help="Pin recent live clips under this profile name")

    # demo
    sub.add_parser("demo", help="Speak 5 auto-suggested sentences from live clips with 8s gaps")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args_list = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(args_list)
    cfg = VoxSpliceConfig.load()

    # Default (no subcommand) or `tui`
    if args.command in (None, "tui"):
        if getattr(args, "mode", None):
            cfg.voice_mode = args.mode
        if getattr(args, "fx", None):
            cfg.fx_mode = args.fx
        engine = build_engine(cfg)
        udp_srv = LocalUDPBridgeServer(engine)
        try:
            engine.setup_virtual_microphone()
            udp_srv.start()
            engine.start_background_capture()
            tui = VoxSpliceTUI(engine)
            return tui.run_interactive_loop()
        finally:
            engine.stop_background_capture()
            udp_srv.stop()
            engine.teardown_virtual_microphone()
            engine.db.close()

    if args.command in ("gui", "pill"):
        engine = build_engine(cfg)
        udp_srv = LocalUDPBridgeServer(engine)
        try:
            engine.setup_virtual_microphone()
            udp_srv.start()
            if not getattr(args, "check", False):
                engine.start_background_capture()
            return launch_desktop_studio(
                engine,
                overlay_only=(args.command == "pill"),
                headless_check=bool(getattr(args, "check", False)),
            )
        finally:
            engine.stop_background_capture()
            udp_srv.stop()
            engine.teardown_virtual_microphone()
            engine.db.close()

    if args.command == "daemon":
        if args.mode:
            cfg.voice_mode = args.mode
        if args.fx:
            cfg.fx_mode = args.fx
        if args.format:
            cfg.audio_format = args.format
        engine = build_engine(cfg)
        udp_srv = LocalUDPBridgeServer(engine)
        try:
            vmic = engine.setup_virtual_microphone()
            udp_srv.start()
            api_url = start_local_api_background(engine, port=cfg.api_port)
            engine.start_background_capture()
            print(f"[VoxSplice Daemon] Active | Virtual Mic: {vmic.device_name} | UDP: {cfg.udp_host}:{cfg.udp_port} | API: {api_url}")
            t0 = time.monotonic()
            while True:
                time.sleep(0.25)
                if args.duration > 0 and (time.monotonic() - t0) >= args.duration:
                    break
            return 0
        except KeyboardInterrupt:
            return 0
        finally:
            engine.stop_background_capture()
            udp_srv.stop()
            engine.teardown_virtual_microphone()
            engine.db.close()

    if args.command == "say":
        text = " ".join(args.text).strip()
        if args.no_starter:
            cfg.use_starter_pack_fallback = False

        # Use UDP fast-path only if no local output WAV or custom flag is requested
        if not args.output and not args.no_play and not args.no_starter and not args.json:
            reply = send_udp_say(cfg.udp_host, cfg.udp_port, text, mode=args.mode, fx=args.fx, speaker=args.speaker)
            if reply and reply.get("ok"):
                print(
                    f"[VoxSplice UDP] Stitched \"{reply['assembled_text']}\" "
                    f"({reply['chunks']} chunks) in {reply['total_ms']} ms"
                )
                return 0

        engine = build_engine(cfg, load_starter=(not args.no_starter))
        try:
            res = engine.say(
                text,
                voice_mode=args.mode,
                fx_mode=args.fx,
                play_audio=(not args.no_play),
                blocking=True,
                speaker_filter=args.speaker,
                save_wav_path=args.output,
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "request_text": res.request_text,
                            "assembled_text": res.assembled_text,
                            "voice_mode": res.voice_mode,
                            "fx_mode": res.fx_mode,
                            "search_ms": res.search_ms,
                            "dsp_stitch_ms": res.dsp_stitch_ms,
                            "total_ms": res.total_ms,
                            "output_wav": args.output,
                            "chunks": [c.to_dict() for c in res.chunks],
                        },
                        indent=2,
                    )
                )
            else:
                print(f"[VoxSplice] Request   : \"{res.request_text}\"")
                print(f"[VoxSplice] Assembled : \"{res.assembled_text}\" (Mode: {res.voice_mode}, FX: {res.fx_mode})")
                print(
                    f"[VoxSplice] Latency   : {res.total_ms:.2f} ms "
                    f"(Viterbi Search: {res.search_ms:.2f} ms + DSP Crossfade: {res.dsp_stitch_ms:.2f} ms)"
                )
                if args.output:
                    print(f"[VoxSplice] Saved WAV : {args.output}")
                for idx, ch in enumerate(res.chunks, start=1):
                    lock_tag = "LOCKED-PHRASE" if ch.is_locked_phrase else "SINGLE-WORD"
                    print(
                        f"   #{idx} \"{ch.matched_text}\" ({lock_tag}, {ch.tier}, sim={ch.similarity:.2f}, "
                        f"f0={ch.pitch_hz:.0f}Hz, gain={ch.applied_gain}x, {ch.duration_ms:.0f}ms)"
                    )
            return 0
        finally:
            engine.db.close()

    if args.command == "ingest":
        engine = build_engine(cfg, load_starter=False)
        try:
            wav_p = Path(args.wav_file)
            if not wav_p.exists():
                print(f"Error: WAV file not found: {wav_p}", file=sys.stderr)
                return 1
            clips, seams, words = engine.ingest_wav_file(
                wav_p,
                speaker_tag=args.speaker,
                max_duration_s=args.duration,
                start_offset_s=args.offset,
                known_text_hint=args.hint,
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "wav_file": str(wav_p),
                            "speaker": args.speaker,
                            "words_recognized": len(words),
                            "clips_extracted": len(clips),
                            "seams_analyzed": len(seams),
                            "clips": [c.text for c in clips],
                        },
                        indent=2,
                    )
                )
            else:
                print(f"[VoxSplice Ingest] File       : {wav_p}")
                print(f"[VoxSplice Ingest] Speaker    : {args.speaker}")
                print(f"[VoxSplice Ingest] Words STT  : {len(words)} words recognized")
                print(f"[VoxSplice Ingest] Clips Saved: {len(clips)} clips ({sum(1 for c in clips if c.is_locked_phrase)} locked chains, {sum(1 for c in clips if not c.is_locked_phrase)} single words)")
                for s in seams[:8]:
                    print(f"   Seam {s.left_word} -> {s.right_word}: {s.reason} | {s.coarticulation_note}")
            return 0
        finally:
            engine.db.close()

    if args.command == "stats":
        db = ArenaDatabase(cfg)
        try:
            if cfg.use_starter_pack_fallback:
                db.load_starter_pack_file()
            st = db.get_stats()
            if args.json:
                print(json.dumps(st, indent=2))
            else:
                print("=" * 64)
                print(" VoxSplice — Real-Time Voice Arena & SQLite Database Telemetry")
                print("=" * 64)
                print(f" Total Indexed Clips : {st['total_clips']} ({st['live_clips']} live, {st['starter_clips']} starter)")
                print(f" Single-Word Clips   : {st['single_words']} isolated words")
                print(f" Locked Bleed Chains : {st['locked_chains']} multi-word phrases")
                print(f" Pinned Victim Clips : {st['pinned_clips']} permanent clips")
                print(f" Distinct Phrases    : {st['unique_phrases']}")
                print(f" Arena SSD Usage     : {st['arena_mb']:.3f} MB / {st['max_arena_mb']} MB cap")
                print(f" Arena File Path     : {st['arena_path']}")
                print(f" SQLite WAL DB Path  : {st['db_path']}")
                print("=" * 64)
            return 0
        finally:
            db.close()

    if args.command == "clips":
        db = ArenaDatabase(cfg)
        try:
            clips = db.get_all_clips(include_starter=args.starter, speaker_tag=args.speaker)
            if args.pitch == "low":
                clips = [c for c in clips if c.pitch_hz <= cfg.low_pitch_cutoff_hz]
            elif args.pitch == "high":
                clips = [c for c in clips if c.pitch_hz >= cfg.high_pitch_cutoff_hz]
            print(f"Showing {min(len(clips), args.limit)} of {len(clips)} clips:")
            for c in clips[: args.limit]:
                status = "Locked Phrase" if c.is_locked_phrase else "Single Word  "
                print(
                    f"  #{c.id:<4d} \"{c.text:<26s}\" | words={c.word_count} | f0={c.pitch_hz:5.1f}Hz | "
                    f"conf={c.confidence:.2f} | spk={c.speaker_tag:<12s} | {status}"
                )
            return 0
        finally:
            db.close()

    if args.command == "clear":
        db = ArenaDatabase(cfg)
        try:
            removed = db.clear_live_clips()
            print(f"[VoxSplice] Cleared {removed} unpinned live clips from clips.db.")
            return 0
        finally:
            db.close()

    if args.command == "hardware":
        hw = inspect_system_hardware(cfg.resolve_models_dir())
        if args.json:
            print(json.dumps(hw.to_dict(), indent=2))
        else:
            print("=" * 68)
            print(" VoxSplice — System Hardware Profiler & Model Recommendation")
            print("=" * 68)
            print(f" OS / Display    : {hw.os_name} {hw.os_release} ({hw.arch}) | {hw.display_server}")
            print(f" Audio Subsystem : {hw.audio_subsystem}")
            print(f" CPU             : {hw.cpu_model} ({hw.logical_cores} threads, AVX2={hw.has_avx2})")
            print(f" Memory          : {hw.available_ram_mb} MB free / {hw.total_ram_mb} MB total ({hw.used_ram_percent}% used)")
            print(f" Swap            : {hw.swap_used_mb} MB used / {hw.swap_total_mb} MB total")
            print(f" GPU             : {hw.gpu_model}")
            print(f" Hardware Tier   : {hw.hardware_tier.upper()}")
            print("-" * 68)
            print(f" Recommended STT : {hw.recommended_stt_model}")
            print(f" Recommended Emb : {hw.recommended_embedding_model}")
            print(f" Recommended Tok : {hw.recommended_tokenizer}")
            print(f" Recommended Fmt : {hw.recommended_audio_format} (Arena Cap: {hw.recommended_arena_cap_mb} MB)")
            print("-" * 68)
            for r in hw.recommendation_rationale:
                print(f" • {r}")
            print("=" * 68)
        return 0

    if args.command in ("models", "model"):
        hub = ModelHubManager(cfg)
        action = getattr(args, "action", "list")
        target = getattr(args, "target", None)
        if args.command == "model":
            if getattr(args, "model_id", None):
                action = "install"
                target = args.model_id
            else:
                action = "list"

        if action == "search":
            res = hub.search_huggingface_hub(query=target or "whisper onnx")
            print(json.dumps(res, indent=2))
            return 0

        if action == "install" and target:
            engine = build_engine(cfg, load_starter=False)
            try:
                ok, msg = engine.stt.download_and_switch_model(
                    target, progress_cb=lambda s: print(f"  [ModelHub] {s}")
                )
                print(f"[{'OK' if ok else 'ERROR'}] {msg}")
                return 0 if ok else 1
            finally:
                engine.db.close()

        entries = hub.list_catalog(category=getattr(args, "category", None))
        if getattr(args, "json", False):
            print(json.dumps([e.to_dict() for e in entries], indent=2))
        else:
            print("=" * 78)
            print(" VoxSplice — Built-in Model & Tokenizer Browser")
            print("=" * 78)
            for e in entries:
                inst = "INSTALLED" if e.installed else "AVAILABLE"
                print(f" [{e.category.upper():<9s}] {e.model_id:<22s} | {inst:<9s} | Min RAM: {e.min_ram_mb:>3d} MB | {e.compatibility_status.upper()}")
                print(f"             {e.name}")
                print(f"             Summary  : {e.summary}")
                print(f"             Tradeoffs: {e.tradeoffs}")
                for v in e.variants:
                    print(f"             Variant  : {v.variant_id} ({v.label}, {v.size_mb} MB) -> {v.download_url}")
                print("-" * 78)
        return 0

    if args.command == "devices":
        engine = build_engine(cfg, load_starter=False)
        try:
            devs = engine.list_audio_devices()
            vmic = engine.setup_virtual_microphone()
            print("Virtual Microphone Status:", json.dumps(vmic.to_dict(), indent=2))
            print("Audio Inputs :", devs["inputs"])
            print("Audio Outputs:", devs["outputs"])
            return 0
        finally:
            engine.db.close()

    if args.command == "profiles":
        db = ArenaDatabase(cfg)
        try:
            if args.pin:
                cnt = db.pin_speaker_profile(args.pin)
                print(f"Pinned {cnt} clips to speaker profile '{args.pin}'.")
            profs = db.list_speaker_profiles()
            for p in profs:
                print(
                    f"  Speaker '{p['speaker_tag']}' | clips={p['clip_count']} | "
                    f"avg_f0={p['avg_pitch_hz']}Hz | centroid={p['avg_centroid_hz']}Hz | pinned={p['pinned_count']}"
                )
            return 0
        finally:
            db.close()

    if args.command == "demo":
        engine = build_engine(cfg)
        try:
            engine.setup_virtual_microphone()
            suggestions = engine.stitcher.suggest_playable_sentences(count=5)
            for idx, sent in enumerate(suggestions, start=1):
                print(f"\n=== [{idx}/5] Speaking: \"{sent}\" ===")
                res = engine.say(sent, play_audio=True, blocking=True)
                print(f"Assembled \"{res.assembled_text}\" ({len(res.chunks)} chunks) in {res.total_ms:.2f} ms")
                if idx < len(suggestions):
                    time.sleep(8.0)
            return 0
        finally:
            engine.db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
