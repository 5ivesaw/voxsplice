"""
tui_app.py — Unified Single-Command Interactive Terminal Dashboard for VoxSplice (`voxsplice` / `5mic`).

Provides a flicker-free terminal command center combining:
- Live desktop audio capture & Recording ON/OFF toggle (`/rec`)
- Self-Monitor toggle (`/hear`)
- Voice Mode (`/mode`) & DSP FX (`/fx`) cyclers
- Confidence threshold tuner (`/conf <0.0-1.0>`)
- Built-in Model & Tokenizer Browser (`/models`, `/model <id>`)
- Hardware profiler (`/hw`)
- 5 Auto-Suggested Playable Sentences (`/1`..`/5`)
- 1-Key Database Reset (`/clear`) & Multi-Phrase Scramble Demo (`/demo`)
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import List, Optional, Tuple

from .audio_io import CrossPlatformAudioEngine
from .config import FX_MODES, VOICE_MODES
from .hardware import inspect_system_hardware
from .model_hub import ModelHubManager
from .slicer import SlicedClip
from .stitcher import StitchResult


class VoxSpliceTUI:
    def __init__(self, engine: CrossPlatformAudioEngine) -> None:
        self.engine = engine
        self.config = engine.config
        self.hub = ModelHubManager(self.config)
        self._running = threading.Event()
        self._recent_captured: List[Tuple[str, float, float, bool]] = []
        self._last_stitch: Optional[StitchResult] = None
        self._suggestions: List[str] = []
        self._status_banner: str = "Ready — Type any sentence to stitch & speak, or /help for commands."
        self.engine.add_on_clip_callback(self._on_clip_ingested)
        self.refresh_suggestions()

    def _on_clip_ingested(self, clip: SlicedClip) -> None:
        self._recent_captured.insert(
            0, (clip.text, clip.pitch_hz, clip.confidence, clip.is_locked_phrase)
        )
        if len(self._recent_captured) > 15:
            self._recent_captured = self._recent_captured[:15]

    def refresh_suggestions(self) -> List[str]:
        self._suggestions = self.engine.stitcher.suggest_playable_sentences(count=5)
        return self._suggestions

    def execute_command_or_say(self, raw_line: str) -> bool:
        line = raw_line.strip()
        if not line:
            return True

        low = line.lower()
        if low in ("/q", "/quit", "/exit"):
            return False

        if low == "/rec":
            active = self.engine.toggle_recording()
            self._status_banner = (
                "Recording RESUMED — Listening to desktop audio."
                if active
                else "Recording PAUSED — Capture muted."
            )
            return True

        if low == "/hear":
            mon = self.engine.toggle_monitor_self()
            self.config.save()
            self._status_banner = (
                "Self-Monitor ON: Stitched audio plays to Headphones AND MimicMic."
                if mon
                else "Self-Monitor OFF: Stitched audio plays ONLY to MimicMic."
            )
            return True

        if low.startswith("/mode"):
            parts = low.split()
            modes = list(VOICE_MODES.keys())
            if len(parts) >= 2 and parts[1] in VOICE_MODES:
                self.config.voice_mode = parts[1]
            else:
                cur_idx = modes.index(self.config.voice_mode) if self.config.voice_mode in modes else 0
                self.config.voice_mode = modes[(cur_idx + 1) % len(modes)]
            self.config.save()
            self._status_banner = f"Voice Mode -> {self.config.voice_mode.upper()} ({VOICE_MODES[self.config.voice_mode]['label']})"
            return True

        if low.startswith("/fx"):
            parts = low.split()
            fxs = list(FX_MODES.keys())
            if len(parts) >= 2 and parts[1] in FX_MODES:
                self.config.fx_mode = parts[1]
            else:
                cur_idx = fxs.index(self.config.fx_mode) if self.config.fx_mode in fxs else 0
                self.config.fx_mode = fxs[(cur_idx + 1) % len(fxs)]
            self.config.save()
            self._status_banner = f"DSP FX Mode -> {self.config.fx_mode.upper()} ({FX_MODES[self.config.fx_mode]['label']})"
            return True

        if low.startswith("/conf"):
            parts = low.split()
            if len(parts) >= 2:
                try:
                    val = max(0.10, min(0.98, float(parts[1])))
                    self.config.min_word_confidence = round(val, 2)
                    self.config.save()
                    self._status_banner = f"Minimum Word Confidence set to {self.config.min_word_confidence:.2f}"
                except ValueError:
                    self._status_banner = "Usage: /conf 0.65 (value between 0.10 and 0.98)"
            else:
                self._status_banner = f"Current Min Word Confidence: {self.config.min_word_confidence:.2f}"
            return True

        if low in ("/models", "/model"):
            entries = self.hub.list_catalog()
            summary = ", ".join(
                f"{e.model_id}{'[INSTALLED]' if e.installed else ''}" for e in entries
            )
            self._status_banner = f"Catalog: {summary} | Use `/model <id>` to download/activate"
            return True

        if low.startswith("/model "):
            target_m = line.split(None, 1)[1].strip()
            self._status_banner = f"Downloading/activating model '{target_m}'..."
            ok, msg = self.engine.stt.download_and_switch_model(target_m)
            self._status_banner = msg
            return True

        if low == "/hw":
            hw = inspect_system_hardware(self.config.resolve_models_dir())
            self._status_banner = (
                f"HW: {hw.cpu_model} ({hw.logical_cores}T) | Free RAM: {hw.available_ram_mb}/{hw.total_ram_mb} MB | "
                f"Tier: {hw.hardware_tier.upper()} | Rec STT: {hw.recommended_stt_model}"
            )
            return True

        if low == "/clear":
            removed = self.engine.db.clear_live_clips()
            self._recent_captured.clear()
            self.refresh_suggestions()
            self._status_banner = f"Cleared {removed} unpinned live clips from clips.db."
            return True

        if low in ("/1", "/2", "/3", "/4", "/5"):
            idx = int(low[1]) - 1
            self.refresh_suggestions()
            if 0 <= idx < len(self._suggestions):
                target = self._suggestions[idx]
                res = self.engine.say(target, play_audio=True, blocking=False)
                self._last_stitch = res
                self._status_banner = f"Spoke #{idx+1}: \"{res.assembled_text}\" ({len(res.chunks)} chunks, {res.total_ms:.1f}ms)"
            return True

        if low == "/demo":
            self.refresh_suggestions()
            threading.Thread(target=self._run_demo_sequence, daemon=True).start()
            return True

        res = self.engine.say(line, play_audio=True, blocking=False)
        self._last_stitch = res
        self._status_banner = (
            f"Stitched & Spoke: \"{res.assembled_text}\" "
            f"({len(res.chunks)} chunks in {res.total_ms:.1f}ms)"
        )
        return True

    def _run_demo_sequence(self) -> None:
        sentences = list(self._suggestions)[:5]
        for idx, sent in enumerate(sentences, start=1):
            self._status_banner = f"[DEMO {idx}/{len(sentences)}] Speaking: \"{sent}\"..."
            res = self.engine.say(sent, play_audio=True, blocking=True)
            self._last_stitch = res
            if idx < len(sentences):
                for rem in range(8, 0, -1):
                    self._status_banner = f"[DEMO {idx}/{len(sentences)}] Waiting {rem}s before next stitch..."
                    time.sleep(1.0)
        self._status_banner = "Demo sequence complete."

    def render_dashboard_text(self) -> str:
        cols = min(104, max(76, shutil.get_terminal_size((92, 28)).columns - 2))
        sep = "=" * cols
        thin = "-" * cols

        stats = self.engine.db.get_stats()
        rec_badge = "● REC [ACTIVE]" if self.engine.is_recording_active else "⏸ REC [PAUSED]"
        mon_badge = "ON (Headphones + MimicMic)" if self.config.monitor_self else "OFF (MimicMic Only)"

        lines = [
            sep,
            f" VoxSplice v2.4 | {rec_badge} | Self-Monitor: {mon_badge} | Mode: {self.config.voice_mode.upper()} | FX: {self.config.fx_mode.upper()}",
            f" STT Engine: {self.engine.stt.active_model_id} ({self.engine.stt.backend_type}) | Min Confidence: {self.config.min_word_confidence:.2f}",
            thin,
            f" Live Clips   : {stats['live_clips']:<5d} (Single Words: {stats['single_words']}, Locked Chains: {stats['locked_chains']})",
            f" Starter Pack : {stats['starter_clips']:<5d} | Arena Storage: {stats['arena_mb']:.3f} / {stats['max_arena_mb']} MB | Dedup Pruned: {stats['total_dedup_pruned']}",
            thin,
            " RECENT CAPTURED CLIPS:",
        ]

        live_rows = self.engine.db.get_all_clips(include_starter=False)[:6]
        if not live_rows:
            lines.append("   (Waiting for desktop speech on hardware monitor...) ")
        else:
            for c in live_rows:
                kind = "LOCKED-CHAIN" if c.is_locked_phrase else "SINGLE-WORD "
                lines.append(
                    f"   [{c.id:03d}] {kind} \"{c.text:<26s}\" | f0={c.pitch_hz:5.1f}Hz | conf={c.confidence:.2f} | spk={c.speaker_tag}"
                )

        lines.append(thin)
        lines.append(" AUTO-SUGGESTED PLAYABLE SENTENCES (Type /1 .. /5 or /demo):")
        self.refresh_suggestions()
        for i, s in enumerate(self._suggestions[:5], start=1):
            lines.append(f"   /{i} -> \"{s}\"")

        if self._last_stitch is not None:
            lines.append(thin)
            lines.append(
                f" LAST STITCH ({self._last_stitch.total_ms:.1f}ms | Mode={self._last_stitch.voice_mode} | FX={self._last_stitch.fx_mode}):"
            )
            lines.append(f"   Assembled: \"{self._last_stitch.assembled_text}\"")

        lines.append(thin)
        lines.append(
            " COMMANDS: /rec (Pause/Resume) | /hear (Self-Monitor) | /mode | /fx | /models | /hw | /clear | /demo | /q"
        )
        lines.append(f" STATUS  : {self._status_banner}")
        lines.append(sep)
        return "\n".join(lines)

    def run_interactive_loop(self) -> int:
        self._running.set()
        while self._running.is_set():
            if sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(self.render_dashboard_text() + "\n")
            sys.stdout.flush()
            try:
                user_in = input("VoxSplice Speak / Command > ")
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down VoxSplice...")
                break
            if not self.execute_command_or_say(user_in):
                break
        return 0
