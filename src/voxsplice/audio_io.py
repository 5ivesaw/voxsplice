"""
audio_io.py — Cross-Platform Desktop Loopback Capture, Virtual Mic Provisioning,
Dual-Output Self-Monitoring, and Atomic Half-Duplex Feedback Lock.

Features:
1. Linux PipeWire / PulseAudio native routing (`pactl load-module module-null-sink` + `module-remap-source`
   for `MimicMic`), plus automatic destruction of any rogue `module-loopback` modules on startup so
   hardware feedback loops can never occur.
2. Windows WASAPI Loopback capture (`soundcard`) + VB-Cable detection and 1-click guided installer.
3. Dual-Output Playback with `HalfDuplexGuard`: plays stitched speech to both `MimicMic` (so Discord,
   VRChat, Roblox, and Minecraft hear your virtual mic) AND your physical headphones/speakers (so you
   hear your own stitched output) while locking capture + 260ms post-playback drain.
4. File and Buffer Ingestion API (`ingest_wav_file`, `ingest_pcm16_utterance`) for offline testing,
   voice pack importing, and live Push-to-Talk voice puppetry.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .arena_db import ArenaDatabase
from .config import VoxSpliceConfig
from .dsp import compute_rms, float32_to_int16, int16_to_float32, resample_audio
from .slicer import CoarticulationSlicer, SeamAnalysis, SlicedClip
from .stitcher import FrankensteinStitcher, StitchResult
from .stt_backends import MultiBackendSTTExtractor, WordTimestamp


@dataclass
class VirtualMicStatus:
    os_name: str
    is_ready: bool
    device_name: str
    monitor_source: str
    backend: str
    message: str
    linux_module_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HalfDuplexGuard:
    """
    Atomic Half-Duplex transmission lock.
    Blocks incoming loopback capture while stitched audio is playing into `MimicMic` and
    the user's headphones, plus a configurable post-playback drain cooldown (`260ms`).
    """

    def __init__(self, cooldown_ms: float = 260.0) -> None:
        self.cooldown_s = cooldown_ms / 1000.0
        self._transmitting = threading.Event()
        self._cooldown_until: float = 0.0
        self._lock = threading.Lock()

    def begin_transmit(self) -> None:
        with self._lock:
            self._transmitting.set()

    def end_transmit(self) -> None:
        with self._lock:
            self._cooldown_until = time.monotonic() + self.cooldown_s
            self._transmitting.clear()

    @property
    def is_blocked(self) -> bool:
        if self._transmitting.is_set():
            return True
        with self._lock:
            return time.monotonic() < self._cooldown_until


class CrossPlatformAudioEngine:
    def __init__(
        self,
        config: VoxSpliceConfig,
        db: ArenaDatabase,
        slicer: CoarticulationSlicer,
        stitcher: FrankensteinStitcher,
        stt_extractor: Optional[MultiBackendSTTExtractor] = None,
    ) -> None:
        self.config = config
        self.db = db
        self.slicer = slicer
        self.stitcher = stitcher
        self.stt = stt_extractor or MultiBackendSTTExtractor(config)
        self.guard = HalfDuplexGuard(cooldown_ms=config.half_duplex_cooldown_ms)

        self._running = threading.Event()
        self._recording_paused = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._linux_module_ids: List[int] = []
        self._hardware_sink_name: Optional[str] = None
        self._last_seams: List[SeamAnalysis] = []
        self._last_waveform: np.ndarray = np.zeros(1600, dtype=np.float32)
        self.current_input_rms: float = 0.0
        self.last_status_message: str = "Ready"
        self._on_clip_callbacks: List[Callable[[SlicedClip], None]] = []

    def add_on_clip_callback(self, cb: Callable[[SlicedClip], None]) -> None:
        self._on_clip_callbacks.append(cb)

    @property
    def is_recording_active(self) -> bool:
        return self._running.is_set() and not self._recording_paused.is_set()

    def toggle_recording(self) -> bool:
        if self._recording_paused.is_set():
            self._recording_paused.clear()
            self.last_status_message = "Recording RESUMED"
            return True
        self._recording_paused.set()
        self.current_input_rms = 0.0
        self.last_status_message = "Recording PAUSED"
        return False

    def toggle_monitor_self(self) -> bool:
        self.config.monitor_self = not bool(self.config.monitor_self)
        state = "ON (Headphones + MimicMic)" if self.config.monitor_self else "OFF (MimicMic Only)"
        self.last_status_message = f"Self-Monitor: {state}"
        return self.config.monitor_self

    def cleanup_rogue_loopbacks(self) -> int:
        if platform.system() != "Linux" or not shutil.which("pactl"):
            return 0
        removed = 0
        try:
            out = subprocess.check_output(
                ["pactl", "list", "short", "modules"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3.0,
            )
            for line in out.splitlines():
                low = line.lower()
                if "module-loopback" in low and (
                    self.config.virtual_mic_name.lower() in low or "mimic" in low
                ):
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        subprocess.run(
                            ["pactl", "unload-module", parts[0]],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=2.0,
                        )
                        removed += 1
        except Exception:
            pass
        return removed

    def detect_hardware_speaker_sink(self) -> Optional[str]:
        if platform.system() != "Linux" or not shutil.which("pactl"):
            return None
        sink_name = self.config.virtual_mic_name.lower()
        try:
            default_sink = subprocess.check_output(
                ["pactl", "get-default-sink"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
            if default_sink and sink_name not in default_sink.lower() and "mimic" not in default_sink.lower():
                self._hardware_sink_name = default_sink
                return default_sink
        except Exception:
            pass

        try:
            out = subprocess.check_output(
                ["pactl", "list", "short", "sinks"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            )
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    candidate = parts[1].strip()
                    if sink_name not in candidate.lower() and "mimic" not in candidate.lower():
                        self._hardware_sink_name = candidate
                        return candidate
        except Exception:
            pass
        return None

    def list_audio_devices(self) -> Dict[str, List[str]]:
        inputs: List[str] = []
        outputs: List[str] = []

        if platform.system() == "Linux" and shutil.which("pactl"):
            try:
                sinks_out = subprocess.check_output(
                    ["pactl", "list", "short", "sinks"], text=True, timeout=2.0, stderr=subprocess.DEVNULL
                )
                for ln in sinks_out.splitlines():
                    parts = ln.split("\t")
                    if len(parts) >= 2:
                        outputs.append(parts[1].strip())
                sources_out = subprocess.check_output(
                    ["pactl", "list", "short", "sources"], text=True, timeout=2.0, stderr=subprocess.DEVNULL
                )
                for ln in sources_out.splitlines():
                    parts = ln.split("\t")
                    if len(parts) >= 2:
                        inputs.append(parts[1].strip())
            except Exception:
                pass

        try:
            import soundcard as sc  # type: ignore

            for spk in sc.all_speakers():
                if spk.name not in outputs:
                    outputs.append(spk.name)
            for mic in sc.all_microphones(include_loopback=True):
                if mic.name not in inputs:
                    inputs.append(mic.name)
        except Exception:
            pass

        return {
            "inputs": inputs or ["Default System Monitor (Loopback)"],
            "outputs": outputs or ["Default System Output", self.config.virtual_mic_name],
        }

    def setup_virtual_microphone(self) -> VirtualMicStatus:
        sys_name = platform.system()
        sink_name = self.config.virtual_mic_name

        if sys_name == "Linux":
            self.cleanup_rogue_loopbacks()
            self.detect_hardware_speaker_sink()

            pactl_bin = shutil.which("pactl")
            if not pactl_bin:
                return VirtualMicStatus(
                    os_name="Linux",
                    is_ready=False,
                    device_name=sink_name,
                    monitor_source="",
                    backend="PipeWire/PulseAudio",
                    message="`pactl` not found. Install `pulseaudio-utils` or `pipewire-pulse`.",
                    linux_module_ids=[],
                )

            try:
                existing = subprocess.check_output(
                    [pactl_bin, "list", "short", "sinks"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=3.0,
                )
                if sink_name in existing:
                    return VirtualMicStatus(
                        os_name="Linux",
                        is_ready=True,
                        device_name=sink_name,
                        monitor_source=f"{sink_name}.monitor",
                        backend="PipeWire/PulseAudio",
                        message=f"Virtual Microphone '{sink_name}' active (Dual-Output + Anti-Feedback Guard enabled).",
                        linux_module_ids=list(self._linux_module_ids),
                    )
            except Exception:
                pass

            mod_ids: List[int] = []
            try:
                out_sink = subprocess.check_output(
                    [
                        pactl_bin,
                        "load-module",
                        "module-null-sink",
                        f"sink_name={sink_name}",
                        f"sink_properties=device.description=VoxSplice_Virtual_Speaker_{sink_name}",
                    ],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=4.0,
                ).strip()
                if out_sink.isdigit():
                    mod_ids.append(int(out_sink))

                try:
                    out_src = subprocess.check_output(
                        [
                            pactl_bin,
                            "load-module",
                            "module-remap-source",
                            f"master={sink_name}.monitor",
                            f"source_name={sink_name}_Source",
                            f"source_properties=device.description={sink_name}",
                        ],
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=4.0,
                    ).strip()
                    if out_src.isdigit():
                        mod_ids.append(int(out_src))
                except Exception:
                    pass

                self._linux_module_ids.extend(mod_ids)
                return VirtualMicStatus(
                    os_name="Linux",
                    is_ready=True,
                    device_name=sink_name,
                    monitor_source=f"{sink_name}.monitor",
                    backend="PipeWire/PulseAudio",
                    message=f"Created PipeWire Virtual Mic '{sink_name}' (select '{sink_name}' in Discord/VRChat).",
                    linux_module_ids=mod_ids,
                )
            except Exception as exc:
                return VirtualMicStatus(
                    os_name="Linux",
                    is_ready=False,
                    device_name=sink_name,
                    monitor_source="",
                    backend="PipeWire/PulseAudio",
                    message=f"PipeWire daemon not reachable ({exc}). Running in local simulation mode.",
                    linux_module_ids=[],
                )

        if sys_name == "Windows":
            has_vb, dev_label = self._detect_windows_vbcable()
            if has_vb:
                return VirtualMicStatus(
                    os_name="Windows",
                    is_ready=True,
                    device_name=dev_label,
                    monitor_source="CABLE Output (VB-Audio Virtual Cable)",
                    backend="WASAPI + VB-Cable",
                    message=f"Connected to {dev_label}. Select 'CABLE Output' as your in-game mic.",
                    linux_module_ids=[],
                )
            return VirtualMicStatus(
                os_name="Windows",
                is_ready=False,
                device_name="CABLE Input (Missing)",
                monitor_source="",
                backend="WASAPI",
                message="VB-Cable Virtual Driver not detected. Use 1-Click Installer in Settings.",
                linux_module_ids=[],
            )

        return VirtualMicStatus(
            os_name=sys_name,
            is_ready=False,
            device_name=sink_name,
            monitor_source="",
            backend="Generic",
            message=f"OS '{sys_name}' virtual mic auto-setup requires BlackHole or Loopback.",
            linux_module_ids=[],
        )

    def teardown_virtual_microphone(self) -> None:
        if platform.system() == "Linux":
            self.cleanup_rogue_loopbacks()
            if self._linux_module_ids and shutil.which("pactl"):
                pactl_bin = shutil.which("pactl") or "pactl"
                for mid in reversed(self._linux_module_ids):
                    try:
                        subprocess.run(
                            [pactl_bin, "unload-module", str(mid)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=2.0,
                        )
                    except Exception:
                        pass
                self._linux_module_ids.clear()

    def _detect_windows_vbcable(self) -> Tuple[bool, str]:
        try:
            import soundcard as sc  # type: ignore

            for spk in sc.all_speakers():
                name = str(spk.name)
                if "cable input" in name.lower() or "vb-audio" in name.lower():
                    return True, name
        except Exception:
            pass
        return False, ""

    def install_windows_vbcable_one_click(self) -> Tuple[bool, str]:
        if platform.system() != "Windows":
            return False, "VB-Cable 1-Click Installer is only required on Windows 10/11."

        vb_url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
        temp_dir = Path(tempfile.gettempdir()) / "voxsplice_vbcable_setup"
        temp_dir.mkdir(parents=True, exist_ok=True)
        zip_path = temp_dir / "VBCABLE_Driver_Pack43.zip"

        try:
            urllib.request.urlretrieve(vb_url, str(zip_path))
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(temp_dir)

            installer_exe = temp_dir / "VBCABLE_Setup_x64.exe"
            if not installer_exe.exists():
                installer_exe = temp_dir / "VBCABLE_Setup.exe"
            if not installer_exe.exists():
                return False, "Downloaded archive did not contain VBCABLE_Setup_x64.exe."

            ret = ctypes.windll.shell32.ShellExecuteW(  # type: ignore[attr-defined]
                None, "runas", str(installer_exe), "-i -h", str(temp_dir), 1
            )
            if int(ret) <= 32:
                return False, f"Administrator UAC prompt declined (code {ret})."
            return True, "VB-Cable silent installer launched. Select 'CABLE Output' in Discord/VRChat."
        except Exception as exc:
            return False, f"VB-Cable auto-install failed: {exc}"

    def ingest_pcm16_utterance(
        self,
        pcm16: np.ndarray,
        sample_rate: int = 16000,
        speaker_tag: str = "lobby",
        override_words: Optional[List[WordTimestamp]] = None,
        known_text_hint: Optional[str] = None,
    ) -> Tuple[List[SlicedClip], List[SeamAnalysis]]:
        if self.guard.is_blocked or self._recording_paused.is_set() or len(pcm16) == 0:
            return [], []

        f32 = int16_to_float32(pcm16)
        self.current_input_rms = compute_rms(f32)
        if len(f32) >= 200:
            idx = np.linspace(0, len(f32) - 1, num=min(1600, len(f32)), dtype=np.int32)
            self._last_waveform = f32[idx].copy()

        words = (
            override_words
            if override_words is not None
            else self.stt.process_utterance_pcm16(
                pcm16, sample_rate=sample_rate, known_text_hint=known_text_hint
            )
        )
        if not words:
            return [], []

        clips, seams = self.slicer.slice_utterance(
            pcm16, words, sample_rate=sample_rate, speaker_tag=speaker_tag
        )
        if seams:
            self._last_seams = seams

        stored_clips: List[SlicedClip] = []
        for clip in clips:
            cid = self.db.insert_clip(clip, pinned=0)
            if cid is not None:
                stored_clips.append(clip)
                for cb in self._on_clip_callbacks:
                    try:
                        cb(clip)
                    except Exception:
                        pass

        return stored_clips, seams

    def ingest_wav_file(
        self,
        wav_path: Path | str,
        speaker_tag: str = "lobby",
        max_duration_s: Optional[float] = None,
        start_offset_s: float = 0.0,
        known_text_hint: Optional[str] = None,
    ) -> Tuple[List[SlicedClip], List[SeamAnalysis], List[WordTimestamp]]:
        """
        Reads a real `.wav` recording from disk, extracts an optional slice `[start_offset_s, start_offset_s + max_duration_s]`,
        runs real STT word timestamping + Coarticulation Slicer, and stores the resulting clips in `clips.db`.
        """
        path = Path(wav_path)
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()

            start_frame = min(total_frames, max(0, int(round(start_offset_s * sr))))
            wf.setpos(start_frame)
            frames_to_read = (
                min(total_frames - start_frame, int(round(max_duration_s * sr)))
                if max_duration_s is not None
                else (total_frames - start_frame)
            )
            raw_bytes = wf.readframes(frames_to_read)

        if sampwidth == 2:
            pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        elif sampwidth == 1:
            u8 = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int16)
            pcm = (u8 - 128) * 256
        else:
            pcm = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            pcm = float32_to_int16(pcm)

        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch)[:, 0]

        pcm_16k = resample_audio(pcm, sr, 16000) if sr != 16000 else pcm
        words = self.stt.process_utterance_pcm16(
            pcm_16k, sample_rate=16000, known_text_hint=known_text_hint
        )
        clips, seams = self.ingest_pcm16_utterance(
            pcm_16k, sample_rate=16000, speaker_tag=speaker_tag, override_words=words
        )
        return clips, seams, words

    def start_background_capture(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._capture_thread = threading.Thread(
            target=self._capture_worker, name="VoxSplice-LoopbackCapture", daemon=True
        )
        self._capture_thread.start()

    def stop_background_capture(self) -> None:
        self._running.clear()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.5)
        self._capture_thread = None

    def _capture_worker(self) -> None:
        sr = self.config.capture_sample_rate
        block_frames = int(sr * 0.1)

        if platform.system() == "Linux" and shutil.which("parec"):
            hw_sink = self.detect_hardware_speaker_sink()
            if hw_sink:
                mon_dev = f"{hw_sink}.monitor"
                try:
                    proc = subprocess.Popen(
                        [
                            "parec",
                            f"--device={mon_dev}",
                            "--format=s16le",
                            f"--rate={sr}",
                            "--channels=1",
                            "--latency-msec=40",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    bytes_per_block = block_frames * 2
                    utterance_buf: List[np.ndarray] = []
                    silence_blocks = 0

                    while self._running.is_set() and proc.poll() is None:
                        if not proc.stdout:
                            break
                        raw = proc.stdout.read(bytes_per_block)
                        if not raw or len(raw) < bytes_per_block:
                            continue

                        if self.guard.is_blocked or self._recording_paused.is_set():
                            utterance_buf.clear()
                            silence_blocks = 0
                            self.current_input_rms = 0.0
                            continue

                        pcm16 = np.frombuffer(raw, dtype=np.int16)
                        mono = int16_to_float32(pcm16)
                        rms = compute_rms(mono)
                        self.current_input_rms = rms

                        if rms >= self.config.min_word_rms:
                            utterance_buf.append(pcm16)
                            silence_blocks = 0
                            if len(utterance_buf) >= 42:
                                full_pcm = np.concatenate(utterance_buf)
                                utterance_buf.clear()
                                self.ingest_pcm16_utterance(full_pcm, sample_rate=sr, speaker_tag="lobby")
                        else:
                            if utterance_buf:
                                silence_blocks += 1
                                utterance_buf.append(pcm16)
                                if silence_blocks >= 4:
                                    full_pcm = np.concatenate(utterance_buf)
                                    utterance_buf.clear()
                                    silence_blocks = 0
                                    self.ingest_pcm16_utterance(
                                        full_pcm, sample_rate=sr, speaker_tag="lobby"
                                    )
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    return
                except Exception:
                    pass

        try:
            import soundcard as sc  # type: ignore

            default_spk = sc.default_speaker()
            loopback_mic = None
            for m in sc.all_microphones(include_loopback=True):
                if getattr(m, "isloopback", False) and "mimic" not in m.name.lower():
                    if default_spk and default_spk.name in m.name:
                        loopback_mic = m
                        break
                    if loopback_mic is None:
                        loopback_mic = m

            if loopback_mic is None:
                return

            utterance_buf_sc: List[np.ndarray] = []
            silence_blocks_sc = 0

            with loopback_mic.recorder(samplerate=sr, channels=1, blocksize=block_frames) as rec:
                while self._running.is_set():
                    data = rec.record(numframes=block_frames)
                    if self.guard.is_blocked or self._recording_paused.is_set():
                        utterance_buf_sc.clear()
                        silence_blocks_sc = 0
                        self.current_input_rms = 0.0
                        continue

                    mono = np.asarray(data[:, 0], dtype=np.float32)
                    rms = compute_rms(mono)
                    self.current_input_rms = rms
                    if rms >= self.config.min_word_rms:
                        utterance_buf_sc.append(float32_to_int16(mono))
                        silence_blocks_sc = 0
                        if len(utterance_buf_sc) >= 42:
                            full_pcm = np.concatenate(utterance_buf_sc)
                            utterance_buf_sc.clear()
                            self.ingest_pcm16_utterance(full_pcm, sample_rate=sr, speaker_tag="lobby")
                    else:
                        if utterance_buf_sc:
                            silence_blocks_sc += 1
                            utterance_buf_sc.append(float32_to_int16(mono))
                            if silence_blocks_sc >= 4:
                                full_pcm = np.concatenate(utterance_buf_sc)
                                utterance_buf_sc.clear()
                                silence_blocks_sc = 0
                                self.ingest_pcm16_utterance(
                                    full_pcm, sample_rate=sr, speaker_tag="lobby"
                                )
        except Exception:
            while self._running.is_set():
                time.sleep(0.25)

    def play_stitched_result(self, result: StitchResult, blocking: bool = True) -> None:
        if len(result.samples_f32) == 0:
            return

        def _tx() -> None:
            self.guard.begin_transmit()
            try:
                dur_s = len(result.samples_f32) / float(max(1, result.sample_rate))

                if platform.system() == "Linux" and shutil.which("paplay"):
                    tmp_wav = Path(tempfile.gettempdir()) / f"voxsplice_out_{os.getpid()}_{threading.get_ident()}.wav"
                    try:
                        with wave.open(str(tmp_wav), "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(result.sample_rate)
                            wf.writeframes(result.samples_int16.tobytes())

                        procs: List[subprocess.Popen] = []
                        procs.append(
                            subprocess.Popen(
                                ["paplay", f"--device={self.config.virtual_mic_name}", str(tmp_wav)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        )
                        if self.config.monitor_self:
                            hw_sink = self._hardware_sink_name or self.detect_hardware_speaker_sink()
                            if hw_sink and hw_sink != self.config.virtual_mic_name:
                                procs.append(
                                    subprocess.Popen(
                                        ["paplay", f"--device={hw_sink}", str(tmp_wav)],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                    )
                                )
                            else:
                                procs.append(
                                    subprocess.Popen(
                                        ["paplay", str(tmp_wav)],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                    )
                                )

                        for p in procs:
                            try:
                                p.wait(timeout=max(3.0, dur_s + 2.5))
                            except Exception:
                                pass
                        return
                    except Exception:
                        pass
                    finally:
                        try:
                            tmp_wav.unlink(missing_ok=True)
                        except Exception:
                            pass

                try:
                    import soundcard as sc  # type: ignore

                    target_spk = None
                    default_spk = sc.default_speaker()
                    for spk in sc.all_speakers():
                        low = spk.name.lower()
                        if self.config.virtual_mic_name.lower() in low or "cable input" in low:
                            target_spk = spk
                            break

                    data_col = result.samples_f32.reshape(-1, 1)
                    if target_spk is not None and self.config.monitor_self and default_spk and default_spk.name != target_spk.name:
                        t_Default = threading.Thread(
                            target=lambda: default_spk.play(data_col, samplerate=result.sample_rate),
                            daemon=True,
                        )
                        t_Default.start()
                        target_spk.play(data_col, samplerate=result.sample_rate)
                        t_Default.join(timeout=dur_s + 1.0)
                    elif target_spk is not None:
                        target_spk.play(data_col, samplerate=result.sample_rate)
                    elif default_spk is not None:
                        default_spk.play(data_col, samplerate=result.sample_rate)
                    return
                except Exception:
                    pass

                time.sleep(min(0.05, dur_s))
            finally:
                self.guard.end_transmit()

        if blocking:
            _tx()
        else:
            threading.Thread(target=_tx, name="VoxSplice-Transmit", daemon=True).start()

    def say(
        self,
        text: str,
        voice_mode: Optional[str] = None,
        fx_mode: Optional[str] = None,
        play_audio: bool = True,
        blocking: bool = False,
        speaker_filter: Optional[str] = None,
        save_wav_path: Optional[Path | str] = None,
    ) -> StitchResult:
        res = self.stitcher.stitch_sentence(
            text, voice_mode=voice_mode, fx_mode=fx_mode, speaker_filter=speaker_filter
        )
        if len(res.samples_f32) > 0:
            idx = np.linspace(
                0, len(res.samples_f32) - 1, num=min(1600, len(res.samples_f32)), dtype=np.int32
            )
            self._last_waveform = res.samples_f32[idx].copy()

        if save_wav_path and len(res.samples_int16) > 0:
            out_p = Path(save_wav_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_p), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(res.sample_rate)
                wf.writeframes(res.samples_int16.tobytes())

        if play_audio and len(res.samples_f32) > 0:
            self.play_stitched_result(res, blocking=blocking)
        return res

    @property
    def last_seams(self) -> List[SeamAnalysis]:
        return list(self._last_seams)

    @property
    def last_waveform(self) -> np.ndarray:
        return self._last_waveform.copy()
