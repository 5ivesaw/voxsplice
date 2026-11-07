"""
api_server.py — Local FastAPI REST & WebSocket Bridge Server for the Tauri v2 Desktop App
and Web Studio Interface (`http://127.0.0.1:18420`).

Exposes real-time endpoints for:
- `/api/status`        : Live engine status, database stats, hardware profile, and active audio routing.
- `/api/clips`         : Searchable/filterable live & starter clips in `clips.db` (`all`, `low_pitch`, `high_pitch`, speaker filter).
- `/api/seams`         : Most recent utterance waveform + 20ms Coarticulation Seam Inspector telemetry.
- `/api/preview`       : Instant Viterbi stitch plan preview (tiers, similarity, pitch, gain, subword parts) as the user types.
- `/api/say`           : Reconstructs and transmits speech to `MimicMic` + self-monitor, returning base64 WAV audio for browser/Tauri waveform playback.
- `/api/models`        : Lists the Model & Tokenizer catalog with real hardware compatibility scores.
- `/api/models/search` : Live search against the Hugging Face Hub API (`https://huggingface.co/api/models`).
- `/api/models/install`: Dynamically downloads and activates a selected STT, embedding, or tokenizer model.
- `/api/config`        : Reads or updates slicer/stitcher/routing parameters in real time.
- `/api/ingest-demo`   : Ingests a sample from local voice recordings on disk for immediate interactive testing.
"""

from __future__ import annotations

import base64
import io
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .audio_io import CrossPlatformAudioEngine
from .config import AUDIO_FORMAT_SPECS, FX_MODES, VOICE_MODES, get_base_dir
from .hardware import inspect_system_hardware
from .model_hub import ModelHubManager


class SayRequest(BaseModel):
    text: str
    voice_mode: Optional[str] = None
    fx_mode: Optional[str] = None
    speaker_filter: Optional[str] = None
    play_audio: bool = True


class PreviewRequest(BaseModel):
    text: str
    voice_mode: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    voice_mode: Optional[str] = None
    fx_mode: Optional[str] = None
    fx_intensity: Optional[float] = None
    audio_format: Optional[str] = None
    seam_rms_threshold: Optional[float] = None
    silence_gap_ms: Optional[float] = None
    min_word_confidence: Optional[float] = None
    crossfade_ms: Optional[float] = None
    monitor_self: Optional[bool] = None
    use_starter_pack_fallback: Optional[bool] = None
    enable_subword_splicing: Optional[bool] = None
    locked_victim_id: Optional[str] = None


class ModelInstallRequest(BaseModel):
    model_id: str
    variant_id: Optional[str] = None


class PinSpeakerRequest(BaseModel):
    profile_name: str
    recent_seconds: float = 300.0


class IngestWavRequest(BaseModel):
    wav_path: str
    speaker_tag: str = "lobby"
    max_duration_s: Optional[float] = 25.0
    start_offset_s: float = 0.0


def _pcm16_to_wav_base64(pcm16_bytes: bytes, sample_rate: int = 16000) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16_bytes)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_api_app(engine: CrossPlatformAudioEngine) -> FastAPI:
    app = FastAPI(
        title="VoxSplice Studio Engine API",
        version="2.4.0",
        description="Local real-time concatenative voice reconstruction & coarticulation splicing bridge.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_Methods=["*"],
        allow_headers=["*"],
    )

    hub = ModelHubManager(engine.config)
    hw_profile = inspect_system_hardware(engine.config.resolve_models_dir())

    @app.get("/", response_class=HTMLResponse)
    def serve_ui_root() -> Any:
        ui_file = get_base_dir() / "ui" / "index.html"
        if ui_file.exists():
            return FileResponse(str(ui_file), media_type="text/html")
        return HTMLResponse("<h1>VoxSplice API Active</h1>")

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        stats = engine.db.get_stats()
        vmic = engine.setup_virtual_microphone()
        return {
            "ok": True,
            "recording_active": engine.is_recording_active,
            "input_rms": round(float(engine.current_input_rms), 5),
            "status_message": engine.last_status_message,
            "stt_backend": engine.stt.backend_type,
            "stt_model_id": engine.stt.active_model_id,
            "stt_status": engine.stt.status_message,
            "config": engine.config.to_dict(),
            "stats": stats,
            "speakers": engine.db.list_speaker_profiles(),
            "suggestions": engine.stitcher.suggest_playable_sentences(count=5),
            "virtual_mic": vmic.to_dict(),
            "devices": engine.list_audio_devices(),
            "hardware": hw_profile.to_dict(),
            "audio_formats": AUDIO_FORMAT_SPECS,
            "voice_modes": VOICE_MODES,
            "fx_modes": FX_MODES,
        }

    @app.get("/api/clips")
    def list_clips(
        include_starter: bool = False,
        pitch_filter: str = "all",
        speaker_tag: Optional[str] = None,
        limit: int = 120,
    ) -> Dict[str, Any]:
        clips = engine.db.get_all_clips(include_starter=include_starter, speaker_tag=speaker_tag)
        if pitch_filter == "low":
            clips = [c for c in clips if c.pitch_hz <= engine.config.low_pitch_cutoff_hz]
        elif pitch_filter == "high":
            clips = [c for c in clips if c.pitch_hz >= engine.config.high_pitch_cutoff_hz]

        return {
            "ok": True,
            "total": len(clips),
            "clips": [c.to_dict() for c in clips[:limit]],
        }

    @app.get("/api/seams")
    def get_seams() -> Dict[str, Any]:
        wf = engine.last_waveform
        step = max(1, len(wf) // 240)
        wf_down = [round(float(x), 4) for x in wf[::step]] if len(wf) > 0 else []
        return {
            "ok": True,
            "seams": [s.to_dict() for s in engine.last_seams],
            "waveform": wf_down,
        }

    @app.post("/api/preview")
    def preview_stitch(req: PreviewRequest) -> Dict[str, Any]:
        plan = engine.stitcher.preview_sentence_plan(req.text, voice_mode=req.voice_mode)
        return {
            "ok": True,
            "request_text": req.text,
            "assembled_text": " ".join(c.matched_text for c in plan),
            "chunks": [c.to_dict() for c in plan],
        }

    @app.post("/api/say")
    def trigger_say(req: SayRequest) -> Dict[str, Any]:
        res = engine.say(
            req.text,
            voice_mode=req.voice_mode,
            fx_mode=req.fx_mode,
            play_audio=req.play_audio,
            blocking=False,
            speaker_filter=req.speaker_filter,
        )
        wav_b64 = (
            _pcm16_to_wav_base64(res.samples_int16.tobytes(), sample_rate=res.sample_rate)
            if len(res.samples_int16) > 0
            else ""
        )
        step = max(1, len(res.samples_f32) // 240)
        wf_preview = [round(float(x), 4) for x in res.samples_f32[::step]] if len(res.samples_f32) > 0 else []

        return {
            "ok": True,
            "request_text": res.request_text,
            "assembled_text": res.assembled_text,
            "voice_mode": res.voice_mode,
            "fx_mode": res.fx_mode,
            "search_ms": res.search_ms,
            "dsp_stitch_ms": res.dsp_stitch_ms,
            "total_ms": res.total_ms,
            "duration_s": round(len(res.samples_f32) / float(max(1, res.sample_rate)), 2),
            "chunks": [c.to_dict() for c in res.chunks],
            "waveform": wf_preview,
            "audio_wav_base64": wav_b64,
        }

    @app.post("/api/toggle-recording")
    def toggle_recording() -> Dict[str, Any]:
        active = engine.toggle_recording()
        return {"ok": True, "recording_active": active, "message": engine.last_status_message}

    @app.post("/api/toggle-monitor")
    def toggle_monitor() -> Dict[str, Any]:
        mon = engine.toggle_monitor_self()
        engine.config.save()
        return {"ok": True, "monitor_self": mon, "message": engine.last_status_message}

    @app.post("/api/clear-clips")
    def clear_clips() -> Dict[str, Any]:
        removed = engine.db.clear_live_clips()
        return {"ok": True, "removed": removed, "stats": engine.db.get_stats()}

    @app.post("/api/pin-speaker")
    def pin_speaker(req: PinSpeakerRequest) -> Dict[str, Any]:
        count = engine.db.pin_speaker_profile(req.profile_name, recent_seconds=req.recent_seconds)
        return {"ok": True, "pinned_count": count, "speakers": engine.db.list_speaker_profiles()}

    @app.post("/api/config")
    def update_config(req: ConfigUpdateRequest) -> Dict[str, Any]:
        updates = req.dict(exclude_none=True)
        for k, v in updates.items():
            if hasattr(engine.config, k):
                setattr(engine.config, k, v)
        engine.config.save()
        return {"ok": True, "config": engine.config.to_dict()}

    @app.get("/api/models")
    def get_models(category: Optional[str] = None) -> Dict[str, Any]:
        entries = hub.list_catalog(category=category, hw=hw_profile)
        return {
            "ok": True,
            "hardware": hw_profile.to_dict(),
            "models": [e.to_dict() for e in entries],
        }

    @app.get("/api/models/search")
    def search_hf(query: str = "whisper onnx", task: Optional[str] = None) -> Dict[str, Any]:
        return hub.search_huggingface_hub(query=query, task_filter=task)

    @app.post("/api/models/install")
    def install_model(req: ModelInstallRequest) -> Dict[str, Any]:
        ok, msg = engine.stt.download_and_switch_model(req.model_id, variant_id=req.variant_id)
        return {
            "ok": ok,
            "message": msg,
            "active_model_id": engine.stt.active_model_id,
            "stt_status": engine.stt.status_message,
            "models": [e.to_dict() for e in hub.list_catalog(hw=hw_profile)],
        }

    @app.post("/api/ingest-wav")
    def ingest_wav(req: IngestWavRequest) -> Dict[str, Any]:
        path = Path(req.wav_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"WAV file not found: {req.wav_path}")
        clips, seams, words = engine.ingest_wav_file(
            path,
            speaker_tag=req.speaker_tag,
            max_duration_s=req.max_duration_s,
            start_offset_s=req.start_offset_s,
        )
        return {
            "ok": True,
            "ingested_clips": len(clips),
            "recognized_words": [w.to_dict() for w in words],
            "seams": [s.to_dict() for s in seams],
            "stats": engine.db.get_stats(),
        }

    return app
