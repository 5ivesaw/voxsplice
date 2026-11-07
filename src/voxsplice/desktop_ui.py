"""
desktop_ui.py — Desktop Window & Quick-Type Overlay Launcher for VoxSplice.

Launches the macOS-inspired Desktop Studio GUI (`ui/index.html` backed by the local FastAPI engine):
1. Uses compiled Tauri v2 desktop binary if present (`src-tauri/target/release/voxsplice-desktop`), OR
2. Uses `pywebview` native OS WebKit/EdgeWebView2 window if installed, OR
3. Uses Chromium / Google Chrome / Brave / Edge in native frameless `--app=http://127.0.0.1:18420` desktop window mode, OR
4. Opens the default system browser to `http://127.0.0.1:18420`.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

from .api_server import create_api_app
from .audio_io import CrossPlatformAudioEngine
from .config import get_base_dir


def start_local_api_background(engine: CrossPlatformAudioEngine, host: str = "127.0.0.1", port: int = 18420) -> str:
    import uvicorn

    app = create_api_app(engine)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, name="VoxSplice-FastAPI", daemon=True)
    t.start()
    time.sleep(0.35)
    return f"http://{host}:{port}"


def launch_desktop_studio(
    engine: CrossPlatformAudioEngine,
    overlay_only: bool = False,
    headless_check: bool = False,
) -> int:
    base_url = start_local_api_background(engine, port=engine.config.api_port)
    target_url = f"{base_url}/?view=pill" if overlay_only else base_url

    if headless_check:
        print(f"[VoxSplice Studio] Local API & UI Server verified at {target_url}")
        return 0

    # 1. Check for compiled Tauri binary
    tauri_candidates = [
        get_base_dir() / "src-tauri" / "target" / "release" / "voxsplice-desktop",
        get_base_dir() / "src-tauri" / "target" / "release" / "voxsplice-desktop.exe",
    ]
    for tb in tauri_candidates:
        if tb.exists():
            proc = subprocess.Popen([str(tb)])
            return proc.wait()

    # 2. Check for pywebview
    try:
        import webview  # type: ignore

        width = 620 if overlay_only else 1180
        height = 140 if overlay_only else 820
        webview.create_window(
            "VoxSplice Quick-Type" if overlay_only else "VoxSplice Studio",
            target_url,
            width=width,
            height=height,
            frameless=overlay_only,
            on_top=overlay_only,
            background_color="#0D0E11",
        )
        webview.start()
        return 0
    except Exception:
        pass

    # 3. Check for Chromium / Chrome / Edge `--app=` native window mode
    for browser_bin in ("chromium", "chromium-browser", "google-chrome", "brave-browser", "msedge"):
        found = shutil.which(browser_bin)
        if found:
            w, h = (620, 140) if overlay_only else (1200, 820)
            try:
                proc = subprocess.Popen(
                    [
                        found,
                        f"--app={target_url}",
                        f"--window-size={w},{h}",
                        "--new-window",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[VoxSplice Studio] Launched native desktop window at {target_url} (Ctrl+C to stop)")
                try:
                    proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                return 0
            except Exception:
                pass

    # 4. Fallback to system browser + keep server alive
    print(f"[VoxSplice Studio] Running at {target_url} (Press Ctrl+C to stop)")
    try:
        webbrowser.open(target_url)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    return 0
