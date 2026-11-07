"""
minecraft_chat_bridge.py — Universal Minecraft Chat Log Tailer & Localhost UDP Forwarder for VoxSplice.

Monitors `~/.minecraft/logs/latest.log` for messages prefixed with `.m ` (e.g. `.m hello im human`)
and dispatches them over UDP (`127.0.0.1:9999`) to the VoxSplice daemon for immediate playback
through `MimicMic` inside Simple Voice Chat (SVC).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import time
from pathlib import Path
from typing import Optional

_CHAT_PREFIX_RE = re.compile(r"(?:\[CHAT\]|\<[^>]+?\>|\:)\s*\.m\s+(.+)$", re.IGNORECASE)


def default_minecraft_log_path() -> Path:
    sys_name = platform.system()
    if sys_name == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / ".minecraft" / "logs" / "latest.log"
    if sys_name == "Darwin":
        return Path.home() / "Library" / "Application Support" / "minecraft" / "logs" / "latest.log"
    return Path.home() / ".minecraft" / "logs" / "latest.log"


def extract_mimic_command(log_line: str) -> Optional[str]:
    line = log_line.strip()
    if ".m " not in line:
        return None
    m = _CHAT_PREFIX_RE.search(line)
    if m:
        text = m.group(1).strip()
        return text if text else None
    idx = line.find(".m ")
    if idx != -1:
        text = line[idx + 3 :].strip()
        return text if text else None
    return None


def forward_to_daemon(text: str, host: str = "127.0.0.1", port: int = 9999, mode: Optional[str] = None) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = json.dumps({"text": text, "mode": mode}).encode("utf-8")
    try:
        sock.sendto(payload, (host, port))
    finally:
        sock.close()


def tail_minecraft_log(
    log_path: Path, host: str = "127.0.0.1", port: int = 9999, mode: Optional[str] = None
) -> None:
    print(f"[VoxSplice MC Bridge] Watching {log_path} for `.m <message>` commands -> UDP {host}:{port}")
    while not log_path.exists():
        time.sleep(1.0)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.03)
                continue
            cmd = extract_mimic_command(line)
            if cmd:
                print(f"[VoxSplice MC Bridge] Triggered: \"{cmd}\"")
                forward_to_daemon(cmd, host=host, port=port, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxSplice Minecraft Chat Log Bridge")
    parser.add_argument("--log", type=Path, default=default_minecraft_log_path())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--mode", default=None)
    parsed = parser.parse_args()
    try:
        tail_minecraft_log(parsed.log, host=parsed.host, port=parsed.port, mode=parsed.mode)
    except KeyboardInterrupt:
        pass
