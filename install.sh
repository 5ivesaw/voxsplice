#!/usr/bin/env bash
# ==============================================================================
# VoxSplice (`voxsplice` / `5mic` / `mimic`) — Linux One-Line Installer
# Supports Pop!_OS 24.04 (COSMIC Wayland), Ubuntu 22.04+, Debian, Fedora, Arch
# ==============================================================================
set -euo pipefail

APP_DIR="${HOME}/.local/share/voxsplice-app"
BIN_DIR="${HOME}/.local/bin"
CONFIG_DIR="${HOME}/.config/voxsplice"
DATA_DIR="${HOME}/.local/share/voxsplice"
DESKTOP_DIR="${HOME}/.local/share/applications"

echo "=================================================================="
echo "  Installing VoxSplice v2.4.0 (Real-Time Voice Splicer & Studio)  "
echo "=================================================================="

mkdir -p "${APP_DIR}" "${BIN_DIR}" "${CONFIG_DIR}" "${DATA_DIR}" "${DESKTOP_DIR}"

# 1. Install native audio & STT packages when apt-get is available
if command -v apt-get >/dev/null 2>&1; then
  echo "[1/5] Checking native audio & PocketSphinx packages (apt)..."
  sudo apt-get update -y || true
  sudo apt-get install -y \
    python3 python3-venv python3-pip \
    pulseaudio-utils ffmpeg \
    libpocketsphinx3 libsphinxbase3 pocketsphinx-en-us \
    flite libflite1 || true
fi

# 2. Copy repository files into ~/.local/share/voxsplice-app
echo "[2/5] Copying VoxSplice runtime files to ${APP_DIR}..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -r "${SCRIPT_DIR}/src" "${SCRIPT_DIR}/ui" "${SCRIPT_DIR}/assets" "${SCRIPT_DIR}/bridges" "${SCRIPT_DIR}/scripts" "${SCRIPT_DIR}/pyproject.toml" "${SCRIPT_DIR}/requirements.txt" "${SCRIPT_DIR}/config.default.toml" "${APP_DIR}/"

if [ ! -f "${CONFIG_DIR}/config.toml" ]; then
  cp "${SCRIPT_DIR}/config.default.toml" "${CONFIG_DIR}/config.toml"
fi

# 3. Create isolated virtual environment & install package
echo "[3/5] Setting up Python virtual environment in ${APP_DIR}/.venv..."
python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
"${APP_DIR}/.venv/bin/pip" install -e "${APP_DIR}"

# 4. Create CLI launchers (`voxsplice`, `5mic`, `mimic`) in ~/.local/bin
echo "[4/5] Creating CLI symlinks in ${BIN_DIR} (voxsplice, 5mic, mimic)..."
for cmd in voxsplice 5mic mimic; do
  ln -sf "${APP_DIR}/.venv/bin/${cmd}" "${BIN_DIR}/${cmd}"
done

# 5. Create Desktop Launcher for COSMIC / GNOME / KDE
echo "[5/5] Installing desktop application entry..."
cat > "${DESKTOP_DIR}/voxsplice.desktop" <<EOF
[Desktop Entry]
Name=VoxSplice Studio
Comment=Real-Time Concatenative Voice Reconstruction & Coarticulation Splicer
Exec=${BIN_DIR}/voxsplice gui
Icon=${APP_DIR}/assets/icon.png
Terminal=false
Type=Application
Categories=AudioVideo;Audio;Utility;
Keywords=voice;splicer;5mic;mimic;pipewire;
EOF

echo ""
echo "✅ VoxSplice v2.4.0 installed!"
echo "   • Run 'voxsplice' (or '5mic') for the interactive single-command TUI."
echo "   • Run 'voxsplice gui' for the macOS-inspired Desktop Studio & Quick-Type Pill."
echo "   • Run 'voxsplice doctor' to inspect hardware & audio routing."
