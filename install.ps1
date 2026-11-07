# ==============================================================================
# VoxSplice (`voxsplice` / `5mic` / `mimic`) — Windows 10/11 PowerShell Installer
# Supports Windows 10/11 (WASAPI + VB-Audio Virtual Cable)
# ==============================================================================
$ErrorActionPreference = "Stop"

$AppDir = Join-Path $env:LOCALAPPDATA "VoxSplice\app"
$ConfigDir = Join-Path $env:APPDATA "VoxSplice"
$DataDir = Join-Path $env:LOCALAPPDATA "VoxSplice\data"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Installing VoxSplice v2.4.0 for Windows 10/11 (WASAPI/VB-Cable) " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $AppDir, $ConfigDir, $DataDir | Out-Null

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item -Recurse -Force (Join-Path $ScriptDir "src"), (Join-Path $ScriptDir "ui"), (Join-Path $ScriptDir "assets"), (Join-Path $ScriptDir "bridges"), (Join-Path $ScriptDir "scripts"), (Join-Path $ScriptDir "pyproject.toml"), (Join-Path $ScriptDir "requirements.txt"), (Join-Path $ScriptDir "config.default.toml") $AppDir

$ConfigTarget = Join-Path $ConfigDir "config.toml"
if (-not (Test-Path $ConfigTarget)) {
    Copy-Item (Join-Path $ScriptDir "config.default.toml") $ConfigTarget
}

Write-Host "[1/2] Creating Python virtual environment..." -ForegroundColor Green
python -m venv (Join-Path $AppDir ".venv")
& (Join-Path $AppDir ".venv\Scripts\pip.exe") install --upgrade pip
& (Join-Path $AppDir ".venv\Scripts\pip.exe") install -r (Join-Path $AppDir "requirements.txt") sounddevice vosk
& (Join-Path $AppDir ".venv\Scripts\pip.exe") install -e $AppDir

Write-Host "[2/2] Adding VoxSplice CLI (voxsplice, 5mic, mimic) to User PATH..." -ForegroundColor Green
$VenvScripts = Join-Path $AppDir ".venv\Scripts"
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$VenvScripts*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$VenvScripts", "User")
}

Write-Host ""
Write-Host "VoxSplice v2.4.0 installed! Open a new terminal and run:" -ForegroundColor Green
Write-Host "  voxsplice          (Interactive TUI)"
Write-Host "  voxsplice gui      (Desktop Studio + Pinnable Quick-Type Overlay)"
Write-Host "  voxsplice doctor   (Hardware & VB-Cable Diagnostics)"
