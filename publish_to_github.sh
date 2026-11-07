#!/usr/bin/env bash
# ==============================================================================
# VoxSplice (`voxsplice` / `5mic`) — Automated GitHub Repository & Release Publisher
#
# Creates a GitHub repository, commits and pushes the complete VoxSplice project
# (including `.github/workflows/ci.yml` and `.github/workflows/release.yml`),
# and pushes tag `v2.4.0` to trigger GitHub Actions to build and publish the
# lightweight Tauri v2 desktop installers (.AppImage, .deb, .msi, .exe) and
# standalone archives to GitHub Releases.
#
# Usage:
#   ./publish_to_github.sh [REPO_NAME] [PUBLIC_OR_PRIVATE]
#
# Examples:
#   ./publish_to_github.sh voxsplice public
#   GITHUB_TOKEN=ghp_xxx GITHUB_USER=myuser ./publish_to_github.sh 5mic public
# ==============================================================================
set -euo pipefail

REPO_NAME="${1:-voxsplice}"
VISIBILITY="${2:-public}"   # "public" or "private"
RELEASE_TAG="${RELEASE_TAG:-v2.4.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo "  VoxSplice — GitHub Repository & Automated Tauri Release Publisher"
echo "======================================================================"
echo "  Target Repository : ${REPO_NAME} (${VISIBILITY})"
echo "  Release Tag       : ${RELEASE_TAG}"
echo "----------------------------------------------------------------------"

# 1. Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
  echo "[1/5] Initializing Git repository..."
  git init -b main
else
  echo "[1/5] Git repository already initialized."
  git branch -M main
fi

# Ensure git user identity is set locally if not configured globally
if ! git config user.name >/dev/null 2>&1; then
  git config user.name "${GITHUB_USER:-VoxSplice Maintainer}"
fi
if ! git config user.email >/dev/null 2>&1; then
  git config user.email "${GITHUB_EMAIL:-maintainer@voxsplice.local}"
fi

# 2. Stage and commit all project files, workflows, Tauri shell, and starter pack
echo "[2/5] Staging complete VoxSplice source tree & GitHub Actions workflows..."
git add -A
if ! git diff --cached --quiet; then
  git commit -m "Release VoxSplice ${RELEASE_TAG}: 5-Tier Viterbi Stitcher, PocketSphinx C/Vosk/Whisper STT, Model Hub & Tauri v2 Studio"
else
  echo "      Working tree clean; commit already up to date."
fi

# 3. Create the GitHub repository via `gh` CLI (preferred) or GitHub REST API (`GITHUB_TOKEN`)
echo "[3/5] Creating or connecting to GitHub repository '${REPO_NAME}'..."
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  GH_USER="$(gh api user -q .login)"
  FULL_REPO="${GH_USER}/${REPO_NAME}"
  if gh repo view "${FULL_REPO}" >/dev/null 2>&1; then
    echo "      Repository https://github.com/${FULL_REPO} already exists."
  else
    gh repo create "${REPO_NAME}" "--${VISIBILITY}" \
      --description "Lightweight real-time concatenative voice reconstruction & coarticulation-aware audio splicer (Linux PipeWire + Windows WASAPI + Tauri v2)" \
      --source=. \
      --remote=origin
  fi
  REMOTE_URL="https://github.com/${FULL_REPO}.git"
  if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "${REMOTE_URL}"
  else
    git remote set-url origin "${REMOTE_URL}"
  fi
elif [ -n "${GITHUB_TOKEN:-}" ]; then
  if [ -z "${GITHUB_USER:-}" ]; then
    GITHUB_USER="$(curl -fsSL -H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json" https://api.github.com/user | python3 -c "import sys,json; print(json.load(sys.stdin)['login'])")"
  fi
  PRIVATE_BOOL="false"
  if [ "${VISIBILITY}" = "private" ]; then
    PRIVATE_BOOL="true"
  fi
  echo "      Creating repository ${GITHUB_USER}/${REPO_NAME} via GitHub REST API..."
  HTTP_CODE="$(curl -sS -o /tmp/gh_create_resp.json -w "%{http_code}" \
    -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"${REPO_NAME}\",\"description\":\"Lightweight real-time concatenative voice reconstruction & coarticulation-aware audio splicer (Tauri v2 + PipeWire + WASAPI)\",\"private\":${PRIVATE_BOOL},\"has_issues\":true,\"has_wiki\":false}")"
  if [ "${HTTP_CODE}" != "201" ] && [ "${HTTP_CODE}" != "422" ]; then
    echo "ERROR: GitHub API returned HTTP ${HTTP_CODE}:"
    cat /tmp/gh_create_resp.json
    exit 1
  fi
  REMOTE_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
  if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "${REMOTE_URL}"
  else
    git remote set-url origin "${REMOTE_URL}"
  fi
else
  echo "ERROR: Neither an authenticated 'gh' CLI session nor GITHUB_TOKEN was found."
  echo "Please authenticate first using one of:"
  echo "  1) gh auth login && ./publish_to_github.sh ${REPO_NAME} ${VISIBILITY}"
  echo "  2) GITHUB_TOKEN=ghp_your_token ./publish_to_github.sh ${REPO_NAME} ${VISIBILITY}"
  exit 1
fi

# 4. Push `main` branch and tag `${RELEASE_TAG}` to trigger CI and Release workflows
echo "[4/5] Pushing 'main' branch to origin..."
git push -u origin main --force

echo "[5/5] Tagging '${RELEASE_TAG}' and pushing tag to trigger GitHub Actions Tauri v2 Release build..."
git tag -f "${RELEASE_TAG}"
git push origin "${RELEASE_TAG}" --force

# Clean any token from remote URL after push
if [ -n "${GITHUB_USER:-}" ]; then
  git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

echo ""
echo "======================================================================"
echo "  ✅ Published to GitHub!"
echo "  • Repository : $(git remote get-url origin | sed 's/\.git$//')"
echo "  • Actions CI : $(git remote get-url origin | sed 's/\.git$//')/actions"
echo "  • Releases   : $(git remote get-url origin | sed 's/\.git$//')/releases/tag/${RELEASE_TAG}"
echo "======================================================================"
