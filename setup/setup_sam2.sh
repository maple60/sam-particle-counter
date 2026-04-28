#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  SAM2 setup for macOS/Linux
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_DIR=".venv"
SAM2_DIR="sam2_repo"
SAM2_REPO="https://github.com/facebookresearch/sam2.git"
SAM2_REF="main"

# Choose one or more models: tiny small base_plus large
# If you want only one, e.g.:
# SAM2_MODELS=(base_plus)
SAM2_MODELS=(tiny small base_plus large)

# 1 = run uv sync before setup, 0 = skip
SYNC_PROJECT=1

# 1 = let uv auto-select the PyTorch backend after sync, 0 = keep synced torch build
AUTO_TORCH_BACKEND=1

# 1 = skip SAM2 CUDA extension build (safer on some environments)
# 0 = try normal install
SKIP_SAM2_CUDA=0

printf '\n'
printf '============================================================\n'
printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'SAM2 repo:    %s\n' "${SAM2_REPO}"
printf 'SAM2 ref:     %s\n' "${SAM2_REF}"
printf 'Models:       %s\n' "${SAM2_MODELS[*]}"
printf 'Torch backend auto: %s\n' "${AUTO_TORCH_BACKEND}"
printf '============================================================\n\n'

require_command() {
  local cmd="$1"
  local msg="$2"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[Error] ${msg}" >&2
    exit 1
  fi
}

detect_downloader() {
  if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
    echo "[OK] Downloader: curl"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
    echo "[OK] Downloader: wget"
    return
  fi

  echo "[Error] Neither curl nor wget was found." >&2
  exit 1
}

download_model() {
  local model="$1"
  local file=""
  local url_base="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

  case "${model}" in
    tiny) file="sam2.1_hiera_tiny.pt" ;;
    small) file="sam2.1_hiera_small.pt" ;;
    base_plus) file="sam2.1_hiera_base_plus.pt" ;;
    large) file="sam2.1_hiera_large.pt" ;;
    *)
      echo "[Error] Unknown model name: ${model}" >&2
      exit 1
      ;;
  esac

  local target="${SAM2_DIR}/checkpoints/${file}"
  local url="${url_base}/${file}"

  if [[ -f "${target}" ]]; then
    echo "[Skip] ${file} already exists."
    return
  fi

  echo "[Download] ${file}"

  if [[ "${DOWNLOADER}" == "curl" ]]; then
    if ! curl -L --fail -o "${target}" "${url}"; then
      echo "[Error] Failed to download ${file} with curl." >&2
      rm -f "${target}"
      exit 1
    fi
    return
  fi

  if [[ "${DOWNLOADER}" == "wget" ]]; then
    if ! wget -O "${target}" "${url}"; then
      echo "[Error] Failed to download ${file} with wget." >&2
      rm -f "${target}"
      exit 1
    fi
  fi
}

install_torch_auto() {
  echo "[Info] Selecting PyTorch backend automatically with uv..."
  if ! uv pip install --upgrade torch --torch-backend=auto; then
    echo "[Error] Failed to install PyTorch with automatic backend selection." >&2
    echo "[Hint] Upgrade uv if --torch-backend is unavailable, or install PyTorch manually from README." >&2
    exit 1
  fi

  uv run --no-sync python -c "import torch; print('[OK] torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"
}

require_command uv "uv is not installed or not on PATH."
require_command git "git is not installed or not on PATH."
detect_downloader

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[Info] Creating virtual environment with uv..."
  uv venv
else
  echo "[OK] Virtual environment already exists: ${VENV_DIR}"
fi

if [[ "${SYNC_PROJECT}" == "1" ]]; then
  echo "[Info] Syncing project environment..."
  uv sync
  if [[ "${AUTO_TORCH_BACKEND}" == "1" ]]; then
    install_torch_auto
  else
    echo "[Skip] PyTorch backend auto-selection skipped."
  fi
else
  echo "[Skip] uv sync skipped."
  echo "[Skip] PyTorch backend auto-selection skipped because uv sync was skipped."
fi

if [[ ! -d "${SAM2_DIR}/.git" ]]; then
  echo "[Info] Cloning SAM2 repository..."
  git clone "${SAM2_REPO}" "${SAM2_DIR}"
else
  echo "[OK] SAM2 repository already exists."
fi

echo "[Info] Fetching latest refs from SAM2 repository..."
git -C "${SAM2_DIR}" fetch --all --tags --prune

echo "[Info] Checking out SAM2 ref: ${SAM2_REF}"
git -C "${SAM2_DIR}" checkout "${SAM2_REF}"

if [[ "${SAM2_REF}" == "main" ]]; then
  echo "[Info] Pulling latest changes for main..."
  git -C "${SAM2_DIR}" pull --ff-only origin main
fi

mkdir -p "${SAM2_DIR}/checkpoints"

echo "[Info] Ensuring SAM2 checkpoints..."
for model in "${SAM2_MODELS[@]}"; do
  download_model "${model}"
done

echo "[Info] Reinstalling SAM2 in editable mode..."
uv pip uninstall -y SAM-2 >/dev/null 2>&1 || true

pushd "${SAM2_DIR}" >/dev/null
if [[ "${SKIP_SAM2_CUDA}" == "1" ]]; then
  echo "[Info] Installing SAM2 with SAM2_BUILD_CUDA=0 ..."
  SAM2_BUILD_CUDA=0 uv pip install -e .
else
  uv pip install -e .
fi
popd >/dev/null

echo "[OK] SAM2 setup completed successfully."
