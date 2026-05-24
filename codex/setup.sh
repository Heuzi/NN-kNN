#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPECTED_PYTHON_VERSION="3.11.9"
VENV_DIR="$ROOT_DIR/.venv"

python_version() {
  "$1" -c 'import platform; print(platform.python_version())'
}

if [[ -x "$VENV_DIR/bin/python" ]] && [[ "$(python_version "$VENV_DIR/bin/python")" == "$EXPECTED_PYTHON_VERSION" ]]; then
  :
elif command -v python3 >/dev/null 2>&1 && [[ "$(python_version python3)" == "$EXPECTED_PYTHON_VERSION" ]]; then
  python3 -m venv "$VENV_DIR"
else
  UV_BIN="$(command -v uv || true)"
  if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
  fi
  if [[ -z "$UV_BIN" ]]; then
    echo "[codex setup] Python $EXPECTED_PYTHON_VERSION is required; install it or install uv." >&2
    exit 1
  fi
  "$UV_BIN" venv --python "$EXPECTED_PYTHON_VERSION" --seed "$VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"

echo "[codex setup] repository: $ROOT_DIR"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# The checked-in requirements are CUDA-oriented for local GPU workflows.
# Cloud Codex tasks are commonly CPU-only, so install CPU PyTorch explicitly.
REQUIREMENTS_EXCLUDE='^(--extra-index-url|torch==|torchvision==)'
if [[ "$(uname -s)-$(uname -m)" == "Darwin-x86_64" ]]; then
  # Current PyTorch releases no longer publish Intel macOS wheels.
  "$PYTHON_BIN" -m pip install numpy==1.26.4 torch==2.2.2 torchvision==0.17.2
  REQUIREMENTS_EXCLUDE='^(--extra-index-url|torch==|torchvision==|numpy==)'
else
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.1 torchvision==0.24.1
fi

FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
grep -Ev "$REQUIREMENTS_EXCLUDE" requirements.txt > "$FILTERED_REQUIREMENTS"
"$PYTHON_BIN" -m pip install -r "$FILTERED_REQUIREMENTS"

mkdir -p checkpoints results

touch ~/.bashrc
grep -qxF 'export MPLBACKEND=Agg' ~/.bashrc || printf '\nexport MPLBACKEND=Agg\n' >> ~/.bashrc
grep -qxF 'export PYTHONUNBUFFERED=1' ~/.bashrc || printf 'export PYTHONUNBUFFERED=1\n' >> ~/.bashrc
grep -qxF 'export NNKNN_DEVICE="${NNKNN_DEVICE:-cpu}"' ~/.bashrc || \
  printf 'export NNKNN_DEVICE="${NNKNN_DEVICE:-cpu}"\n' >> ~/.bashrc

"$PYTHON_BIN" codex/smoke_test.py --mode imports

echo "[codex setup] complete"
