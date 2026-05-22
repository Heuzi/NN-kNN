#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

echo "[codex setup] repository: $ROOT_DIR"
"$PYTHON_BIN" --version

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# The checked-in requirements are CUDA-oriented for local GPU workflows.
# Cloud Codex tasks are commonly CPU-only, so install CPU PyTorch explicitly.
"$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.9.1 torchvision==0.24.1

FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
grep -Ev '^(--extra-index-url|torch==|torchvision==)' requirements.txt > "$FILTERED_REQUIREMENTS"
"$PYTHON_BIN" -m pip install -r "$FILTERED_REQUIREMENTS"

mkdir -p checkpoints results

grep -qxF 'export MPLBACKEND=Agg' ~/.bashrc || printf '\nexport MPLBACKEND=Agg\n' >> ~/.bashrc
grep -qxF 'export PYTHONUNBUFFERED=1' ~/.bashrc || printf 'export PYTHONUNBUFFERED=1\n' >> ~/.bashrc
grep -qxF 'export NNKNN_DEVICE="${NNKNN_DEVICE:-cpu}"' ~/.bashrc || \
  printf 'export NNKNN_DEVICE="${NNKNN_DEVICE:-cpu}"\n' >> ~/.bashrc

"$PYTHON_BIN" codex/smoke_test.py --mode imports

echo "[codex setup] complete"
