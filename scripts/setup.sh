#!/usr/bin/env bash
# Creates a self-contained .venv for depth-subtitle-fixer. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
    echo "==> Creating .venv"
    python3 -m venv .venv
fi

echo "==> Upgrading pip/setuptools/wheel"
"$PY" -m pip install --upgrade pip setuptools wheel

# torch FIRST, from the CUDA index. Blackwell (RTX 50-series, sm_120) needs CUDA >= 12.8.
# Change the index or drop it entirely for CPU-only / ROCm setups.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
echo "==> Installing torch from $TORCH_INDEX"
"$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX"

echo "==> Installing depth-subtitle-fixer + deps"
"$PY" -m pip install -e ".[ui,dev]"

# python-doctr wants opencv-python, easyocr wants opencv-python-headless, and they unpack
# into the same cv2/ directory. Force headless to win - we never call cv2.imshow.
echo "==> Pinning opencv to the headless build"
"$PY" -m pip install --force-reinstall --no-deps opencv-python-headless

echo "==> Verifying"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
"$PY" -m dsf.cli --version

echo
echo "Done. Activate with:  source .venv/bin/activate"
