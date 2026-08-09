# Creates a self-contained .venv for depth-subtitle-fixer.
# Safe to re-run. Nothing outside the project directory is modified.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "==> Creating .venv (Python 3.12)" -ForegroundColor Cyan
    py -3.12 -m venv .venv
}

Write-Host "==> Upgrading pip/setuptools/wheel" -ForegroundColor Cyan
& $py -m pip install --upgrade pip setuptools wheel

# torch FIRST, from the CUDA 13.0 index. The RTX 50-series (Blackwell, sm_120) needs a
# CUDA >= 12.8 build; the default PyPI wheel on Windows is CPU-only.
Write-Host "==> Installing torch (cu130)" -ForegroundColor Cyan
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

Write-Host "==> Installing depth-subtitle-fixer + deps" -ForegroundColor Cyan
& $py -m pip install -e ".[ui,dev]"

# python-doctr depends on opencv-python, easyocr on opencv-python-headless. Both unpack into
# the same cv2/ directory, so whichever installs last wins -- and neither is built with CUDA,
# which leaves `dsf.accel` running the mask chain on the CPU at roughly half the speed. So the
# last word goes to a CUDA-enabled build instead. This must stay the final step: anything
# installed after it can overwrite cv2/ again.
Write-Host "==> Installing the CUDA-enabled OpenCV" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "install_opencv_cuda.ps1")
if ($LASTEXITCODE -ne 0) {
    # A missing CUDA toolkit or a failed download must not fail the whole setup: everything
    # works without it, only slower. Say so and carry on.
    Write-Host "==> CUDA OpenCV unavailable; falling back to the headless CPU build" -ForegroundColor Yellow
    & $py -m pip install --force-reinstall --no-deps opencv-python-headless
}

Write-Host "==> Verifying" -ForegroundColor Cyan
& $py -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
& $py -m dsf.cli --version

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
