# Replace the stock OpenCV in .venv with a CUDA-enabled build.
#
# The stock `opencv-python` wheels are built without CUDA, so `cv2.cuda` exists as a namespace
# with no devices in it and `dsf.accel` falls back to numpy. This installs cudawarped's
# contrib build instead, which is the same OpenCV with `WITH_CUDA=ON`.
#
# Safe to re-run, and re-running is the fix whenever `pip install` has pulled a plain
# `opencv-python` back in as a dependency of docTR or easyocr and unpacked it over this one.
#
#     powershell -ExecutionPolicy Bypass -File scripts\install_opencv_cuda.ps1
#
# Nothing outside the project directory is modified.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv at $py - run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Pinned rather than "latest": the wheel is built against one CUDA minor version and one
# cuDNN, and a silent bump would change what `import cv2` needs on the machine.
$version = "4.13.0.90"
$wheel = "opencv_contrib_python-$version-cp37-abi3-win_amd64.whl"
$url = "https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/$version/$wheel"
$cudaVersion = "v13.1"

# ---------------------------------------------------------------- toolkit check
# The wheel links against the CUDA runtime rather than bundling it, so `import cv2` fails
# outright without the toolkit. Checked up front because the DLL error it raises otherwise
# says only "The specified module could not be found", which names nothing.
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cudaBin = Join-Path $cudaRoot "$cudaVersion\bin\x64"
if (-not (Test-Path (Join-Path $cudaBin "cudart64_13.dll"))) {
    $alt = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "bin\x64\cudart64_13.dll") }
    if (-not $alt) {
        Write-Host "This wheel needs the CUDA 13.x toolkit and none was found under" -ForegroundColor Red
        Write-Host "  $cudaRoot" -ForegroundColor Red
        Write-Host "Install CUDA $cudaVersion from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
        exit 1
    }
    Write-Host "==> CUDA $cudaVersion not found; using $($alt[0].Name)" -ForegroundColor Yellow
    $cudaBin = Join-Path $alt[0].FullName "bin\x64"
}

# ---------------------------------------------------------------- download
$cache = Join-Path $root ".cache"
if (-not (Test-Path $cache)) { New-Item -ItemType Directory -Force $cache | Out-Null }
$local = Join-Path $cache $wheel

if (Test-Path $local) {
    Write-Host "==> Using cached $wheel" -ForegroundColor Cyan
} else {
    Write-Host "==> Downloading $wheel (~194 MB)" -ForegroundColor Cyan
    curl.exe -L --fail --progress-bar -o $local $url
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $local) { Remove-Item $local }
        Write-Host "Download failed." -ForegroundColor Red
        exit 1
    }
}

# ---------------------------------------------------------------- install
# All four names unpack into the same cv2/ directory, so whichever installed last wins and a
# leftover is indistinguishable from a fresh install. Clear the lot before putting ours down.
Write-Host "==> Removing any other OpenCV" -ForegroundColor Cyan
& $py -m pip uninstall -y opencv-python opencv-python-headless `
    opencv-contrib-python opencv-contrib-python-headless

Write-Host "==> Installing $wheel" -ForegroundColor Cyan
# --no-deps: the wheel's only dependency is numpy, which is already pinned by the project, and
# letting pip re-resolve it here can drag in a different one.
& $py -m pip install --no-deps $local

# ---------------------------------------------------------------- cuDNN path
# The wheel is built against cuDNN 9 and expects to find it in the CUDA toolkit directory,
# where the toolkit installer does not put it. torch ships the same soname next to itself, so
# point cv2 at that rather than requiring a second multi-hundred-megabyte download of a
# library the venv already has. Appended to cv2/config.py, which cv2 reads while bootstrapping
# to build its DLL search path.
Write-Host "==> Pointing cv2 at torch's cuDNN" -ForegroundColor Cyan
$patch = @'
import site
from pathlib import Path

MARKER = "# --- added by scripts/install_opencv_cuda.ps1 ---"
BLOCK = """

# --- added by scripts/install_opencv_cuda.ps1 ---
# This wheel is built against cuDNN 9 and expects it inside the CUDA toolkit directory, where
# it is not installed. torch ships the same soname (cudnn64_9.dll) alongside itself, so search
# there too rather than asking for a second copy of a library the venv already has.
import os
BINARIES_PATHS = [os.path.join(LOADER_DIR, "..", "torch", "lib")] + BINARIES_PATHS
"""

for root in site.getsitepackages():
    config = Path(root) / "cv2" / "config.py"
    if not config.exists():
        continue
    text = config.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"    {config} already patched")
    else:
        config.write_text(text + BLOCK, encoding="utf-8")
        print(f"    patched {config}")
    break
else:
    raise SystemExit("could not find cv2/config.py in site-packages")
'@
$patch | & $py -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---------------------------------------------------------------- verify
Write-Host "==> Verifying" -ForegroundColor Cyan
$verify = @'
import cv2
n = cv2.cuda.getCudaEnabledDeviceCount()
print("cv2", cv2.__version__, "| CUDA devices", n)
if not n:
    raise SystemExit("OpenCV imported but reports no CUDA device")
cv2.cuda.printShortCudaDeviceInfo(0)
'@
$verify | & $py -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host 'Done. Note that "pip check" will now report that python-doctr and easyocr want' -ForegroundColor Green
Write-Host 'opencv-python: that is expected, and harmless. Re-run this script after any' -ForegroundColor Green
Write-Host '"pip install", which may have overwritten cv2/ with a CPU-only build.' -ForegroundColor Green
