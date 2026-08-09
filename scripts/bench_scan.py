"""How long does a scanned frame take, and where does the time go?

Two measurements, because they answer different questions.

The *pipeline* number is the honest one: a real clip through `iter_masks_detailed`, once per
backend, reported in milliseconds per frame. It includes decoding and detection, which is what
makes it honest - the mask chain is a share of a frame's cost, not all of it, and a speedup
quoted on the share alone would flatter itself.

The *stages* number takes each full-frame operation on its own at the clip's resolution and
times both backends on it. That is what says which stage is worth porting next, and it is the
number to check when a change is meant to have made something faster.

    .venv/Scripts/python scripts/bench_scan.py samples/demo_rgb.mp4 --profile both

Needs a CUDA-enabled OpenCV to report anything but the CPU column; see
`scripts/install_opencv_cuda.ps1`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402

import cv2  # noqa: E402


def timed(fn, repeats: int) -> float:
    """Milliseconds per call, after one warm-up. Both backends get the same treatment."""
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000.0


def bench_stages(height: int, width: int, repeats: int = 30) -> list[tuple]:
    """Each full-frame stage, on its own, at this frame size."""
    from dsf.accel import CpuOps, CudaOps, cuda_available

    rng = np.random.default_rng(0)
    cpu = CpuOps()
    gpu = CudaOps() if cuda_available() else None
    if gpu is not None:
        gpu.reserve({"shape": 8, "level": 26, "median": 26})

    alpha = rng.random((height, width), np.float32)
    frame = rng.integers(0, 256, (height, width)).astype(np.uint8)
    other = rng.integers(0, 256, (height, width)).astype(np.uint8)
    window = [rng.integers(0, 256, (height, width)).astype(np.uint8) for _ in range(3)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    #: The prior asks this of every frame in a twenty-one frame window, once per frame, which
    #: is why it is measured as a whole window rather than as a single threshold.
    prior_window = 21

    def cases(ops, up):
        a, f, o = up(alpha), up(frame), up(other)
        win = [up(w) for w in window]
        return [
            ("to_u8", lambda: ops.to_u8(a, "shape")),
            ("median(3)", lambda: ops.median(win)),
            (f"prior threshold+or (x{prior_window})", lambda: _prior(ops, win, prior_window)),
            ("dilate 7x7", lambda: ops.dilate(f, kernel)),
            ("scale_by", lambda: ops.scale_by(f, o)),
            ("peak", lambda: ops.peak(f)),
            ("download", lambda: ops.download(f)),
        ]

    def _prior(ops, win, count):
        acc = None
        for i in range(count):
            acc = ops.or_into(acc, ops.over_127(win[i % len(win)]))
        return acc

    rows = []
    cpu_cases = cases(cpu, lambda x: x)
    gpu_cases = cases(gpu, gpu.upload) if gpu is not None else None
    for i, (name, fn) in enumerate(cpu_cases):
        on_cpu = timed(fn, repeats)
        on_gpu = timed(gpu_cases[i][1], repeats) if gpu_cases else float("nan")
        rows.append((name, on_cpu, on_gpu))
    return rows


def bench_pipeline(clip: str, profile: str, frames: int, backend: str) -> tuple[float, int]:
    """A real clip through the real chain. Returns (ms per frame, frames measured)."""
    os.environ["DSF_ACCEL"] = backend

    from dsf.config import configure_model_cache

    configure_model_cache()

    from dsf import accel
    from dsf.detect import build_detectors
    from dsf.media import probe
    from dsf.pipeline import iter_masks_detailed
    from scan_for_text import build_scan_config

    cfg = build_scan_config(profile, batch_size=8, device="cuda")
    info = probe(clip)
    detectors = build_detectors(cfg.detect.detectors, cfg.detect)
    assert accel.resolve(cfg.detect.device) == backend, "backend override did not take"

    # Warm up: the first pass pays for model compilation and the first CUDA allocations, and
    # charging those to whichever backend happened to run first would be meaningless.
    list(iter_masks_detailed(clip, cfg, info, max_frames=8, detectors=detectors))

    count = 0
    start = time.perf_counter()
    for _ in iter_masks_detailed(clip, cfg, info, max_frames=frames, detectors=detectors):
        count += 1
    elapsed = time.perf_counter() - start
    return elapsed / max(1, count) * 1000.0, count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", nargs="?", default=str(_ROOT / "samples/demo_rgb.mp4"))
    parser.add_argument("--profile", default="both", choices=("subtitles", "credits", "both"))
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--stages-only", action="store_true",
                        help="skip the clip and time the individual stages")
    args = parser.parse_args(argv)

    from dsf.accel import cuda_available

    if not cuda_available():
        print("NOTE: OpenCV reports no CUDA device - only the CPU column is real.\n")

    if not args.stages_only and not Path(args.clip).exists():
        print(f"no such clip: {args.clip}", file=sys.stderr)
        return 2

    height, width = 1080, 1920
    if Path(args.clip).exists():
        from dsf.media import probe

        info = probe(args.clip)
        height, width = info.height, info.width

    print(f"stages at {width}x{height}, ms per call")
    print(f"  {'stage':<32}{'cpu':>10}{'cuda':>10}{'speedup':>10}")
    for name, on_cpu, on_gpu in bench_stages(height, width):
        speed = f"{on_cpu / on_gpu:.1f}x" if on_gpu == on_gpu and on_gpu > 0 else "-"
        gpu_text = f"{on_gpu:.3f}" if on_gpu == on_gpu else "-"
        print(f"  {name:<32}{on_cpu:>10.3f}{gpu_text:>10}{speed:>10}")

    if args.stages_only:
        return 0

    # Each backend in its own process: `iter_masks_detailed` reads the backend once per call,
    # and a torch model already resident from the other run would skew the second reading.
    import subprocess

    print(f"\npipeline on {Path(args.clip).name}, {args.frames} frames, "
          f"profile {args.profile}")
    results = {}
    for backend in ("cpu", "cuda"):
        out = subprocess.run(
            [sys.executable, __file__, args.clip, "--profile", args.profile,
             "--frames", str(args.frames), "--_run", backend],
            capture_output=True, text=True, env=dict(os.environ, DSF_ACCEL=backend))
        if out.returncode != 0:
            print(f"  {backend:<32}failed:\n{out.stderr[-2000:]}")
            continue
        results[backend] = float(out.stdout.strip().splitlines()[-1])
        print(f"  {backend:<32}{results[backend]:>10.1f} ms/frame")
    if len(results) == 2 and results["cuda"] > 0:
        print(f"  {'speedup':<32}{results['cpu'] / results['cuda']:>9.2f}x")
    return 0


if __name__ == "__main__":
    # The child process of the pipeline comparison: run one backend and print one number.
    if "--_run" in sys.argv:
        at = sys.argv.index("--_run")
        which = sys.argv[at + 1]
        rest = sys.argv[1:at]
        opts = argparse.ArgumentParser()
        opts.add_argument("clip")
        opts.add_argument("--profile", default="both")
        opts.add_argument("--frames", type=int, default=48)
        parsed = opts.parse_args(rest)
        per_frame, _ = bench_pipeline(parsed.clip, parsed.profile, parsed.frames, which)
        print(f"{per_frame:.4f}")
    else:
        raise SystemExit(main())
