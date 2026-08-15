"""Fetch Hi-SAM's code and weights into ./models, for reading strokes with a trained model.

Not vendored into this repository. The model is about five thousand lines of third-party
PyTorch - the size of this whole project - and copying it in would double the code here with
something nobody here maintains. It is cloned at a pinned commit instead, into `models/`,
where the docTR and EasyOCR weights already live and which git already ignores.

Two of the three things needed come down on their own. The third cannot:

* the source, from GitHub, pinned (Apache 2.0)
* SAM's ViT backbone, 1.2 GB, from Meta's public bucket
* the SAM-TS head, 118 MB, published *only* through a OneDrive share whose account has
  migrated to SharePoint. Every programmatic route into it - the short link, the share API
  against both the short and the resolved URL, and download.aspx - answers 401, 400 or 403.
  It needs a browser, so this script prints the link and the path to save it at.

    .venv/Scripts/python scripts/fetch_hisam.py
    .venv/Scripts/python scripts/fetch_hisam.py --check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from dsf.refine import hisam  # noqa: E402

REPO = "https://github.com/ymy-k/Hi-SAM.git"

#: Pinned rather than tracking main, so a result measured today can be reproduced. Moving it
#: is a deliberate act with a re-run of `scripts/score_masks.py` attached.
COMMIT = "69009434d4dba5541f228d8f5acb0754c333d417"

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
HEAD_URL = "https://1drv.ms/u/s!AimBgYV7JjTlgco2ogP59MnXtR6bSw?e=5iq3Rt"


def _run(*cmd: str, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=None if cwd is None else str(cwd), check=True)


def fetch_source(force: bool = False) -> None:
    marker = hisam.CODE_DIR / "hi_sam" / "modeling" / "build.py"
    if marker.exists() and not force:
        print(f"source already at {hisam.CODE_DIR}")
        return
    hisam.CODE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (hisam.CODE_DIR / ".git").exists():
        print(f"cloning {REPO}")
        _run("git", "clone", "--quiet", REPO, str(hisam.CODE_DIR))
    print(f"pinning to {COMMIT[:12]}")
    _run("git", "-C", str(hisam.CODE_DIR), "fetch", "--quiet", "--depth", "50", "origin",
         COMMIT)
    _run("git", "-C", str(hisam.CODE_DIR), "checkout", "--quiet", COMMIT)


def fetch_backbone() -> None:
    hisam.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = hisam.WEIGHTS_DIR / hisam._ENCODERS["vit_l"]
    if out.exists():
        print(f"backbone already at {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"downloading {out.name} (1.2 GB) from {SAM_URL}")
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(SAM_URL) as response, open(tmp, "wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:.0%}", end="", flush=True)
    print()
    tmp.rename(out)


def report_head() -> bool:
    """The one file a human has to fetch. Returns whether it is already here."""
    out = hisam.WEIGHTS_DIR / hisam.DEFAULT_HEAD
    if out.exists():
        print(f"head already at {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
        return True
    print()
    print("The SAM-TS head cannot be downloaded from here. It is published only through a")
    print("OneDrive share that refuses every programmatic request, so a browser is needed:")
    print()
    print(f"  1. open   {HEAD_URL}")
    print(f"  2. save as {out}")
    print()
    print("That is SAM-TS-L trained on TextSeg (fgIoU 88.77), which is the design-text")
    print("variant and the nearest published domain to a title card.")
    print()
    print("Note the licence: TextSeg is academia-only and cannot be used commercially, and")
    print("that inherits to anything trained on it.")
    return False


def check_deps() -> bool:
    """Two imports the model needs that this project does not otherwise.

    Reported rather than installed. Both are pure Python and harmless in themselves, but
    every `pip install` into this venv is a chance to replace the CUDA-enabled OpenCV with a
    stock wheel - see the README - and a script that fetches weights has no business taking
    that risk on the caller's behalf.
    """
    missing = []
    for name in ("einops", "timm"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print(f"\nAlso needed, and not installed: {', '.join(missing)}")
        print(f"  .venv/Scripts/python -m pip install {' '.join(missing)}")
        print("  then re-run scripts/install_opencv_cuda.ps1 if OpenCV loses CUDA")
    return not missing


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="report what is present and stop")
    p.add_argument("--force", action="store_true", help="re-pin the source even if present")
    args = p.parse_args(argv)

    if args.check:
        ready = hisam.available() and check_deps()
        print(f"code     {'yes' if (hisam.CODE_DIR / 'hi_sam').exists() else 'MISSING'}")
        for name in (hisam._ENCODERS["vit_l"], hisam.DEFAULT_HEAD):
            path = hisam.WEIGHTS_DIR / name
            size = f"{path.stat().st_size / 1e6:.0f} MB" if path.exists() else "MISSING"
            print(f"{name[:20]:20s} {size}")
        print(f"\n{'ready' if ready else 'not ready - run without --check'}")
        return 0 if ready else 1

    fetch_source(args.force)
    fetch_backbone()
    have_head = report_head()
    have_deps = check_deps()
    if have_head and have_deps and hisam.available():
        print("\nready")
        return 0
    print("\nnot ready until the head above is saved")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
