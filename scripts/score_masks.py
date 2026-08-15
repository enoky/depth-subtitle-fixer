"""Score any mask source against a labelled clip, so two of them can be compared honestly.

Takes the labels `label_glyphs.py` wrote and one or more mask sources - a `.mkv` cache from
`dsf detect`, or a folder of PNGs from anything else - and reports foreground IoU, recall and
precision per frame, plus the medians and the worst frame.

    .venv/Scripts/python scripts/score_masks.py --labels labels.npz \
        --mask luma=masks.mkv --mask hisam=./hisam_out --frames 26-61

Three numbers rather than one, because they fail in different directions and the difference
matters here. Recall missed is corrupted depth showing through the writing, which is the
artefact the tool exists to remove. Precision missed is a mask fatter than the letters, which
the heal step largely absorbs. A single score would hide which of those a change traded for
the other.

The worst frame is reported alongside the median on purpose. A method that averages well and
collapses on the frames where the text crosses something its own colour is not the same as
one that never collapses, and the median cannot tell you which you have.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from dsf.maskcache import read_masks  # noqa: E402


def parse_ranges(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_masks(spec: str, shape: tuple[int, int]) -> dict[int, np.ndarray]:
    """A mask source, by frame index. Either a mask cache or a folder of numbered images.

    A folder is resized to the labelled clip's size if it differs, because a source that
    worked on a crop is still worth scoring - but it must be the *same* crop, so the caller
    is trusted about that.
    """
    path = Path(spec)
    if path.is_dir():
        out = {}
        for f in sorted(path.glob("*.png")):
            digits = "".join(c for c in f.stem if c.isdigit())
            if not digits:
                continue
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m.shape != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            out[int(digits)] = m > 127
        return out
    return {i: (m > 127) for i, m in enumerate(read_masks(path))}


def score(pred: np.ndarray, truth: np.ndarray, core: np.ndarray) -> tuple[float, float, float]:
    """Foreground IoU, recall on the glyph interiors, and precision."""
    inter = int((pred & truth).sum())
    union = int((pred | truth).sum())
    return (inter / max(union, 1),
            int((pred & core).sum()) / max(int(core.sum()), 1),
            inter / max(int(pred.sum()), 1) if pred.any() else 0.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help=".npz from label_glyphs.py")
    p.add_argument("--mask", action="append", required=True, metavar="NAME=PATH",
                   help="a mask cache or folder to score; repeat to compare several")
    p.add_argument("--frames", help="which frames to score (default: the labelled ones)")
    p.add_argument("--window", action="store_true",
                   help="score only inside the labelled text's own bounding box")
    p.add_argument("--per-frame", action="store_true", help="print every frame, not a summary")
    args = p.parse_args(argv)

    data = np.load(args.labels, allow_pickle=True)
    truth = data["mask"]
    frames = parse_ranges(args.frames) if args.frames else [int(i) for i in data["on"]]

    box = slice(None), slice(None)
    if args.window:
        ys, xs = np.nonzero(truth)
        box = (slice(max(0, ys.min() - 60), ys.max() + 60),
               slice(max(0, xs.min() - 60), xs.max() + 60))
    truth = truth[box]
    core = cv2.erode(truth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    print(f"labels: {int(truth.sum())} glyph px, method {str(data['method'])}, "
          f"scoring {len(frames)} frames")
    print(f"  {'source':22s} {'fgIoU':>7s} {'recall':>8s} {'prec':>7s} {'worst':>7s}")
    for entry in args.mask:
        name, _, spec = entry.partition("=")
        if not spec:
            name, spec = Path(entry).stem, entry
        masks = load_masks(spec, truth.shape if not args.window else data["mask"].shape)
        rows = []
        for i in frames:
            if i not in masks:
                continue
            m = masks[i]
            rows.append((i, score(m[box] if m.shape != truth.shape else m, truth, core)))
        if not rows:
            print(f"  {name:22s} no frames matched")
            continue
        a = np.array([r[1] for r in rows])
        print(f"  {name:22s} {np.median(a[:, 0]):7.3f} {np.median(a[:, 1]):7.1%} "
              f"{np.median(a[:, 2]):6.1%} {a[:, 0].min():7.3f}")
        if args.per_frame:
            for i, s in rows:
                print(f"      f{i:04d}  {s[0]:.3f}  {s[1]:.1%}  {s[2]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
