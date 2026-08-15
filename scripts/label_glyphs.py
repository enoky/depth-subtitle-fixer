"""Build a ground-truth glyph mask for a clip, so mask quality can be measured rather than eyed.

Every claim about how well the strokes come out needs something to be measured against, and
the only honest reference is the letters themselves - not the detection box, not a threshold
on the picture, and not the tool's own output. This builds one without anybody annotating
anything, by exploiting the fact that burned-in text holds still while a film does not.

Two ways to do that, and which one works is a property of the shot:

*Persistence* asks, per pixel, how often it is text-coloured across the frames the credit is
up. The scene moves under static writing, so the letters answer every time and the picture
does not. This is the method for a moving shot.

*Differencing* asks the same thing over the frames the credit is up, and subtracts the answer
over frames where it is absent. This is the method for a shot that holds still, where
persistence cannot work at all: a static wall the colour of the credit is exactly as
persistent as the credit.

Neither is safe unattended, which is why `--sheet` is written by default and worth looking at.
On one clip persistence swept in a sunlit window - as static as static text, and as amber -
and the resulting numbers were wrong for an hour before anyone looked at the picture. On
another, lowering the colour threshold far enough to catch a dim caption let the off-frames
register too, and the differencing punched holes through the letters instead.

    .venv/Scripts/python scripts/label_glyphs.py --rgb clip.mp4 --out labels.npz
    .venv/Scripts/python scripts/label_glyphs.py --rgb clip.mp4 --out labels.npz \
        --on 26-61 --off 0-14,72-106 --min-bright 210 --min-sat 0.55
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

from dsf.media import probe, read_rgb  # noqa: E402


def parse_ranges(spec: str) -> list[int]:
    """'26-61' or '0-14,72-106' into a list of frame indices."""
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


def text_coloured(frame: np.ndarray, min_bright: int, min_sat: float,
                  warm: bool = True) -> np.ndarray:
    """Pixels bright and saturated enough to be the writing rather than the shot behind it.

    Saturation as well as brightness because a credit is usually a colour, and brightness
    alone selects every lit thing in the frame. *warm* additionally requires R > G > B, which
    is what an amber title is; turn it off for white or cool text.
    """
    a = frame.astype(np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    hit = (mx > min_bright) & (sat > min_sat)
    if warm:
        hit &= (a[..., 0] > a[..., 1]) & (a[..., 1] > a[..., 2])
    return hit.astype(np.float32)


def find_on_off(row_counts: np.ndarray) -> tuple[list[int], list[int]]:
    """Which frames carry the text and which do not, from text-coloured pixels per row per frame.

    Each row's own median over time is subtracted before anything is counted, which is the
    only way this works on a shot whose scenery is the credit's colour. On a warm interior
    the whole frame answers the colour test: 64,000 pixels of it, against the thousand the
    credit adds. A count, a ratio, or a threshold on that is measuring the room. What the
    room does *not* do is change, so removing each row's static part leaves the writing.

    That also survives text being up for most of the clip, where the median is the text and
    the residual goes the other way: the frames carrying it still come out at the top of the
    range once it is normalised, because being at the median is as high as those frames get.

    Deliberately crude - it only has to bracket the range, and `--on` overrides it whenever a
    clip defeats it.
    """
    if row_counts.size == 0:
        return [], []
    residual = row_counts - np.median(row_counts, axis=0, keepdims=True)
    signal = residual.sum(axis=1)
    lo, hi = float(signal.min()), float(signal.max())
    if hi - lo < 1.0:
        return [], []
    norm = (signal - lo) / (hi - lo)
    return ([i for i, v in enumerate(norm) if v > 0.5],
            [i for i, v in enumerate(norm) if v < 0.2])


def letter_components(mask: np.ndarray, min_area: int, min_h: int, max_h: int,
                      max_w: int) -> tuple[np.ndarray, int]:
    """Keep the blobs shaped like letters, drop everything else.

    This is the step that makes the method trustworthy, and leaving it out is what let a
    sunlit window into a "glyph" mask. Scenery that survives the colour test is the wrong
    size or the wrong shape for a letter: a window is 163 px tall where the letters are 58,
    and a panel is a slab where a letter is a stroke.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(mask.shape, bool)
    kept = 0
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        if area < min_area or not (min_h <= h <= max_h) or w > max_w:
            continue
        keep |= (lab == label)
        kept += 1
    return keep, kept


#: How far apart two blobs' rows may be, as a multiple of a letter's height, and still count
#: as the same block of text. Lines of a credit are set about 1.4 letter-heights apart, so
#: this keeps a two- or three-line block together while dropping anything isolated further
#: off - which is what letter-sized scenery at a distance looks like.
_LINE_GAP = 2.0


def in_text_band(mask: np.ndarray) -> np.ndarray:
    """Keep the block of text and drop letter-shaped things sitting away from it.

    Text is set on lines, and a credit's lines sit close together. Grouping the blobs by how
    far apart their rows are and keeping the group with the most ink in it handles a
    multi-line credit without a band having to be guessed, and drops a lit shape across the
    frame that happened to survive the colour and size tests.
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask
    rows = [(float(cent[l][1]), int(stats[l, cv2.CC_STAT_HEIGHT]),
             int(stats[l, cv2.CC_STAT_AREA]), l) for l in range(1, n)]
    rows.sort()
    gap = _LINE_GAP * float(np.median([h for _, h, _, _ in rows]))

    groups, current = [], [rows[0]]
    for item in rows[1:]:
        if item[0] - current[-1][0] > gap:
            groups.append(current)
            current = [item]
        else:
            current.append(item)
    groups.append(current)

    best = max(groups, key=lambda g: sum(a for _, _, a, _ in g))
    keep = np.zeros(mask.shape, bool)
    for _, _, _, label in best:
        keep |= (lab == label)
    return keep


def contact_sheet(frame: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """The picture to look at before believing any number this produced."""
    ys, xs = np.nonzero(mask)
    if not ys.size:
        return frame
    y0, y1 = max(0, ys.min() - 45), min(frame.shape[0], ys.max() + 45)
    x0, x1 = max(0, xs.min() - 60), min(frame.shape[1], xs.max() + 60)
    crop, sub = frame[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    over = crop.copy()
    over[sub] = (0.25 * over[sub] + 0.75 * np.array([0, 255, 255])).astype(np.uint8)
    gap = np.zeros((4, crop.shape[1], 3), np.uint8)
    sheet = np.vstack([crop, gap, over])
    cv2.putText(sheet, f"{method}: cyan is what was labelled a glyph", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return sheet


def build(rgb_path: str, args) -> dict:
    """Two passes over the clip: find where the text is in time, then average it out of the shot.

    Two rather than one so that nothing bigger than a row-count profile is ever held. A
    couple of hundred 1080p frames is half a gigabyte, and this has to run on whole clips.
    """
    info = probe(rgb_path)
    warm = not args.any_colour

    profile, total = [], 0
    for frame in read_rgb(rgb_path, info=info):
        profile.append(text_coloured(frame, args.min_bright, args.min_sat, warm).sum(axis=1))
        total += 1
    if not total:
        raise SystemExit(f"no frames read from {rgb_path}")
    row_counts = np.asarray(profile, dtype=np.float32)

    on = parse_ranges(args.on) if args.on else None
    off = parse_ranges(args.off) if args.off else None
    if on is None:
        on, auto_off = find_on_off(row_counts)
        off = off if off is not None else auto_off
    elif off is None:
        off = [i for i in range(total) if i not in set(on)]
    on = sorted(i for i in on if 0 <= i < total)
    off = sorted(i for i in off if 0 <= i < total and i not in set(on))
    if len(on) < 3:
        raise SystemExit("could not find frames carrying text; pass --on explicitly")

    # Differencing whenever there are frames without the text to difference against - it is
    # strictly the better evidence, since it cancels anything the shot does in both. Only a
    # clip whose text never leaves falls back to persistence.
    differencing = len(off) >= 3 and not args.persistence
    method = "differencing" if differencing else "persistence"

    on_set, off_set = set(on), set(off)
    on_sum = off_sum = None
    sample = None
    middle = on[len(on) // 2]
    for i, frame in enumerate(read_rgb(rgb_path, info=info)):
        if i == middle:
            sample = frame.copy()
        if i in on_set:
            lit = text_coloured(frame, args.min_bright, args.min_sat, warm)
            on_sum = lit if on_sum is None else on_sum + lit
        elif differencing and i in off_set:
            lit = text_coloured(frame, args.min_bright, args.min_sat, warm)
            off_sum = lit if off_sum is None else off_sum + lit

    # Two questions rather than one, because a single threshold on the difference conflates
    # them. A glyph is lit essentially every frame the text is up - that is `min_on`, and it
    # is what separates the writing from anything that merely flickers warm. And it is lit
    # more often than when the text is gone - that is `min_gain`, and it is what separates
    # the writing from scenery that is permanently its colour. A warm interior can sit at
    # 0.52 in the frames without the credit, so a difference of 0.48 is a strong signal and
    # would fail any threshold set high enough to mean something on its own.
    on_mean = on_sum / len(on)
    candidate = on_mean >= args.min_on
    if differencing and off_sum is not None:
        candidate &= (on_mean - off_sum / len(off)) >= args.min_gain
    kept, n_comp = letter_components(candidate, args.min_area, args.min_height,
                                     args.max_height, args.max_width)
    kept = in_text_band(kept)
    kept, n_comp = letter_components(kept, args.min_area, args.min_height,
                                     args.max_height, args.max_width)
    return {"mask": kept, "method": method, "on": on, "off": off,
            "components": n_comp, "frame": sample, "info": info}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb", required=True, help="the clip carrying the burned-in text")
    p.add_argument("--out", required=True, help="where to write the labels (.npz)")
    p.add_argument("--sheet", help="contact sheet to write (default: alongside --out)")
    p.add_argument("--on", help="frames carrying the text, e.g. 26-61 (default: found)")
    p.add_argument("--off", help="frames without it, e.g. 0-14,72-106 (default: found)")
    p.add_argument("--persistence", action="store_true",
                   help="force persistence even when off-frames exist")
    p.add_argument("--min-bright", type=int, default=210, help="0-255 (default 210)")
    p.add_argument("--min-sat", type=float, default=0.55, help="0-1 (default 0.55)")
    p.add_argument("--any-colour", action="store_true",
                   help="do not require R>G>B; use for white or cool text")
    p.add_argument("--min-on", type=float, default=0.7,
                   help="how much of the time a glyph pixel must be lit while the text is "
                        "up (default 0.7). Not higher: the frames found as carrying the "
                        "text include the fade at each end, where the letters are real but "
                        "not yet at full strength, and 0.9 loses the end of a line")
    p.add_argument("--min-gain", type=float, default=0.3,
                   help="how much more often than while it is absent (default 0.3); only "
                        "used when differencing")
    p.add_argument("--min-area", type=int, default=40)
    p.add_argument("--min-height", type=int, default=8)
    p.add_argument("--max-height", type=int, default=70)
    p.add_argument("--max-width", type=int, default=140)
    args = p.parse_args(argv)

    got = build(args.rgb, args)
    mask = got["mask"]
    if not mask.any():
        print("labelled nothing - try lowering --min-bright/--min-sat or passing --on",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mask=mask, method=got["method"],
                        on=np.array(got["on"]), off=np.array(got["off"]))
    sheet_path = Path(args.sheet) if args.sheet else out.with_suffix(".png")
    cv2.imwrite(str(sheet_path),
                cv2.cvtColor(contact_sheet(got["frame"], mask, got["method"]),
                             cv2.COLOR_RGB2BGR))

    ys, xs = np.nonzero(mask)
    where = f"{len(got['on'])} frames with text"
    if got["method"] == "differencing":
        where += f" against {len(got['off'])} without"
    print(f"{got['method']} over {where}")
    print(f"  {int(mask.sum())} px in {got['components']} letter-shaped components, "
          f"rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()}")
    print(f"  wrote {out} and {sheet_path}")
    print("  look at the sheet before trusting anything measured against this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
