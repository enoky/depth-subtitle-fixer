# depth-subtitle-fixer

Fix DepthCrafter depth maps by detecting & masking embedded subtitles/credits in RGB video,
then overlaying the text onto the depth map in adjustable grayscale.

## The problem

DepthCrafter estimates depth from RGB frames. It has no way to know that burned-in subtitles
or overlaid intro credits are not part of the scene, so it invents depth for the glyphs *and*
bleeds a smeared halo into the depth around them. After stereo conversion that reads as text
flickering at random depths.

This tool repairs that. It detects the overlaid text in the RGB clip with open-source ML
models, extracts a **glyph-level** mask (the strokes, not the bounding box), heals the
corrupted depth around each glyph, and stamps the text at a grey level you choose - so
subtitles sit on one clean, constant depth plane.

## Install

Needs Python 3.10+ and ffmpeg on PATH (or it falls back to the `imageio-ffmpeg` binary).

```powershell
.\scripts\setup.ps1
```

That creates a project-local `.venv`, installs PyTorch from the CUDA 13.0 index (required for
RTX 50-series / Blackwell `sm_120`), installs everything else, and prints the detected GPU.
On Linux/macOS use `scripts/setup.sh`.

Model weights (~180 MB, docTR + CRAFT) download on first run into `./models/`.

## Try it without your own footage

```bash
.venv/Scripts/python scripts/make_demo_clip.py
```

That writes `samples/demo_rgb.mp4` (1080p, burned-in subtitles plus two in-scene signs that
must keep their real depth) and `samples/demo_depth.mp4` (a plausible 10-bit depth map with
the depth wrecked and smeared over the subtitles).

## Use

```bash
dsf fix --rgb clip.mp4 --depth clip_depth.mp4 --out clip_depth_fixed.mp4 --brightness 0.92
```

Check a few frames before committing to a full render:

```bash
dsf preview --rgb clip.mp4 --depth clip_depth.mp4 --frames 120,450,900 --out-dir previews
```

Each preview is a 2x2 sheet: source + detections | glyph mask | depth before | depth after.

Detect once, then re-render at different brightness in seconds:

```bash
dsf detect --rgb clip.mp4 --out-mask masks.mkv
```
```bash
dsf render --depth clip_depth.mp4 --mask masks.mkv --out fixed.mp4 --brightness 0.85
```

Or tune interactively with a frame scrubber and live sliders:

```bash
dsf ui
```

## Commands

| Command | What it does |
|---|---|
| `dsf probe <video>` | resolution, pix_fmt, bit depth, fps, frame count, colour tags |
| `dsf detect` | detect text, write a mask cache |
| `dsf render` | composite a cached mask onto the depth map |
| `dsf fix` | detect + composite in one streaming pass |
| `dsf preview` | contact sheets for chosen frames |
| `dsf ui` | local Gradio app |

## Key options

**Profiles** set sensible defaults; any explicit flag overrides them.

- `--profile subtitles` (default) - static text in the lower band
- `--profile credits` - full frame, tracks vertical scrolling
- `--profile both`

**Finding the text**

- `--detectors doctr,easyocr` - docTR (DBNet, segmentation-based) and EasyOCR (CRAFT,
  scene-text). Results are unioned; either works alone.
- `--roi bottom:0.30 | top:0.20 | full | x0,y0,x1,y1` - normalised region to search
- `--scene-text keep|mask` - `keep` (default) leaves filmed text such as shop signs and
  licence plates with its real depth; `mask` masks everything found
- `--detect-every N` - detect every Nth frame and propagate between (static text only)

**Repairing the depth**

- `--brightness 0..1` - grey level for the painted text, mapped into the depth map's legal
  code range (10-bit limited = 64..940, so `0.92` ≈ code 870)
- `--brightness-mode absolute|relative` - a constant plane, or a fixed offset in front of
  the local depth
- `--heal edt|none` and `--heal-scope glyph|region` - repair the smear around the strokes
  (default) or flatten the whole detection box
- `--dilate N` / `--feather S` - grow and soften the mask edge
- `--lossless` - bit-exact output outside the mask, much larger files

## How it works

1. **Detect** - docTR DBNet and EasyOCR CRAFT run on the RGB frames; boxes are unioned.
2. **Filter** - detections must sit in the ROI, be a plausible text size, be high-contrast
   and flat-coloured, and persist across several frames. This is what keeps in-scene text
   intact.
3. **Extract strokes** - inside each box, a 3-class split separates outline / background /
   glyph, and connected components are filtered by area, stroke width and whether they carry
   a dark rim. The mask then grows from the glyph core into that rim (`--rim-expand`): the
   outline is part of the burned-in overlay and its depth is corrupted too. The result is a
   *soft* alpha - hard edges ring after stereo warping.
4. **Smooth** - a temporal median (or max, for credits) across a small window kills detector
   flicker.
5. **Composite** - heal the corrupted depth in a halo around the strokes from the nearest
   valid depth, then alpha-blend the glyphs at the requested code value.

Everything streams frame by frame, so clip length is limited by disk, not RAM. The depth
luma plane is decoded in its source pixel format and written back untouched outside the
mask - no colour-range rescale, no 8-bit round trip.

## Notes

- `python-doctr` depends on `opencv-python` and `easyocr` on `opencv-python-headless`; they
  share one `cv2/` install path. `setup.ps1` force-installs the headless build last. If you
  later run a plain `pip install`, re-run that line.
- Depth maps at a different resolution than the source clip are handled - masks are resampled
  with `INTER_AREA` so stroke anti-aliasing survives.

## Tests

```bash
.venv/Scripts/python -m pytest
```

Add `--runslow` to include the end-to-end tests, which need the model weights. All tests
build their own synthetic media; no sample files are required.
