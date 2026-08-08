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

Add `--scan-demo` for a small folder of clips to point the scanner (below) at:
`samples/scan_demo/rgb` holds one subtitled clip, one with scrolling credits and one
carrying nothing but signage the camera photographed, with matching `_depth` maps alongside.

## Use

```bash
dsf fix --rgb clip.mp4 --depth clip_depth.mp4 --out clip_depth_fixed.mp4 --brightness 0.92
```

Folders of frames work anywhere a video does - DepthCrafter is often driven and reviewed as
stills, so a PNG/TIFF sequence is a first-class input rather than a conversion step. Give an
output path with no extension and the frames come back with their original filenames, bit
depth and channel layout, and every pixel the mask did not touch is byte-identical:

```bash
dsf fix --rgb ./rgb_png --depth ./depth_png --out ./depth_fixed --profile credits
```

16-bit PNG depth survives intact - sequences are read straight off disk rather than through
a video pipeline that would force an 8-bit or YUV round trip.

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

On Windows, `RUN_dsf.bat` in the project root does the same from a double-click - it activates
the `.venv` for you and forwards any flags, so `RUN_dsf.bat --port 7861` works too.

The app exposes **every** setting the command line does - the everyday ones up front, the
rest under *Detection — advanced*, *Glyph extraction — advanced* and *Encoding*. A test
compares the two lists, so a flag cannot exist in one and go missing from the other.
Switching profile resets the controls to that preset's defaults, and anything you change
afterwards applies on top, exactly as an explicit flag overrides a profile on the CLI.

The **Browse…** buttons open a normal OS file dialog and hand back the path - nothing is
copied, so picking a two-hour 4K master costs nothing. Choosing any frame inside a folder
selects that folder, since a file dialog cannot pick one. The output path fills itself in
next to the source when you hit Load. Browse needs `tkinter` and is disabled behind
`--share`, since the dialog would open on the machine running the server rather than the
viewer's; paths can always be pasted instead.

## Sorting a folder of clips

Before fixing anything you usually need to know *which* clips need fixing. `scan_for_text.py`
is a separate tkinter app that walks a folder of RGB clips, decides which carry burned-in
subtitles or overlaid credits, and copies those into `rgb_with_text/` under an output folder
you choose. Give it a depth folder too and each flagged clip's depth map - same name with
`_depth` on the stem - is copied into `depth_with_text/` beside it.

```bash
.venv/Scripts/python scripts/scan_for_text.py
```

`RUN_scan.bat` does the same from a double-click. Nothing is re-encoded; files are copied
byte for byte, and an existing destination is reported rather than overwritten.

It detects nothing of its own - it runs the same pipeline `dsf fix` does, with
`--scene-text keep`, and reads the detections that survive the gates. So a licence plate, a
shop sign or a T-shirt logo does not flag a clip, for exactly the reasons described under
*How it works* below.

Each clip goes through up to three stages, and only pays for a stage if it got past the one
before. That ordering is what makes a folder of 2500 clips finishable.

1. **Sweep** - detection alone, on eight short clusters of consecutive frames spread across
   the clip, at a reduced detector input size. No stroke extraction, no temporal filter, no
   prior. A clip with nothing text-shaped in it stops here, which is most of a folder.
2. **Confirm** - the full pipeline, on windows centred where the sweep actually found
   something. This is the stage that separates an overlay from a shop sign.
3. **Read** - the strongest confirmed frame goes to a recogniser, and the clip is only
   flagged if something reads as an actual word.

Stage 3 exists because stages 1 and 2 ask how a region *behaves* - does it hold still, hold
its colour, sit where subtitles sit - and a railing, a window grid or a run of compression
noise can answer yes to all of it. Asking what it *says* is what separates writing from
structure. The words are recorded in `scan_report.csv` either way, so a verdict you disagree
with can be argued with rather than guessed at.

Several clips are scanned at once, sharing one set of models. Each worker spends most of its
life waiting - on ffmpeg starting up, on decoding, on the disk - so a second and third fill
the gaps the first leaves rather than competing for the GPU. Past three the GPU is saturated
and more workers only cost memory. Results then arrive as they finish, so the log and the
report are in completion order rather than the folder's.

Measured on a mixed folder of eight 1080p clips of 480 frames: **112s to 19.5s, 5.7x**, with
every verdict unchanged except a railing-and-window-grid clip that the old scan copied and
this one correctly leaves behind. Staging accounts for most of it, concurrency for about
1.5x on top. The gain scales with clip length, because it comes from not reading frames the
old scan read blindly.

- **Profile** - the same `subtitles` / `credits` / `both` presets, so a scan for subtitles
  only looks in the lower band while `credits` follows text scrolling up the full frame.
- **Verdict** - a clip is flagged once it accumulates enough frames carrying real overlay
  text (default 6), including a run of consecutive ones (default 3), each covering more of
  the frame than a stray speck would. One box on one frame is not a subtitle. The scan
  stops on a clip the moment the bar is met.
- **Sampling** - windows of *consecutive* frames, on purpose: the persistence gate decides
  by looking at a detection's neighbours in time, and it is the gate that spares filmed
  text. The sweep uses the same number of sample points as the confirm pass would, so brief
  text is no likelier to be missed than before; raise *Sweep clusters* for long clips whose
  titles appear only briefly. Tick *Scan every frame* for an exhaustive pass, which skips
  the sweep entirely.
- **Reading is Latin-script only** (docTR `crnn_vgg16_bn`, ~63 MB on first run). Turn
  *Require the text to read as words* off for footage subtitled in Chinese, Japanese,
  Korean, Cyrillic or Arabic - it would reject every clip otherwise. Clips that pass the
  overlay gates but fail to read are reported as `rejected` rather than silently dropped,
  so they are easy to review.
- **Dry run** decides and reports without copying; *Skip clips already in the output* makes
  an interrupted scan resumable; every decision and its evidence lands in `scan_report.csv`.

If clips are still being copied that have no text in them, the report says which stage let
each one through. Raise *Min word length* or *Min word conf.* to tighten the reader, and
*Min text frames* / *Min coverage* to tighten the overlay gates.

## Commands

| Command | What it does |
|---|---|
| `dsf probe <video\|folder>` | resolution, pix_fmt, bit depth, fps, frame count, colour tags |
| `dsf detect` | detect text, write a mask cache |
| `dsf render` | composite a cached mask onto the depth map |
| `dsf fix` | detect + composite in one streaming pass |
| `dsf preview` | contact sheets for chosen frames |
| `dsf ui` | local Gradio app |
| `scripts/scan_for_text.py` | tkinter app: sort a folder of clips into those with text and those without |

## Key options

**Profiles** set sensible defaults; any explicit flag overrides them.

- `--profile subtitles` (default) - static text in the lower band
- `--profile credits` - full frame, tracks vertical scrolling
- `--profile both`

**Finding the text**

- `--detectors doctr` (default) - docTR's DBNet, segmentation-based. Adding `easyocr`
  unions in CRAFT, which reads stylised scene text better but costs roughly double the
  detection time; on the clips measured it found 99.9% of the same mask without it, so it
  is worth turning on for a title card that needs it rather than for every render.
- `--roi bottom:0.30 | top:0.20 | full | x0,y0,x1,y1` - normalised region to search
- `--scene-text keep|mask` - `keep` (default) leaves filmed text such as shop signs and
  licence plates with its real depth; `mask` masks everything found
- `--detect-every N` - detect every Nth frame and propagate between (static text only)
- `--min-response` - minimum stroke contrast for a box to count as text at all
- `--background-scale` - the background window as a fraction of text height. It must span
  roughly a whole letter; raise it if heavy or bold text comes back hollow
- `--luma-tol` - colour tolerance when following a glyph's outline (the fill is found by
  opacity, which is what makes fades work)
- `--polarity auto|light|dark` - override if a clip's text is consistently mis-read

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
3. **Extract strokes** - inside each box, a median over a window wider than a stroke
   estimates the picture *without* the writing, and the text is read off as the difference.
   Because a pixel where text of colour `T` covers background `B` at opacity `a` reads back
   as `a*T + (1-a)*B`, that difference is exactly `a*(T-B)` - so dividing by `(T-B)` returns
   the text's opacity directly, whatever the shot is doing behind it. Components are then
   filtered by area and by a stroke width that scales with the text size, and the mask grows
   from the glyph core into its outline (`--rim-expand`), since the outline is part of the
   overlay and its depth is corrupted too.

   Two failures this avoids. Thresholding raw luminance does not survive a detection box
   that straddles a lighting boundary - the threshold ends up describing the *background*
   rather than separating text from it, and whichever word sits over the brighter half is
   silently dropped. And testing whether each pixel matches the colour of the other strokes
   only holds while the text is opaque: the moment a credit fades, every pixel is part
   background, one glyph reads as many shades, and the mask comes back shredded.

   Working in opacity also gives the right answer for free - a credit at 30% opacity yields
   a 30% mask, so the depth is pushed 30% of the way and the text eases in instead of
   snapping. Below roughly 25% opacity a faded credit is quieter than the scene's own detail
   at stroke scale, and no single-frame method can recover it; `--temporal max` borrows from
   neighbouring frames there.
4. **Smooth** - a temporal median (or max, for credits) across a small window kills detector
   flicker. It is applied to the stroke *shape* only, and each frame is then scaled by the
   opacity it measured for itself. Smoothing the finished mask instead would drag a fading
   credit up to its neighbours' strength, so the frame after a fade-out ended - nothing
   detected on it at all - would still get a near-solid mask stamped into depth that was
   never corrupted.
5. **Composite** - heal the corrupted depth in a halo around the strokes from the nearest
   valid depth, then alpha-blend the glyphs at the requested code value.

Everything streams frame by frame, so clip length is limited by disk, not RAM. The depth
luma plane is decoded in its source pixel format and written back untouched outside the
mask - no colour-range rescale, no 8-bit round trip.

Decoding runs a chunk ahead of detection on its own thread, because otherwise the two take
turns: ffmpeg idles while the GPU works and the GPU idles while ffmpeg decodes. Measured on
a 1080p clip the GPU was below 20% utilisation for 65% of the wall clock. The read-ahead
depth is sized against the frame size rather than fixed, so the streaming promise holds at
8K as well as at 720p.

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

Add `--runslow` to include the end-to-end tests, which need the model weights. The tests
build their own synthetic media, so nothing external is required.

`tests/test_real_footage.py` additionally runs against a real DepthCrafter pair - a title
card fading in and out over a moving crowd - and is skipped unless frames are available. To
enable it, supply a folder containing `rgb_png/` and `depth_png/` either at
`samples/real_footage/` (`samples/` is git-ignored) or via an environment variable:

```bash
DSF_REAL_FOOTAGE=/path/to/footage .venv/Scripts/python -m pytest tests/test_real_footage.py --runslow
```
