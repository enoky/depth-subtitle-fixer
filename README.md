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

### The CUDA OpenCV build

`setup.ps1` finishes by installing a CUDA-enabled OpenCV in place of the stock wheel, because
the mask chain runs on it - see [Where the time goes](#where-the-time-goes). You can install or
repair it on its own:

```powershell
.\scripts\install_opencv_cuda.ps1
```

It needs the **CUDA 13.x toolkit** (the wheel links against the CUDA runtime rather than
bundling it) and finds cuDNN inside the venv's own `torch/lib`, so there is no separate cuDNN
download. Without a toolkit the script says so and changes nothing; everything still works on
the CPU, just slower.

Two things worth knowing:

- `pip check` will report that `python-doctr` and `easyocr` want `opencv-python`. Expected and
  harmless - they want *an* OpenCV, and this is one.
- **Any later `pip install` can silently replace it.** Both of those packages depend on a stock
  OpenCV, and all the OpenCV distributions unpack into the same `cv2/` directory, so whichever
  lands last wins. Re-run the script afterwards. The scanner logs which backend it is using on
  every run, and `dsf` warns once when torch can see a GPU and OpenCV cannot, so this failure
  announces itself rather than just costing you half your speed in silence.

## Try it without your own footage

```bash
.venv/Scripts/python scripts/make_demo_clip.py
```

That writes `samples/demo_rgb.mp4` (1080p, burned-in subtitles plus two in-scene signs that
must keep their real depth) and `samples/demo_depth.mp4` (a plausible 10-bit depth map with
the depth wrecked and smeared over the subtitles).

The signs are photographed rather than pasted on - skewed off-axis, softened by the lens,
taking the scene's own light - because that is the only version of them worth shipping. Drawn
flat, as they were until recently, they are burned-in overlays in everything but colour, with
a chroma variance of exactly zero that nothing photographed has, and the clip cannot show
what it claims to: the appearance gate is asked to separate two things that are identical.
Painted properly they score 0.099 of contrast against the 0.12 floor and are turned away,
which is the behaviour the demo is there to demonstrate.

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
   something. This is the stage that separates an overlay from a shop sign, and the stage
   that throws away regions whose ink does not run level.
3. **Read** - the strongest confirmed frame goes to a recogniser, and the clip is only
   flagged if something reads as an actual word.

Stage 3 exists because stages 1 and 2 ask how a region *behaves* - does it hold still, hold
its colour, sit where subtitles sit - and a railing, a window grid or a run of compression
noise can answer yes to all of it. Asking what it *says* is what separates writing from
structure. The words are recorded in `scan_report.csv` either way, so a verdict you disagree
with can be argued with rather than guessed at.

Two rules catch what a recogniser answers when there is nothing to read. It still answers,
and regular structure is what it answers with: the bars of a fireplace grill come back as
`111`, a run of railings as `IIII`, each as long and as confident as a real word. **A word
that is one character repeated is not a word**, so those are discarded. And **overlay text
sits level**: subtitles and credits are composited square to the frame, while the straight
edges that get this far - railings, window frames, roof lines, reflections - are square to
nothing. The scan shears the ink in each region until its rows line up, and the angle that
takes is the angle the writing runs at; more than 8 degrees of it and the region is dropped
(`tilted_regions` in the report counts them).

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
- **Level text only**, by default. Turn *Require the text to sit level on screen* off for
  footage whose overlays are deliberately angled, or raise *Max tilt (deg)* from 8 rather
  than turning it off outright. Regions too small or too narrow to have a baseline - a lone
  glyph, a single stroke - are passed through unjudged rather than guessed at, and so are
  regions whose ink has no direction at all, like a solid block or a patch of noise.
- **Big enough to read** - *Min text height %* sets the shortest region the scan will
  consider, as a percentage of frame height, with an absolute floor of 14px underneath it
  for small clips. The default 2% works out at 43px on 4K, 22px on 1080p and 14px from 720p
  down, where the pixel floor takes over. A subtitle runs 4-6% and the small print at the
  end of a credit roll about 2%, so the default sits just under the smallest text worth
  finding; raise it towards 4% if only dialogue subtitles matter. This is the pipeline's own
  `min_text_height` gate raised for the scan alone - `dsf fix` keeps its 1.2%, because
  painting over small text costs far less than leaving the depth under it wrecked. It
  applies from the sweep onwards, so a clip whose only text is too small is dropped at the
  cheap stage rather than escalated and then discarded. 0 keeps the pipeline's own value.
- **Dry run** decides and reports without copying; *Skip clips already in the output* makes
  an interrupted scan resumable; every decision and its evidence lands in `scan_report.csv`.

If clips are still being copied that have no text in them, the report says which stage let
each one through. Raise *Min word length* or *Min word conf.* to tighten the reader, lower
*Max tilt* to insist on straighter text, raise *Min text height %* to ignore smaller
specks, and raise *Min text frames* / *Min coverage* to tighten the overlay gates.

If clips with real subtitles are being left behind, check `tilted_regions` and `min_text_px`
in the report first: a large tilt count on a clip you expected to be flagged means the level
check is reading the text as sloped and *Max tilt* wants raising, and `min_text_px` says how
tall a region had to be on that clip to be looked at at all.

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
- `--profile credits` - full frame, tracks vertical scrolling, and places the text just in
  front of the depth around it rather than on a constant plane (see `--brightness-mode`)
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
- `--depth-tol` / `--chroma-tol` - how far one glyph may sit from the depth, or the colour,
  that the rest of its line shares before it is treated as something else that happened to
  be inside the box. `--depth-tol` needs the depth map to be supplied to the detection pass,
  which `dsf fix`, `dsf preview` and the app all do; `dsf detect` takes an optional
  `--depth` for it. `0` disables either. See *Two things in one box*, below.

**Repairing the depth**

- `--brightness 0..1` - grey level for the painted text, mapped into the depth map's legal
  code range (10-bit limited = 64..940, so `0.92` ≈ code 870)
- `--brightness-mode absolute|relative` - a constant plane, or a fixed offset in front of
  the local depth. `absolute` suits subtitles, which sit at the bottom of the frame over
  whatever is furthest away; `credits` defaults to `relative` because a credit lands
  anywhere, and a constant plane over a face in the near field can end up *further forward
  than the corruption it is replacing*. On a credit across a subject's shoulder the text
  stood 240 codes proud of its surroundings untouched, 243 after an absolute repair, and 80
  after a relative one
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

   `T` is taken as pure white for light text, which is right for a subtitle and wrong for
   everything else - an amber credit at luma 0.77 over a shot at 0.15 divides by 0.85 where
   0.62 was wanted, so a fully opaque credit reports 0.76 and a quarter of the corruption it
   was meant to bury shows back through the letters. So the strength is read once per colour
   channel and the loudest kept: a bright saturated colour is bright because some channel is
   at its maximum, and amber is (255, 190, 80), so in red it *is* white. White text answers
   the same in all three channels, so a fade is untouched by this. It will not lift a
   reading that is under half showing, because a filmed shop sign is an opaque colour too
   and scoring it correctly is worth nothing when it must keep its real depth.

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
   **Which side is the writing.** Both signs of the residual almost always respond: text is
   thin, but so is every gap between its strokes, and outlined text answers one sign with its
   fill and the other with its rim. So the decision asks, strongest question first, whether
   the other sign responded at all; then which sign is *contained* by the other, since
   writing is figure and everything else - a drawn outline, or just the picture behind it -
   is ground; and only then which sign is the flatter colour, each measured against its own
   response so that two signs answering at different strengths are asked the same question.

   Getting the order right is the whole game. The first question used to be a ratio, "did
   this sign respond 1.6x harder", which is a far weaker thing than it sounds and was asked
   *first*: white-on-black-outline, the commonest subtitle style there is, lands at about
   0.62 against a 0.625 cut-off, so which way a line went depended on how tightly the
   detector had cropped it, and the mask was stamped onto the ring around the letters instead
   of the letters. Containment separated those two readings 1.00 to 0.18 at every crop.

   **Two things in one box.** The residual is measured on luma, so anything inside the
   detection box as bright as the glyphs answers it exactly as a glyph does - right area,
   right thickness, right strength - and gets masked with them. A background object the
   colour of the text is the case, and nothing in the luma channel can separate the two.

   Two things can. Burned-in text is one flat colour, and DepthCrafter pastes it onto one
   flat slab of wrong depth - that slab being the whole reason this tool exists. So each
   blob is asked whether it agrees with the blobs around it, on depth and on hue, and the
   ones that do not are dropped. Both are vetoes and both give ground rather than guess: the
   bar scales with how much the line's own letters disagree among themselves, marks too
   small for a blurred depth map to resolve sit the vote out, and if the majority does not
   agree in the first place the test stands down entirely and masks everything it found.
   That last one is what keeps a shot safe where the depth map never responded to the text -
   there the letters take the depth of the wall behind them, and a wall receding across the
   shot would otherwise cost the far end of every line.

   `--depth-tol` is set against clips with the glyphs labelled, so every pixel it takes is
   known to be text or not. On a credit with a lens flare and a strand of hair inside its
   box, the shipped 0.15 removes 2725 px of background and touches no glyph; 0.10 catches
   more background (3950) but bites 1612 px out of the letters, and 0.06 removes twice as
   much text as background. It sits where the text stops being touched rather than where the
   most background is caught: a missed intruder leaves a small wrong patch in the depth,
   while an eaten letter shows the corruption through the writing, which is the artefact
   this tool exists to remove. Turn it down if something behind the text is surviving, and
   check a preview when you do.

   The colour test is the weaker of the two by construction - the hard cases are the ones
   where the object matches the text's hue as well as its brightness - but it needs no depth
   map.
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

When the depth agreement test is in use the depth map is decoded twice, once alongside
detection and once again for compositing, which cost 12% of the detection rate on a 1080p
clip. Holding a clip's depth from the first pass to hand back at the second is exactly the
unbounded buffer this design exists to avoid, so it is decoded again instead.

Decoding runs a chunk ahead of detection on its own thread, because otherwise the two take
turns: ffmpeg idles while the GPU works and the GPU idles while ffmpeg decodes. Measured on
a 1080p clip the GPU was below 20% utilisation for 65% of the wall clock. The read-ahead
depth is sized against the frame size rather than fixed, so the streaming promise holds at
8K as well as at 720p.

## Where the time goes

Detection ran on the GPU from the start. The mask chain behind it - composing glyph patches
into a full-frame alpha, the byte conversions, the temporal median, and the prior's
threshold-and-or over its twenty-one frame window - did not, and on a 1080p frame it was 33 of
the 99 ms a frame cost while the card sat idle. `src/dsf/accel.py` moves that chain onto the
GPU through OpenCV's CUDA module, keeping it resident there from the patches going up to the
finished mask coming down, so a frame crosses the bus once instead of at every step.

Buffers are pooled, and that is most of why it is faster: a `cv2.cuda.min` that allocates its
own destination costs 0.335 ms at 1080p against 0.0068 ms into a preallocated one. The kernels
were never the expense, `cudaMalloc` was.

Measured on a 1920x1080 clip, an RTX 5080, `--profile both`:

| | CPU | CUDA |
|---|---|---|
| prior threshold+or, 21-frame window | 42.5 ms | 0.33 ms |
| `scale_by` (shape x level) | 10.3 ms | 0.017 ms |
| `to_u8` | 4.03 ms | 0.034 ms |
| temporal median of 3 | 0.98 ms | 0.13 ms |
| **whole mask chain, per frame** | **126 ms** | **60 ms** |

`scripts/bench_scan.py` reproduces that table on your own footage, and `DSF_ACCEL=cpu` forces
the numpy path so the two can be compared directly.

**Where it does not help.** A folder triage scan barely moves - about 1.03x on the same
footage. The sweep that decides whether a clip is worth a closer look is detection and nothing
else, it never builds a mask, and a clip that flags stops the moment it has enough evidence -
so the mask chain runs on a dozen frames per clip while the detector runs on sixty. The gain
lands on work that masks every frame: `dsf fix` on the demo pair goes 15.5s to 12.4s, held
back from the full 2x by the heal-and-composite step and the x265 encode, which are still CPU.

What is left in a GPU frame is mostly not ours: docTR's detection and its post-processing are
57% of it, and the ffmpeg decode another 13%.

Two things were measured and deliberately **not** done:

- **Stroke extraction stays on the CPU.** It works on crops a few hundred pixels across, where
  kernel-launch overhead swamps the work: a 3x3 erode on a 70x900 crop is 0.006 ms on the CPU
  and 0.064 ms on the GPU. `medianBlur` is a wash at typical crop sizes, and `distanceTransform`
  has no CUDA implementation at all.
- **NVDEC decoding was tried and rejected.** `cv2.cudacodec` decodes 1080p H.264 at 14.8 ms a
  frame against the ffmpeg pipe's 8.7, opening a reader costs 328 ms against ffmpeg's 97, and on
  the sweep's own pattern - eight short scattered windows - it was 2.8x slower. It also cannot
  decode 4:4:4 at all, which is what `make_demo_clip.py` writes, and its YUV-to-RGB conversion
  disagrees with swscale on 0.48% of pixels, which would feed the detector.

## Notes

- `python-doctr` depends on `opencv-python` and `easyocr` on `opencv-python-headless`; they
  share one `cv2/` install path with the CUDA build, so whichever installs last wins.
  `setup.ps1` puts the CUDA one last - see [The CUDA OpenCV build](#the-cuda-opencv-build).
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
