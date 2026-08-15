# Hi-SAM as a stroke source

A plan, not a decision. Written after evaluating [Hi-SAM](https://github.com/ymy-k/Hi-SAM)
(SAM-TS-L, TextSeg weights) against the extractor on two real credits, and holding the
measurements up next to what ships today. Phase 0's validation gate has passed and Phases 1
to 4 are built.

## Why this is worth considering at all

The extractor reads strokes off a luma residual: it estimates the picture without the
writing and takes the difference. That fails, by construction, where the text is the
brightness of what it sits on - and on a credit crossing a subject's hair and shoulder it
does not fail gracefully.

Measured on two clips against the labels `scripts/label_glyphs.py` writes, scored by
`scripts/score_masks.py`, with Hi-SAM run full-frame as Phase 3 would run it:

**V1-0005, "EMILY BEECHAM", 28 frames** - a credit crossing hair and shoulder

| | fgIoU (median) | recall | precision | worst frame | speed |
|---|---|---|---|---|---|
| luma (ships today) | 0.755 | 83.5% | **94.5%** | 0.226 | 0.074 s/frame |
| luma + `--depth-strokes` | 0.762 | 87.6% | 90.0% | 0.406 | - |
| Hi-SAM SAM-TS-L | **0.809** | **99.3%** | 82.3% | **0.786** | 0.25 s/frame |

**V1-0007, "WITH / DAVID CORENSWET", 36 frames** - a near-static cockpit interior

| | fgIoU (median) | recall | precision | worst frame | speed |
|---|---|---|---|---|---|
| luma | 0.784 | 89.1% | **91.9%** | 0.480 | - |
| luma + `--depth-strokes` | 0.807 | 93.5% | 89.0% | 0.506 | - |
| Hi-SAM SAM-TS-L | **0.814** | **99.8%** | 82.3% | **0.746** | 0.25 s/frame |

**On the median this is a modest win, and on the second clip a marginal one** - 0.814 against
0.807 for `--depth-strokes`. What is bought is the tail rather than the average: the worst
frame goes from 0.226 to 0.786 on the first clip and from 0.480 to 0.746 on the second. If
the frames where a credit crosses hair matter more than the frames where it does not, that is
worth paying for. If they do not, `--depth-strokes` already gets most of the way for nothing.

Three things repeat across both clips, measured identically:

- **Recall of 99.3% and 99.8%.** It finds essentially every glyph pixel.
- **Precision of 82.3% on both, to the decimal.** A consistent over-fill, not noise.
- **It never collapses.** The luma path's worst frame is less than a third of its median on
  the first clip; Hi-SAM's worst is within 3% of its own median on both.

Full-frame inference costs the same as a crop - 0.25 s/frame either way, because SAM resizes
to 1024 internally, so the four-fold increase in pixels is absorbed. That makes Phase 3's
"once per frame, full-frame" the cheap option as well as the clean one.

An earlier version of this document reported 0.828 and 0.872 for Hi-SAM against 0.717 and
0.785, a margin of 0.111 and 0.062. Those were scored against a hand-built truth mask that
contained a sunlit window Hi-SAM had correctly ignored, which flattered it. The corrected
margins are 0.054 and 0.007. The conclusion holds; the size of it did not.

Both methods find the small dim "WITH" caption on the second clip, so that class of text is
not a differentiator. Hi-SAM covers 677 px of it against luma's 469, which is the same
over-filling seen everywhere else rather than a difference in what was found.

Where the luma path collapses is where a credit crosses something its own brightness, and
that is the same failure the polarity fix, the depth agreement veto and `--depth-strokes`
were all written to work around. A model trained on stroke masks does not have it: this is
the difference between patching a heuristic where it breaks and using something that does not
break there.

Timings are an RTX 5080 with a cu130 torch build. On CPU the same inference is 6.0 s/frame
with identical masks, so a GPU is not optional. At 0.25 s/frame a 129-frame render goes from
about ten seconds to thirty-five - fine offline, and not fine for a 2500-clip scan.

## The architectural fit

`analyse_crop` already returns two separable things, and they are separable for a reason:

- `shape` - the stroke geometry, normalised so a faded credit reads the same as a solid one
- `level` - how opaque the text actually is

Hi-SAM produces a binary stroke mask and **no opacity whatever**. So it maps onto exactly one
of them:

> **Hi-SAM replaces `shape`. The luma opacity model keeps producing `level`.**

This is what makes the job tractable rather than a rewrite. Fades keep working, because
`level` still comes from the residual, which is the thing that measures opacity correctly.
Everything downstream - the component filter, both agreement vetoes, the temporal stack,
`compose_levels`, the compositor - is untouched, because all of it consumes `shape` and
`level` rather than pixels.

A segmentation confidence is *not* an opacity. A confidently-detected 30%-opacity credit
scores high, and stamping it at full strength would make the text snap in instead of easing.
Do not be tempted.

## Phase 0 - validation gate: PASSED

The gate was one credit is not evidence of generalisation, so run a second clip with a
different credit style and stop if Hi-SAM does not also beat ~0.72 median fgIoU.

It scored 0.814 against the luma path's 0.784 on V1-0007, and 0.809 against 0.755 on
V1-0005. **Phase 1 onwards is justified**, though see above: the margin over
`--depth-strokes` on the second clip is 0.007, and it is the worst frame rather than the
median that carries the argument.

Two clips is still two clips. What would change the conclusion now is a credit that is *not*
this typeface family - both of these are the same stylised sci-fi face from the same title
sequence - or one over a very different background: white subtitles on a night exterior,
say, where the luma path is at its strongest and there is little for a learned model to add.

## Phase 1 - make the measurement first-class: BUILT

`scripts/label_glyphs.py` and `scripts/score_masks.py`, and every number above is scored with
them. Worth having **whether or not Hi-SAM ever ships**: it would have caught three of the
wrong turns that produced this plan, and it caught a fourth while being written.

- `scripts/label_glyphs.py` - build a labelled glyph mask from a clip with static text.
  **Then filter components by the text line's own row band and a letter-plausible height**,
  and emit a contact sheet alongside the mask. Two methods are needed, because which one
  works depends on the shot:

  - **Persistence**, for a moving shot: per pixel, how often is it bright text-coloured
    across the frames the credit is up. The credit holds still while the scene moves under
    it, so the letters answer always and the background does not.
  - **On/off differencing**, for a static shot: mean text-coloured over the frames the credit
    is up, minus the same over frames where it is absent. V1-0007 needs this - the camera
    barely moves and the cockpit is the same amber as the credit, so persistence alone marks
    the scenery as confidently as the writing.

  Both need a sanity look at the picture. Persistence on V1-0005 swept in a sunlit window,
  which is exactly as static as static text, and it cost an hour of wrong conclusions before
  anyone looked. Differencing on V1-0007 breaks down if the colour threshold is lowered far
  enough that the off frames also register, which shows up as holes punched through the
  letters.
- `scripts/score_masks.py` - fgIoU, recall and precision of any mask source against a
  labelled clip.

## Phase 2 - get the model behind a boundary: BUILT

`src/dsf/refine/hisam.py` and `scripts/fetch_hisam.py`.

**It is not vendored, and the plan said it would be.** That was written before anyone counted
the code: `hi_sam/modeling` is about 5,000 lines against this project's 5,357, so copying it
in would double the repository with third-party PyTorch nobody here maintains, and take on
Apache-2.0 redistribution obligations to do it. It is cloned at a pinned commit into
`models/` instead - already where downloaded weights live, already ignored by git - and the
pin is what keeps a measurement reproducible.

Two things that only showed up in the building:

The upstream builder resolves SAM's backbone through a path relative to the *process's*
working directory, which is why its own demo has to be run from its own folder. A library
cannot inherit that: frames are decoded on a worker thread and `chdir` is process-wide, so a
load would move the ground under a decode running beside it. Building with no checkpoint and
merging the two state dicts by absolute path is the same arithmetic without the hazard.

And the head cannot be fetched by script at all, so `fetch_hisam.py` prints the link and the
target path and returns non-zero. It reports the two extra imports rather than installing
them, because every `pip install` into this venv is a chance to lose the CUDA OpenCV.

**Setup cannot be fully scripted.** The SAM-TS head is distributed only via OneDrive and
refuses programmatic access - the short link, the share API against both the short and the
resolved URL, and `download.aspx` all returned 401/400/403. `scripts/setup.ps1` must print the
link and the target path and stop. The SAM ViT-L encoder *is* directly fetchable from
`dl.fbaipublicfiles.com` (1.25 GB), so only the 118 MB head needs a human.

## Phase 3 - wire it in: BUILT

`--strokes-from hisam`. Through the pipeline it scores 0.797 and 0.796 fgIoU against the
residual's 0.755 and 0.784, with recall of 99.4% and 99.9% and worst frames of 0.752 and
0.720 against 0.226 and 0.480 - which is the standalone result, arrived at through every
gate and filter rather than beside them.

Getting there needed two fixes, and they are the same mistake twice. The component filter
asks three times whether a blob is too big to be a letter - its area against a share of the
crop, whether it spans the crop, and its stroke width - and all three are calibrated on what
a *residual* gets wrong. A residual answers to anything thin and contrasty, so size is
evidence against text. A model trained on stroke masks has already settled that, and its
strokes come out a little fatter, which on tightly-set text joins the letters up. So one word
arrived as a single 9,649 px component against a 7,669 px cap and was discarded, and on the
second clip a joined word measured 37.7 of stroke width against a 36.5 limit - because
`_stroke_width` takes the *peak* of the distance transform and the junction where two fat
strokes meet is thicker than either of them, while the letters either side measured 15 to 19.

The first cost 0.809 -> 0.447 fgIoU and the second 0.814 -> 0.696, and neither showed up
anywhere except in a score against labelled glyphs. The filters now skip those three tests
when the shape came from a model, and keep the lower stroke bound, which is still about noise
rather than about size.

### As originally planned

Run **once per frame, full-frame**, not once per detection crop: one inference beats N, and
the geometry and appearance gates then intersect the result with accepted boxes, so filmed
signage is still excluded by the gates rather than by the model (which is trained to segment
all text, including shop signs).

```
StrokeConfig.strokes_from: "luma" (default) | "hisam"
```

In `extract_patch`, when `hisam`: take `shape` from the model's mask over the crop, `level`
from `analyse_crop` as now. If either is missing, fall back to luma - never invent a level.

Full-frame cost was the open question here and it is answered: 0.25 s/frame at 1920x800,
the same as on an 888x448 crop, because SAM resizes to 1024 internally and the four-fold
increase in pixels never reaches the model. ViT-B remains the lever if that is still too much
- 87.15 against 88.77 fgIoU on TextSeg.

## Phase 4 - hybrid - BUILT

`--strokes-from auto`. The residual answers first; the model is called only for the frames
whose answer looks like it lost letters, and `--weak-fill` is where that line sits.

### What the trigger turned out to be

Three signals were proposed above - low `level`, few surviving blobs, the agreement vote
standing down. **Two of the three do not work**, measured over both credits:

- **`level` does not separate.** It reads 1.00 on every frame of both clips, ruined and
  intact alike. The text is fully opaque while it is being lost; opacity is not the problem.
- **Blob count does not separate.** Against the letters a box that wide should hold, the
  intact frames run 2.1-2.6 and the collapsed ones 2.1-2.5. There is no gap to cut in.
- **Fill does.** How much of the detector's boxes the mask actually fills: 0.199-0.222 on the
  ruined frames against 0.315-0.405 on the intact ones, and 0.270-0.298 against 0.312-0.382
  on the second clip.

It has to be judged **over the frame, not box by box**. A box's own fill confuses a residual
that lost letters with a line that is legitimately sparse - a short name in a wide box - and
measured that way the trigger fired on some box of nearly every frame of a three-line credit:
100% of frames at any threshold that helped quality, which is the model's whole cost for less
than the model's quality. The cost is per frame regardless, because one forward pass serves
every box in the picture.

The raw margin is narrow - 0.298 against 0.312 on V1-0007 - but the *outcome* is flat from
0.30 to 0.34 on both clips, so the plateau is much wider than the margin. 0.32 is its middle.

### What it delivers

Scored through the full pipeline against the same labels, over every labelled frame:

**V1-0005**, 44 frames

| | fgIoU | recall | precision | worst frame |
|---|---|---|---|---|
| luma | 0.758 | 84.0% | **94.5%** | 0.158 |
| Hi-SAM | **0.795** | 99.4% | 80.6% | **0.434** |
| `auto` | 0.780 | 86.2% | 94.0% | **0.434** |

**V1-0007**, 44 frames (the credit; see the note below on 93-94)

| | fgIoU | recall | precision | worst frame |
|---|---|---|---|---|
| luma | 0.781 | 88.9% | **92.1%** | 0.251 |
| Hi-SAM | **0.794** | 99.8% | 80.2% | 0.249 |
| `auto` | 0.787 | 89.3% | 91.8% | 0.249 |

**`auto` takes Hi-SAM's worst frame and keeps luma's precision.** That is the whole of what it
is for. The frames it does not trigger on are the ones the residual was already cutting more
tightly than the model, and it leaves them alone.

Cost on V1-0005, 129 frames: luma 10.2 s, `auto` 25.6 s, Hi-SAM 34 s. Roughly 10 s of `auto`'s
time is constructing the ViT-L, paid once, and a clip whose credits never collapse never
constructs it at all - `--weak-fill 0` measures 10.2 s against luma's 10.2 s.

### What it does not do

**It does not beat Hi-SAM on fgIoU, and the hypothesis above that it would is wrong.** Hi-SAM
scores higher on both clips at every trigger setting; `auto` interpolates between the two
rather than exceeding either. The case for it is worst-case insurance at a fraction of the
cost, and precision that a fatter model mask gives up - not a better mask than the model's.

Which of the three to use is therefore a real choice, not a ranking:

- `luma` when the credits sit on clean backgrounds. Cheapest, most precise.
- `auto` as the general default for mixed material. It cannot collapse, and it costs nothing
  on the frames that did not need help.
- `hisam` when recall matters more than precision or render time - the only one that reaches
  99%+ of the glyph pixels.

### One thing this does not touch

The labels for V1-0007 include frames 93-94, a second caption later in the clip. All three
sources score 0.000 there: nothing is detected at all. That is a detection-stage miss, not a
stroke-shape one, and no setting here reaches it. It is why the V1-0007 table above is scored
over the credit rather than over every labelled frame - scored over all 46, every source reads
0.000 worst and the comparison says nothing.

## Risks

**Precision regression.** 83% against 90% means slightly fat strokes on easy frames. That is
the forgiving direction for depth work - a fat mask heals, a missing letter shows corruption
through the writing - but it is a real trade, and the `--dilate` / `--feather` defaults may
want revisiting alongside it.

**Licence.** The TextSeg weights are academia-only and that inherits to anything trained on
them. Fine for this project, which is entirely non-commercial, but it must be documented so a
future fork does not inherit the restriction silently. The HierText variant (fgIoU 78.37) may
carry friendlier terms if that ever matters.

**Non-determinism.** GPU and CPU masks were bit-identical on only 27 of 41 frames. The median
fgIoU was unchanged, so this is float noise at the mask threshold rather than anything
meaningful - but byte-stable renders across machines are off the table.

**Scan throughput.** 3.6x on a render is acceptable; on a 2500-clip scan it is not.
`scripts/scan_for_text.py` should keep the luma path regardless.

## How to measure, and how not to

Every wrong number in the evaluation behind this document came from scoring against something
other than what the code operates on. In order:

1. `_SCATTER_K` was calibrated on masks produced while the polarity decision was inverting, so
   the blobs measured were the gaps between glyphs rather than the glyphs.
2. A saturation-threshold "truth" mask marked the whole sunlit background as text, which
   inverted the depth veto's measured precision.
3. The demo clip's "in-scene signage" was drawn flat, so the appearance gate was asked to
   separate two identical things.
4. The halo was profiled from the ground-truth glyph mask when `--heal-dilate` grows the
   *tool's* mask, which is smaller - a factor of two.
5. The first Hi-SAM score was 64.2% recall, against a labelled mask that included a sunlit
   window the model had correctly ignored. Corrected, it was 100%.
6. Then the corrected mask, built by hand, still had enough of that window left in it to
   inflate Hi-SAM's margin from 0.054 to 0.111 on one clip and from 0.007 to 0.062 on the
   other - because it was scoring the model for ignoring scenery. Only labelling the clips
   with the tool built for it settled the numbers.
7. And `--min-on` in that tool wanted 0.7 rather than the 0.9 that seemed obviously right,
   because the frames found as carrying the text include the fade at each end. At 0.9 it
   quietly dropped the last two letters of a line, which the contact sheet showed at once
   and the pixel count did not.

Two habits follow. **Look at the picture, not only the score** - a diff image found the last
of these in seconds after the metric had hidden it for an hour. And **be suspicious of a
number that does not move**: 64.2% recall, identical to a decimal across four inference
settings that should have changed it, was the tell.
