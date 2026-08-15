# Hi-SAM as a stroke source

A plan, not a decision. Written after evaluating [Hi-SAM](https://github.com/ymy-k/Hi-SAM)
(SAM-TS-L, TextSeg weights) against the extractor on two real credits, and holding the
measurements up next to what ships today. The validation gate in Phase 0 has passed; the
phases after it are still unbuilt.

## Why this is worth considering at all

The extractor reads strokes off a luma residual: it estimates the picture without the
writing and takes the difference. That fails, by construction, where the text is the
brightness of what it sits on - and on a credit crossing a subject's hair and shoulder it
does not fail gracefully.

Measured against labelled glyph masks on two clips, 77 frames in total:

**V1-0005, "EMILY BEECHAM", 41 frames** - a credit crossing hair and shoulder

| | fgIoU (median) | recall | precision | worst frame | speed |
|---|---|---|---|---|---|
| luma (ships today) | 0.717 | 82.6% | **90.3%** | 0.153 | 0.074 s/frame |
| luma + `--depth-strokes` | 0.714 | 85.2% | 86.1% | 0.270 | - |
| Hi-SAM SAM-TS-L | **0.828** | **100%** | 83.0% | **0.704** | 0.20 s/frame |

**V1-0007, "WITH / DAVID CORENSWET", 36 frames** - a near-static cockpit interior

| | fgIoU (median) | recall | precision | worst frame | speed |
|---|---|---|---|---|---|
| luma | 0.785 | 89.0% | **92.1%** | 0.484 | - |
| luma + `--depth-strokes` | 0.810 | 93.6% | 89.5% | 0.512 | - |
| Hi-SAM SAM-TS-L | **0.872** | **100%** | 87.5% | **0.786** | 0.23 s/frame |

Hi-SAM was better on **69 of 69** frames across the two clips. On both, its *worst* frame beat
the luma path's median, and its recall was exactly 100%.

The per-frame trace on the first clip is the vivid part: luma holds 0.72-0.78 and then
collapses to 0.38 across frames 67-73, where the shot moves the credit over hair, while
Hi-SAM does not move at all - 0.80 to 0.84, monotonic. But the second clip is the more
useful result. There the luma path never collapses, never dropping below 0.484, and Hi-SAM
still won every frame. So this is not only a rescue for disasters.

What does not generalise is the size of the win: 0.111 of fgIoU on the hard clip against
0.062 on the easier one. What does generalise is the shape of it - perfect recall, slightly
lower precision, and no variance across frames.

Both methods find the small dim "WITH" caption on the second clip, so that class of text is
not a differentiator. Hi-SAM covers 677 px of it against luma's 469, which is the same
over-filling seen everywhere else rather than a difference in what was found.

That collapse is the same failure that motivated the polarity fix, the depth agreement veto
and `--depth-strokes`. All three are workarounds for a contrast heuristic meeting text the
colour of its background. A model trained on stroke masks does not have the failure mode.

Speed was measured on an RTX 5080 with a cu130 torch build: 0.2 s/frame against 6.0 s/frame
on CPU, identical accuracy. A 129-frame render goes from ~10 s to ~35 s.

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

It scored 0.872 against the luma path's 0.785 on V1-0007, winning 36 of 36 frames. Combined
with 33 of 33 on V1-0005, that is 69 of 69. **Phase 1 onwards is justified.**

Two clips is still two clips. What would change the conclusion now is a credit that is *not*
this typeface family - both of these are the same stylised sci-fi face from the same title
sequence - or one over a very different background: white subtitles on a night exterior,
say, where the luma path is at its strongest and there is little for a learned model to add.

## Phase 1 - make the measurement first-class

Worth building **whether or not Hi-SAM ever ships**. It would have caught three of the wrong
turns that produced this plan.

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

## Phase 2 - vendor the model behind a flag

- Vendor `hi_sam/modeling/`; it is not a pip package. It derives from `segment-anything` and
  `sam-hq` - carry their notices.
- `src/dsf/refine/hisam.py`: load once, cache the predictor, expose
  `strokes(frame) -> float32 mask`.
- Checkpoints into `models/hisam/`, alongside the docTR and EasyOCR caches that
  `configure_model_cache` already manages.

**Setup cannot be fully scripted.** The SAM-TS head is distributed only via OneDrive and
refuses programmatic access - the short link, the share API against both the short and the
resolved URL, and `download.aspx` all returned 401/400/403. `scripts/setup.ps1` must print the
link and the target path and stop. The SAM ViT-L encoder *is* directly fetchable from
`dl.fbaipublicfiles.com` (1.25 GB), so only the 118 MB head needs a human.

## Phase 3 - wire it in

Run **once per frame, full-frame**, not once per detection crop: one inference beats N, and
the geometry and appearance gates then intersect the result with accepted boxes, so filmed
signage is still excluded by the gates rather than by the model (which is trained to segment
all text, including shop signs).

```
StrokeConfig.strokes_from: "luma" (default) | "hisam"
```

In `extract_patch`, when `hisam`: take `shape` from the model's mask over the crop, `level`
from `analyse_crop` as now. If either is missing, fall back to luma - never invent a level.

**Measure full-frame cost before committing to it.** The 0.2 s/frame above is an 888x448 crop;
1920x800 is about four times the pixels, though SAM's internal 1024 resize means it will not
scale linearly. If it is bad, ViT-B is the lever: 87.15 against 88.77 fgIoU on TextSeg.

## Phase 4 - hybrid, only if the above holds

The interesting endpoint, because the two are complementary rather than ranked: the luma path
is *more precise* (90% against 83%) where contrast exists, and Hi-SAM is unbreakable where it
does not. Use luma by default and fall back where it is weak - low `level`, few surviving
blobs, the agreement vote standing down.

Defer this. The trigger needs validating across several clips or it is just a new way to be
wrong.

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

Two habits follow. **Look at the picture, not only the score** - a diff image found the last
of these in seconds after the metric had hidden it for an hour. And **be suspicious of a
number that does not move**: 64.2% recall, identical to a decimal across four inference
settings that should have changed it, was the tell.
