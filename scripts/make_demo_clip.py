"""Generate demo footage so the tools can be tried without your own clips.

By default: samples/demo_rgb.mp4 (1080p, burned-in subtitles plus two in-scene signs that
must keep their real depth) and samples/demo_depth.mp4 (a plausible 10-bit DepthCrafter
output with the depth wrecked and smeared over the subtitles).

    .venv/Scripts/python scripts/make_demo_clip.py
    dsf preview --rgb samples/demo_rgb.mp4 --depth samples/demo_depth.mp4 --frames 12,30 --out-dir samples/previews

With --scan-demo, also writes samples/scan_demo/{rgb,depth}: a small folder of clips - one
subtitled, one with scrolling credits, one carrying nothing but signage the camera
photographed - for pointing scripts/scan_for_text.py at.

    .venv/Scripts/python scripts/make_demo_clip.py --scan-demo
"""
import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dsf.videoio import synth_rgb_video, synth_test_video


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def rolling_scene(width: int, height: int, seed: int = 7) -> np.ndarray:
    """A smooth, textured backdrop in [0, 1] - nothing in it should look like a glyph."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (height // 8, width // 8)).astype(np.float32)
    base = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    base = cv2.GaussianBlur(base, (0, 0), 9)
    return (base - base.min()) / (np.ptp(base) + 1e-6)


def build_demo():
    """The original 1080p pair: subtitles over a moving scene, plus two signs."""
    W, H, N = 1920, 1080, 48
    base = rolling_scene(W, H)

    SUBS = ["I never asked for this.", "You should have come sooner."]
    rgb_frames, depth_frames = [], []
    sub_font = load_font(52)
    sign_font = load_font(44)

    for i in range(N):
        shift = i * 3
        scene = np.roll(base, shift, axis=1)
        tint = np.stack([scene * 180 + 40, scene * 160 + 45, scene * 140 + 55], -1)
        img = Image.fromarray(np.clip(tint, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        # in-scene signage: perspective-ish text up in the frame, lower contrast, coloured
        d.text((260, 300), "MOTEL", font=sign_font, fill=(210, 120, 60))
        d.text((1330, 380), "OPEN 24H", font=sign_font, fill=(90, 180, 200))
        # burned-in subtitle: white, black outline, bottom centre
        text = SUBS[(i // 24) % len(SUBS)]
        bb = d.textbbox((0, 0), text, font=sub_font, stroke_width=3)
        tw = bb[2] - bb[0]
        d.text(((W - tw) // 2, int(H * 0.86)), text, font=sub_font, fill=(255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0))
        rgb_frames.append(np.array(img))

        # a plausible DepthCrafter output: smooth depth from the scene, wrecked over the text
        depth = (scene * 400 + 250).astype(np.float32)
        depth += np.linspace(0, 180, H, dtype=np.float32)[:, None]
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).text(((W - tw) // 2, int(H * 0.86)), text, font=sub_font,
                               fill=255, stroke_width=3, stroke_fill=255)
        m = np.array(m).astype(np.float32) / 255.
        halo = np.clip(cv2.GaussianBlur(m, (0, 0), 11) * 1.6, 0, 1)
        depth = depth * (1 - halo) + 935 * halo   # smear + wrong depth
        depth_frames.append(np.clip(depth, 64, 940).astype(np.uint16))

    synth_rgb_video("samples/demo_rgb.mp4", rgb_frames, fps=24)
    synth_test_video("samples/demo_depth.mp4", depth_frames, fps=24, lossless=True)
    print("wrote samples/demo_rgb.mp4 and samples/demo_depth.mp4", N, "frames", W, "x", H)


# --------------------------------------------------------------------------- scan demo

def photographed_sign(background: np.ndarray, text: str, at, size: int = 30,
                      tint=(150, 120, 90), alpha: float = 0.55) -> np.ndarray:
    """Text as the camera found it: skewed, softened by the lens, lit by the scene.

    This is the case the scanner must *not* flag, and it is not the same thing as text
    drawn flat onto the finished frame. A shop sign sits at an angle, has soft edges and
    takes the scene's own light - which is exactly what the appearance gate reads.
    """
    h, w = background.shape[:2]
    layer = Image.new("L", (w, h), 0)
    ImageDraw.Draw(layer).text(at, text, font=load_font(size), fill=255)
    mask = np.array(layer).astype(np.float32) / 255.0

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, 12], [w, h - 18], [0, h]])  # a plane seen off-axis
    mask = cv2.warpPerspective(mask, cv2.getPerspectiveTransform(src, dst), (w, h))
    mask = cv2.GaussianBlur(mask, (0, 0), 1.8)[..., None] * alpha

    paint = np.array(tint, dtype=np.float32)[None, None, :]
    return np.clip(background.astype(np.float32) * (1 - mask) + paint * mask,
                   0, 255).astype(np.uint8)


def overlay_text(background: np.ndarray, lines, size: int = 34, y: float = 0.85,
                 stroke: int = 3) -> np.ndarray:
    """Burned-in text: flat white, hard outline, pasted on after the fact."""
    h, w = background.shape[:2]
    img = Image.fromarray(background.copy())
    draw = ImageDraw.Draw(img)
    font = load_font(size)
    for n, line in enumerate(lines):
        bb = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        top = int(h * y) + n * int(size * 1.4)
        draw.text(((w - (bb[2] - bb[0])) // 2, top), line, font=font,
                  fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
    return np.array(img)


def depth_for(rgb_frames, corrupted: bool):
    """A depth ramp per frame; over burned-in text, the smear DepthCrafter leaves."""
    h, w = rgb_frames[0].shape[:2]
    ramp = np.linspace(200, 780, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    out = []
    for frame in rgb_frames:
        depth = ramp.copy()
        if corrupted:
            # The overlay is the only near-white, hard-edged thing in these frames.
            bright = (frame.min(axis=2) > 225).astype(np.float32)
            halo = np.clip(cv2.GaussianBlur(bright, (0, 0), 7) * 2.0, 0, 1)
            depth = depth * (1 - halo) + 935 * halo
        out.append(np.clip(depth, 64, 940).astype(np.uint16))
    return out


def build_scan_demo():
    """Three short clips plus their depth maps, laid out the way the scanner expects."""
    W, H, N = 640, 360, 24
    rgb_dir = pathlib.Path("samples/scan_demo/rgb")
    depth_dir = pathlib.Path("samples/scan_demo/depth")
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    base = rolling_scene(W, H, seed=3)
    SUBS = ["I never asked for this.", "You should have come sooner."]
    CREDITS = ["DIRECTED BY", "A. N. OTHER", "PRODUCED BY", "SOMEONE ELSE"]

    talky, credits, street = [], [], []
    for i in range(N):
        scene = np.roll(base, i * 4, axis=1)
        tint = np.clip(np.stack([scene * 170 + 45, scene * 155 + 50,
                                 scene * 135 + 60], -1), 0, 255).astype(np.uint8)

        talky.append(overlay_text(photographed_sign(tint, "MOTEL", (40, 60)),
                                  [SUBS[(i // 12) % len(SUBS)]]))

        # Scrolling credits over a darker plate, moving up a few px per frame.
        plate = (tint * 0.35).astype(np.uint8)
        credits.append(overlay_text(plate, CREDITS, size=28, y=0.55 - i * 0.012, stroke=2))

        # Signage only: one high in the frame, one down in the subtitle band.
        lit = photographed_sign(tint, "MOTEL", (40 + i, 60))
        street.append(photographed_sign(lit, "AB 51 DQ", (330 + i, 300), size=22,
                                        tint=(120, 125, 130), alpha=0.5))

    for name, frames, corrupted in (("talky_shot", talky, True),
                                    ("end_credits", credits, True),
                                    ("street_shot", street, False)):
        synth_rgb_video(rgb_dir / f"{name}.mp4", frames, fps=24)
        synth_test_video(depth_dir / f"{name}_depth.mp4", depth_for(frames, corrupted),
                         fps=24, lossless=True)

    print(f"wrote {rgb_dir} and {depth_dir}: talky_shot + end_credits carry overlay text, "
          f"street_shot carries only signage the camera photographed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scan-demo", action="store_true",
                        help="also write samples/scan_demo for scripts/scan_for_text.py")
    parser.add_argument("--only-scan-demo", action="store_true",
                        help="write only samples/scan_demo, skipping the 1080p pair")
    args = parser.parse_args()

    pathlib.Path("samples").mkdir(exist_ok=True)
    if not args.only_scan_demo:
        build_demo()
    if args.scan_demo or args.only_scan_demo:
        build_scan_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
