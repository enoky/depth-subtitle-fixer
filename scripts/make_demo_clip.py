"""Generate a demo clip pair so the tool can be tried without your own footage.

Writes samples/demo_rgb.mp4 (1080p, burned-in subtitles plus two in-scene signs that must
keep their real depth) and samples/demo_depth.mp4 (a plausible 10-bit DepthCrafter output
with the depth wrecked and smeared over the subtitles).

    .venv/Scripts/python scripts/make_demo_clip.py
    dsf preview --rgb samples/demo_rgb.mp4 --depth samples/demo_depth.mp4 --frames 12,30 --out-dir samples/previews
"""
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dsf.videoio import synth_rgb_video, synth_test_video

pathlib.Path("samples").mkdir(exist_ok=True)


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)

W,H,N = 1920,1080,48
rng = np.random.default_rng(7)
# a moving, textured "scene"
base = rng.normal(0,1,(H//8, W//8)).astype(np.float32)
base = cv2.resize(base,(W,H),interpolation=cv2.INTER_CUBIC)
base = cv2.GaussianBlur(base,(0,0),9)
base = (base-base.min())/(np.ptp(base)+1e-6)

SUBS = ["I never asked for this.", "You should have come sooner."]
rgb_frames, depth_frames = [], []
sub_font = load_font(52); sign_font = load_font(44)

for i in range(N):
    shift = i*3
    scene = np.roll(base, shift, axis=1)
    tint = np.stack([scene*180+40, scene*160+45, scene*140+55], -1)
    img = Image.fromarray(np.clip(tint,0,255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # in-scene signage: perspective-ish text up in the frame, lower contrast, coloured
    d.text((260, 300), "MOTEL", font=sign_font, fill=(210,120,60))
    d.text((1330, 380), "OPEN 24H", font=sign_font, fill=(90,180,200))
    # burned-in subtitle: white, black outline, bottom centre
    text = SUBS[(i//24) % len(SUBS)]
    bb = d.textbbox((0,0), text, font=sub_font, stroke_width=3)
    tw = bb[2]-bb[0]
    d.text(((W-tw)//2, int(H*0.86)), text, font=sub_font, fill=(255,255,255),
           stroke_width=3, stroke_fill=(0,0,0))
    frame = np.array(img)
    rgb_frames.append(frame)

    # a plausible DepthCrafter output: smooth depth from the scene, wrecked over the text
    depth = (scene*400 + 250).astype(np.float32)
    depth += np.linspace(0,180,H,dtype=np.float32)[:,None]
    txt = np.zeros((H,W),np.uint8)
    td = ImageDraw.Draw(Image.fromarray(txt))
    m = Image.new("L",(W,H),0); ImageDraw.Draw(m).text(((W-tw)//2, int(H*0.86)), text,
        font=sub_font, fill=255, stroke_width=3, stroke_fill=255)
    m = np.array(m).astype(np.float32)/255.
    halo = cv2.GaussianBlur(m,(0,0),11)
    depth = depth*(1-np.clip(halo*1.6,0,1)) + 935*np.clip(halo*1.6,0,1)   # smear + wrong depth
    depth_frames.append(np.clip(depth,64,940).astype(np.uint16))

synth_rgb_video("samples/demo_rgb.mp4", rgb_frames, fps=24)
synth_test_video("samples/demo_depth.mp4", depth_frames, fps=24, lossless=True)
print("wrote samples/demo_rgb.mp4 and samples/demo_depth.mp4", N, "frames", W, "x", H)
