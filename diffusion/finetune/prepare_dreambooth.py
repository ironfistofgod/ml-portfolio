"""
Prepares DreamBooth training data:
  - Reads JPEGs from ~/Downloads/images/ (HEIC already converted to jpg via sips)
  - Center-crops and resizes to 1024x1024
  - Writes per-image descriptive captions (sks man + visual description)
  - Clears existing data/dreambooth/ and repopulates fresh
  - Saves to ml-portfolio/diffusion/finetune/data/dreambooth/
"""

from pathlib import Path
from PIL import Image
import shutil
import sys

SRC_DIR = Path.home() / "Downloads" / "images"
OUT_DIR = Path(__file__).parent / "data" / "dreambooth"
RESOLUTION = 1024

# Per-image captions — keyed by source filename
CAPTIONS = {
    "0009.jpg":      "sks man, close-up selfie, round black glasses, dark jacket, brick wall background, mouth slightly open, natural daylight",
    "IMG_0597.jpg":  "sks man, close-up selfie, round glasses, white sleeveless shirt, slight smile, outdoor night setting with string lights and trees",
    "IMG_0995.jpg":  "sks man, close-up selfie, no glasses, striped shirt, urban street at night, direct gaze at camera",
    "IMG_0999.jpg":  "sks man, close-up selfie, no glasses, striped shirt, city scaffolding background at night, neutral expression",
    "IMG_1037.jpg":  "sks man, close-up selfie, round glasses, striped t-shirt, plain white indoor background, neutral expression, both earrings visible",
    "IMG_1046.jpg":  "sks man, close-up selfie, round glasses, striped t-shirt, bright indoor mall background, slight smile",
    "IMG_1055.jpg":  "sks man, close-up selfie, round glasses, dark shirt, textured grey indoor background, neutral expression",
    "IMG_1868.jpg":  "sks man, overhead portrait looking up at camera, no glasses, curly dark hair, lying on patterned pillow, soft indoor lighting, slight smile",
    "IMG_3237.jpg":  "sks man, close-up selfie, no glasses, thick mustache, shirtless, indoor bedroom, direct gaze, bright daylight",
    "IMG_3252.jpg":  "sks man, close-up selfie, no glasses, mustache, black and white striped t-shirt, outdoor park with city buildings, overcast daylight",
    "IMG_3253.jpg":  "sks man, close-up selfie, no glasses, mustache, black and white striped t-shirt, outdoor park with trees and skyline, soft expression",
    "IMG_3258.jpg":  "sks man, close-up selfie, no glasses, mustache, earbuds in, striped t-shirt, outdoor park path, overcast sky",
    "IMG_4880.jpg":  "sks man, close-up portrait, no glasses, longer wavy dark hair, teal polo shirt, outdoor green park, bright natural sunlight",
    "IMG_9536.jpg":  "sks man, close-up selfie, no glasses, navy zip jacket, slight smile, indoor background with colorful abstract paintings",
}

# Skip original HEIC files — we use the converted .jpg versions
exts = {".jpg", ".jpeg", ".png"}
sources = sorted([
    p for p in SRC_DIR.iterdir()
    if p.suffix.lower() in exts and p.name in CAPTIONS
])

if not sources:
    print(f"No matching images found in {SRC_DIR}")
    sys.exit(1)

# Clear existing dreambooth data and start fresh
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

print(f"Found {len(sources)} images in {SRC_DIR}")
print(f"Output: {OUT_DIR}\n")

for i, src in enumerate(sources):
    out_img = OUT_DIR / f"{i+1:04d}.jpg"
    out_txt = OUT_DIR / f"{i+1:04d}.txt"
    caption = CAPTIONS[src.name]

    img = Image.open(src).convert("RGB")

    # Center crop to square then resize to 1024x1024
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

    img.save(out_img, "JPEG", quality=95)
    out_txt.write_text(caption)

    print(f"  [{i+1:02d}/{len(sources)}] {src.name}")
    print(f"           → {out_img.name}")
    print(f"           → \"{caption}\"\n")

print(f"Done. {len(sources)} images saved to {OUT_DIR}")
