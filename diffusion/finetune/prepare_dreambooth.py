"""
Prepares DreamBooth training data:
  - Reads JPEGs + HEICs from ~/Downloads/cbkr-person/
  - Center-crops and resizes to 1024x1024
  - Writes caption "a photo of sks person" for each image
  - Saves to ml-portfolio/diffusion/finetune/data/dreambooth/
"""

from pathlib import Path
from PIL import Image
import pillow_heif
import sys

pillow_heif.register_heif_opener()

SRC_DIR = Path.home() / "Downloads" / "cbkr-person"
OUT_DIR = Path(__file__).parent / "data" / "dreambooth"
RESOLUTION = 1024
CAPTION = "a photo of sks person"

OUT_DIR.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
sources = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in exts])

if not sources:
    print(f"No images found in {SRC_DIR}")
    sys.exit(1)

print(f"Found {len(sources)} images in {SRC_DIR}")

for i, src in enumerate(sources):
    out_img = OUT_DIR / f"{i+1:04d}.jpg"
    out_txt = OUT_DIR / f"{i+1:04d}.txt"

    img = Image.open(src).convert("RGB")

    # Center crop to square then resize
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

    img.save(out_img, "JPEG", quality=95)
    out_txt.write_text(CAPTION)

    print(f"  [{i+1:02d}/{len(sources)}] {src.name} → {out_img.name}")

print(f"\nDone. {len(sources)} images saved to {OUT_DIR}")
