"""
Downloads finetrainers/3dgs-dissolve directly to OUT_DIR.

Dataset structure (already correct):
  videos/0.mp4 ... videos/100.mp4
  prompt.txt   ← one caption per line (renamed to prompts.txt)
  videos.txt   ← relative paths, one per line
"""
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

OUT_DIR = Path("/workspace/data/wan-dissolve")
HF_REPO = "finetrainers/3dgs-dissolve"

print(f"Downloading {HF_REPO} ...")
raw_dir = Path(snapshot_download(
    repo_id=HF_REPO,
    repo_type="dataset",
    local_dir=str(OUT_DIR),
))
print(f"Downloaded to {raw_dir}")

# Training script expects prompts.txt; dataset ships prompt.txt
prompt_src = OUT_DIR / "prompt.txt"
prompt_dst = OUT_DIR / "prompts.txt"
if prompt_src.exists() and not prompt_dst.exists():
    shutil.copy(prompt_src, prompt_dst)
    print("Copied prompt.txt → prompts.txt")

videos = list((OUT_DIR / "videos").glob("*.mp4"))
print(f"\nDone. {len(videos)} videos ready at {OUT_DIR}")
print(f"  {OUT_DIR}/videos/      ← mp4 files")
print(f"  {OUT_DIR}/prompts.txt  ← captions")
print(f"  {OUT_DIR}/videos.txt   ← relative paths")
