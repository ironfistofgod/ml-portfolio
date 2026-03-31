"""
Downloads finetrainers/3dgs-dissolve from HuggingFace and formats it
for finetrainers training:
  videos/00000.mp4 ...
  prompts.txt
  videos.txt
"""
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

OUT_DIR  = Path("/workspace/data/wan-dissolve")
HF_REPO  = "finetrainers/3dgs-dissolve"

print(f"Downloading {HF_REPO} ...")
raw_dir = snapshot_download(
    repo_id=HF_REPO,
    repo_type="dataset",
    local_dir="/workspace/data/wan-dissolve-raw",
)
print(f"Downloaded to {raw_dir}")

# Clear and recreate output dirs
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)
(OUT_DIR / "videos").mkdir()

raw_path = Path(raw_dir)
video_files = sorted(raw_path.glob("*.mp4"))

print(f"Found {len(video_files)} videos")

prompts = []
video_paths = []

for i, video in enumerate(video_files):
    caption_file = video.with_suffix(".txt")
    if not caption_file.exists():
        print(f"  WARNING: no caption for {video.name}, skipping")
        continue

    dst = OUT_DIR / "videos" / f"{i:05d}.mp4"
    shutil.copy(video, dst)

    caption = caption_file.read_text().strip()
    prompts.append(caption)
    video_paths.append(f"videos/{i:05d}.mp4")
    
(OUT_DIR / "prompts.txt").write_text("\n".join(prompts))
(OUT_DIR / "videos.txt").write_text("\n".join(video_paths))

print(f"\nDone. {len(prompts)} videos ready at {OUT_DIR}")
print(f"  {OUT_DIR}/videos/       ← mp4 files")
print(f"  {OUT_DIR}/prompts.txt   ← captions")
print(f"  {OUT_DIR}/videos.txt    ← relative paths")