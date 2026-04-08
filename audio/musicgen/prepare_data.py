import os
import torch
import numpy as np
import soundfile as sf
import subprocess
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from tqdm import tqdm

DATA_DIR    = "/workspace/data/musiccaps"
SAMPLE_RATE = 32_000
MAX_DURATION = 10   # seconds — MusicCaps clips are 10s
MAX_SAMPLES  = None  # set to e.g. 500 for a quick test run

def download_clip(ytid, start_s, end_s, out_path):
    """Download a YouTube clip and trim to [start_s, end_s]."""
    url = f"https://www.youtube.com/watch?v={ytid}"
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"-ss {start_s} -t {end_s - start_s} -ar {SAMPLE_RATE} -ac 1",
        "-o", out_path,
        "--quiet",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

def encode_audio(wav_path, model, processor):
    """Load wav, encode with EnCodec, return token labels [n_q, T]."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    # trim or pad to MAX_DURATION
    max_samples = MAX_DURATION * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    inputs = processor(
        audio=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    with torch.no_grad():
        encoded = model.audio_encoder(
            input_values = inputs["input_values"].cuda(),
            padding_mask = inputs["padding_mask"].cuda() if "padding_mask" in inputs else None,
        )

    # audio_codes shape: [1, 1, n_q, T] → squeeze to [n_q, T]
    labels = encoded.audio_codes.squeeze(0).squeeze(0).cpu()
    return labels.tolist()

def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    audio_dir = f"{DATA_DIR}/audio"
    os.makedirs(audio_dir, exist_ok=True)

    # skip if already done
    if os.path.exists(f"{DATA_DIR}/train.arrow"):
        print("Data already prepared, skipping.")
        return

    # load model + processor for encoding
    print("Loading MusicGen for EnCodec encoding...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.audio_encoder.cuda().eval()

    # load MusicCaps metadata
    print("Loading MusicCaps metadata...")
    meta = load_dataset("google/MusicCaps", split="train")
    if MAX_SAMPLES:
        meta = meta.select(range(MAX_SAMPLES))

    rows = []
    for item in tqdm(meta, desc="Processing MusicCaps"):
        ytid    = item["ytid"]
        start_s = item["start_s"]
        end_s   = item["end_s"]
        caption = item["caption"]

        wav_path = f"{audio_dir}/{ytid}.wav"

        # download if not already cached
        if not os.path.exists(wav_path):
            ok = download_clip(ytid, start_s, end_s, wav_path)
            if not ok:
                continue  # skip unavailable videos

        # tokenize caption
        tokens = processor.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids = tokens["input_ids"][0].tolist()

        # encode audio → labels
        try:
            labels = encode_audio(wav_path, model, processor)
        except Exception as e:
            print(f"Skipping {ytid}: {e}")
            continue

        rows.append({"input_ids": input_ids, "labels": labels, "caption": caption})

    # save as arrow
    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(DATA_DIR)
    print(f"Saved {len(rows)} clips to {DATA_DIR}")


if __name__ == "__main__":
    prepare()