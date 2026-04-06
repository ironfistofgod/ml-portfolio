import json
import os
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from pathlib import Path
from tqdm import tqdm

DATASET_DIR  = "/workspace/data/LJSpeech-1.1"
SAVE_DIR     = "/workspace/data/LJSpeech_char"
META_FILE    = os.path.join(DATASET_DIR, "metadata.csv")
MIN_DURATION = 0.3   # seconds — skip clips shorter than this
MAX_DURATION = 30.0  # seconds — skip clips longer than this


def prepare():
    records = []
    durations = []
    vocab = set()

    with open(META_FILE, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Scanning LJSpeech"):
        parts = line.strip().split("|")
        utt_id    = parts[0]                          # e.g. LJ001-0001
        norm_text = parts[2]                          # normalized text
        wav_path  = Path(DATASET_DIR) / "wavs" / f"{utt_id}.wav"

        duration = sf.info(wav_path).duration         # read header only, no audio loaded
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue

        records.append({
            "audio_path": str(wav_path),
            "text":       norm_text,
            "duration":   duration,
        })
        durations.append(duration)
        vocab.update(list(norm_text))
        
    os.makedirs(SAVE_DIR, exist_ok=True)

    # write raw.arrow
    with ArrowWriter(path=f"{SAVE_DIR}/raw.arrow") as writer:
        for record in tqdm(records, desc="Writing raw.arrow"):
            writer.write(record)
        writer.finalize()

    # write duration.json
    with open(f"{SAVE_DIR}/duration.json", "w") as f:
        json.dump({"duration": durations}, f)

    # write vocab.txt
    with open(f"{SAVE_DIR}/vocab.txt", "w") as f:
        for char in sorted(vocab):
            f.write(char + "\n")

    print(f"Done. {len(records)} samples")
    print(f"Vocab size: {len(vocab)}")
    print(f"Total audio: {sum(durations)/3600:.2f} hours")


if __name__ == "__main__":
    prepare()