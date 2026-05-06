import os
import json
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm

MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_ID  = "liuhaotian/LLaVA-Instruct-150K"
NUM_SAMPLES = 20_000
OUTPUT_DIR  = "/workspace/data/llava_visual_tokens"

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()

dataset = load_dataset(DATASET_ID, split="train")
dataset = dataset.select(range(NUM_SAMPLES))

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, example in enumerate(tqdm(dataset)):
    image = example["image"]
    conversations = example["conversations"]

    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to("cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        visual_tokens = model.visual(
            inputs["pixel_values"],
            grid_thw=inputs["image_grid_thw"],
        )

    torch.save(
        {
            "visual_tokens":  visual_tokens.cpu(),
            "image_grid_thw": inputs["image_grid_thw"].cpu(),
            "conversations":  conversations,
        },
        os.path.join(OUTPUT_DIR, f"{idx:06d}.pt"),
    )

print(f"Pre-encoded {NUM_SAMPLES} samples to {OUTPUT_DIR}")
