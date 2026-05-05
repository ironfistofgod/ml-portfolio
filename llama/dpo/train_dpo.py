import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
BASE_MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
SFT_ADAPTER   = "chethan1988/llama-coder-lora"
HF_REPO       = "chethan1988/llama-coder-dpo"
OUTPUT_DIR    = "/workspace/llama-coder-dpo"
NUM_SAMPLES   = 20_000


local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    wandb.init(project="llama-coder", job_type="dpo")

raw = load_dataset("Vezora/Code-Preference-Pairs", split="train")
raw = raw.select(range(NUM_SAMPLES))

def format_dpo(example):
    return {
        "prompt":   example["instruction"],
        "chosen":   example["accepted"],
        "rejected": example["rejected"],
    }

dataset = raw.map(format_dpo, remove_columns=raw.column_names)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=None,
)
model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
    report_to="wandb",
    deepspeed="/app/ds_config.json",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

if local_rank == 0:
    artifact = wandb.Artifact(
        name="llama-coder-dpo",
        type="model",
        metadata={
            "base_model": BASE_MODEL_ID,
            "sft_adapter": SFT_ADAPTER,
            "dataset": "Vezora/Code-Preference-Pairs",
            "num_samples": NUM_SAMPLES,
            "beta": 0.1,
            "epochs": 1,
        },
    )
    artifact.add_dir(OUTPUT_DIR)
    wandb.log_artifact(artifact)
    wandb.finish()

trainer.model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"DPO model pushed to https://huggingface.co/{HF_REPO}")