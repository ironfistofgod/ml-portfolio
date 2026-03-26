import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    wandb.init(project="llama-coder", job_type="train")

dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
dataset = dataset.select(range(20_000))

def format_example(example):
    return {
        "text": f"### Problem:\n{example['problem']}\n\n### Solution:\n{example['solution']}"
    }
dataset = dataset.map(format_example)

model_id = "unsloth/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=None,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

training_args = SFTConfig(
    output_dir="/workspace/llama-coder",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    dataset_text_field="text",
    report_to="wandb",
    deepspeed="/app/ds_config.json",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

if local_rank == 0:
    artifact = wandb.Artifact(
        name="llama-coder-lora",
        type="model",
        metadata={
            "base_model": model_id,
            "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
            "num_samples": 20_000,
            "epochs": 2,
            "lora_rank": 16,
            "lora_alpha": 32,
            "training": "multi-gpu DeepSpeed ZeRO-2",
        }
    )
    artifact.add_dir("/workspace/llama-coder")
    wandb.log_artifact(artifact)
    wandb.finish()

hf_repo = "chethan1988/llama-coder-lora"
trainer.model.push_to_hub(hf_repo)
tokenizer.push_to_hub(hf_repo)
print(f"Model pushed to https://huggingface.co/{hf_repo}")
