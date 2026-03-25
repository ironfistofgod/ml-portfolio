import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import wandb

dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
dataset = dataset.select(range(20_000))

def format_example(example):
    return {
        "text": f"### Problem:\n{example['problem']}\n\n### Solution:\n{example['solution']}"
    }
dataset = dataset.map(format_example)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="/workspace/llama-coder",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    max_seq_length=1024,
    dataset_text_field="text",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

run = wandb.init(project="llama-coder", job_type="train")
trainer.train()

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
    }
)
artifact.add_dir("/workspace/llama-coder")
run.log_artifact(artifact)
run.finish()

hf_repo = "chethan1988/llama-coder-lora"

model.push_to_hub(hf_repo)
tokenizer.push_to_hub(hf_repo)
print(f"Model pushed to https://huggingface.co/{hf_repo}")