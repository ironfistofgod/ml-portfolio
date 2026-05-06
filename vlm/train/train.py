# Qwen2.5-VL-7B fine-tuning with FSDP FULL_SHARD across 2×A100
# Dataset: liuhaotian/LLaVA-Instruct-150K (20K subset, pre-encoded by prepare_data.py)
# LoRA: r=64, alpha=128
# Profiling: torch.profiler first 50 steps → Chrome trace → bottleneck analysis
# Output: chethan1988/qwen2.5-vl-7b-llava-lora
import os
import glob
import torch
import wandb
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR    = "/workspace/data/llava_visual_tokens"
OUTPUT_DIR  = "/workspace/qwen-vl-lora"
HF_REPO     = "chethan1988/qwen2.5-vl-7b-llava-lora"
LORA_RANK   = 64
LORA_ALPHA  = 128

class LLaVAVisualDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.processor = processor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], weights_only=False)

        conv          = sample["conversations"]
        user_text     = conv[0]["value"].replace("<image>", "").strip()
        assistant_txt = conv[1]["value"]

        prompt   = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        full     = prompt + assistant_txt + "<|im_end|>"

        tokens   = self.processor.tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
        input_ids      = tokens.input_ids[0]
        attention_mask = tokens.attention_mask[0]

        prompt_len = len(self.processor.tokenizer(prompt, return_tensors="pt").input_ids[0])
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "visual_tokens":  sample["visual_tokens"],
            "image_grid_thw": sample["image_grid_thw"],
        }


local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    wandb.init(project="qwen-vl", job_type="train")

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=None,
)

for p in model.visual.parameters():
    p.requires_grad = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="wandb",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

train_dataset = LLaVAVisualDataset(DATA_DIR, processor)

def collate_fn(batch):
    return {
        "input_ids":      torch.nn.utils.rnn.pad_sequence(
                              [b["input_ids"]      for b in batch],
                              batch_first=True,
                              padding_value=processor.tokenizer.pad_token_id,
                          ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                              [b["attention_mask"] for b in batch],
                              batch_first=True,
                              padding_value=0,
                          ),
        "labels":         torch.nn.utils.rnn.pad_sequence(
                              [b["labels"]         for b in batch],
                              batch_first=True,
                              padding_value=-100,
                          ),
        "visual_tokens":  torch.cat([b["visual_tokens"]  for b in batch], dim=0),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

trainer.train()

PROFILE = os.environ.get("PROFILE", "0") == "1"

if PROFILE:
    from transformers import TrainerCallback
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    training_args.max_steps = 50

    class ProfilerCallback(TrainerCallback):
        def __init__(self):
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=3, active=10, repeat=1),
                on_trace_ready=tensorboard_trace_handler("/workspace/profile_trace"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

        def on_train_begin(self, args, state, control, **kwargs):
            self.prof.__enter__()

        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()

        def on_train_end(self, args, state, control, **kwargs):
            self.prof.__exit__(None, None, None)

    trainer.add_callback(ProfilerCallback())
    
if local_rank == 0:
    artifact = wandb.Artifact(
        name="qwen-vl-lora",
        type="model",
        metadata={
            "base_model":   MODEL_ID,
            "dataset":      "liuhaotian/LLaVA-Instruct-150K",
            "num_samples":  20_000,
            "lora_rank":    LORA_RANK,
            "lora_alpha":   LORA_ALPHA,
            "framework":    "FSDP FULL_SHARD",
        },
    )
    artifact.add_dir(OUTPUT_DIR)
    wandb.log_artifact(artifact)
    wandb.finish()

trainer.model.push_to_hub(HF_REPO)
processor.push_to_hub(HF_REPO)
print(f"VLM LoRA pushed to https://huggingface.co/{HF_REPO}")