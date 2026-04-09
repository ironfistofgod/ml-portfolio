import os
import torch
import torch.profiler
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi, create_repo

DATA_DIR  = "/workspace/data/musiccaps"
CKPT_DIR  = "/workspace/ckpts/musicgen-small"
HF_REPO   = "chethan1988/musicgen-small-musiccaps"

SAMPLE_RATE   = 32_000
EPOCHS        = 20
LR            = 1e-4
WARMUP_STEPS  = 500
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
MAX_GRAD_NORM = 1.0
SAVE_EVERY    = 500
LOG_EVERY     = 10
KEEP_CKPTS    = 3

def main():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )

    # load pretrained — from_pretrained handles weights correctly
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", local_files_only=True)
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", local_files_only=True)

    model.config.decoder_start_token_id = 2048
    model.config.decoder.decoder_start_token_id = 2048
    model.generation_config.decoder_start_token_id = 2048

    # freeze text encoder (T5) and audio encoder (EnCodec) — only train decoder LM
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.audio_encoder.parameters():
        param.requires_grad = False

    # apply LoRA to decoder attention layers only
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
        
    
        # dataset — pre-tokenized by prepare_data.py
    dataset = HFDataset.load_from_disk(DATA_DIR)

    def collate_fn(batch):
        input_ids = [{"input_ids": torch.tensor(item["input_ids"])} for item in batch]

        # transpose [n_q, T] → [T, n_q] as MusicGen forward expects [B, T, n_q]
        labels = [torch.tensor(item["labels"]).T for item in batch]

        text = processor.tokenizer.pad(input_ids, return_tensors="pt")

        # pad along T dimension → [B, T, n_q]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids":      text["input_ids"],
            "attention_mask": text["attention_mask"],
            "labels":         labels_padded,
        }

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 4,
        pin_memory  = True,
    )
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=WARMUP_STEPS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=len(loader) * EPOCHS - WARMUP_STEPS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS],
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    global_step = 0

    if accelerator.is_main_process:
        wandb.init(
            project="musicgen-finetune",
            config={"epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE},
        )

    PROFILE_STEPS = 5

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", disable=not accelerator.is_local_main_process):
            grad_norm = 0.0
            with accelerator.accumulate(model):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()

                if global_step < PROFILE_STEPS:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_stack=False,
                    ) as prof:
                        outputs = model(
                            input_ids      = batch["input_ids"],
                            attention_mask = batch["attention_mask"],
                            labels         = batch["labels"],
                        )
                        loss = outputs.loss
                    if accelerator.is_main_process and global_step == PROFILE_STEPS - 1:
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                else:
                    outputs = model(
                        input_ids      = batch["input_ids"],
                        attention_mask = batch["attention_mask"],
                        labels         = batch["labels"],
                    )
                    loss = outputs.loss

                end_event.record()
                torch.cuda.synchronize()
                gpu_ms = start_event.elapsed_time(end_event)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM))

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss  += loss.item()
            epoch_steps += 1

            if accelerator.is_main_process:
                if global_step % LOG_EVERY == 0:
                    wandb.log({
                        "loss":      loss.item(),
                        "grad_norm": grad_norm,
                        "lr":        scheduler.get_last_lr()[0],
                        "gpu_ms":    gpu_ms,
                        "step":      global_step,
                    })

                if global_step % SAVE_EVERY == 0:
                    os.makedirs(CKPT_DIR, exist_ok=True)
                    save_path = f"{CKPT_DIR}/model_{global_step}.pt"
                    accelerator.save(
                        {
                            "model_state_dict":     accelerator.unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "global_step":          global_step,
                            "epoch":                epoch,
                        },
                        save_path,
                    )
                    ckpts = sorted([
                        f for f in os.listdir(CKPT_DIR)
                        if f.startswith("model_") and f.endswith(".pt")
                    ], key=lambda x: int(x.split("_")[1].split(".")[0]))
                    for old in ckpts[:-KEEP_CKPTS]:
                        os.remove(f"{CKPT_DIR}/{old}")

        if accelerator.is_main_process:
            wandb.log({
                "epoch_loss": epoch_loss / epoch_steps,
                "epoch":      epoch + 1,
                "step":       global_step,
            })

    if accelerator.is_main_process:
            os.makedirs(CKPT_DIR, exist_ok=True)
            accelerator.save(
                {
                    "model_state_dict":     accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step":          global_step,
                    "epoch":                EPOCHS,
                },
                f"{CKPT_DIR}/model_final.pt",
            )
            create_repo(HF_REPO, exist_ok=True, repo_type="model")
            HfApi().upload_folder(folder_path=CKPT_DIR, repo_id=HF_REPO, repo_type="model")
            wandb.finish()

if __name__ == "__main__":
    main()