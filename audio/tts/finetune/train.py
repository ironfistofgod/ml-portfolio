import os
import json
import shutil
import torch
import torchaudio
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from tqdm import tqdm
from accelerate import Accelerator

from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import CustomDataset, DynamicBatchSampler, collate_fn
from f5_tts.model.utils import get_tokenizer

# paths
DATA_DIR    = "/workspace/data/LJSpeech_char"
CKPT_DIR    = "/workspace/ckpts/f5tts-ljspeech"
HF_REPO     = "chethan1988/f5tts-ljspeech-lora"

# mel
TARGET_SR     = 24_000
N_MEL         = 100
HOP_LENGTH    = 256
N_FFT         = 1024

# training
EPOCHS        = 100
LR            = 1e-5
WARMUP_STEPS  = 1000
BATCH_FRAMES  = 3200   # total mel frames per batch
MAX_SAMPLES   = 64     # max clips per batch
GRAD_ACCUM    = 2
MAX_GRAD_NORM = 1.0
SAVE_EVERY      = 1000   # save every N steps 
LOG_EVERY       = 10
KEEP_CKPTS      = 5      # keep last N step checkpoints

def main():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )

    # "custom" mode takes a direct path to vocab.txt, bypassing F5-TTS's internal data dir
    vocab_char_map, vocab_size = get_tokenizer(f"{DATA_DIR}/vocab.txt", "custom")

    # build model — same architecture as F5TTS_v1_Base
    model = CFM(
        transformer=DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            text_num_embeds=vocab_size,
            mel_dim=N_MEL,
        ),
        mel_spec_kwargs=dict(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mel_channels=N_MEL,
            target_sample_rate=TARGET_SR,
        ),
        vocab_char_map=vocab_char_map,
    )

    # load pretrained F5TTS_v1_Base weights — fine-tune from pretrained, not from scratch
    if accelerator.is_main_process:
        print("Loading pretrained F5TTS_v1_Base weights...")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    ckpt_path = hf_hub_download(
        repo_id="SWivid/F5-TTS",
        filename="F5TTS_v1_Base/model_1250000.safetensors",
        cache_dir=os.environ.get("HF_HOME", "/workspace/hf_cache"),
    )
    state_dict = load_file(ckpt_path)
    # strict=False — text_embed may differ in vocab size, all other weights load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if accelerator.is_main_process:
        print(f"Pretrained weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # add LoRA adapters on top of pretrained weights
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    # dataset
    hf_dataset = HFDataset.from_file(f"{DATA_DIR}/raw.arrow")

    with open(f"{DATA_DIR}/duration.json", "r") as f:
        durations = json.load(f)["duration"]

    dataset = CustomDataset(
        hf_dataset,
        durations          = durations,
        target_sample_rate = TARGET_SR,
        n_mel_channels     = N_MEL,
        hop_length         = HOP_LENGTH,
        n_fft              = N_FFT,
    )

    from torch.utils.data import SequentialSampler
    sampler = DynamicBatchSampler(
        SequentialSampler(dataset),
        frames_threshold = BATCH_FRAMES,
        max_samples      = MAX_SAMPLES,
        random_seed      = 42,
    )

    loader = DataLoader(
        dataset,
        batch_sampler = sampler,
        collate_fn    = collate_fn,
        num_workers   = 4,
        pin_memory    = True,
    )
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor = 1e-8,   # starts at near-zero LR
        end_factor   = 1.0,    # reaches full LR
        total_iters  = WARMUP_STEPS,
    )
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=10_000_000
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup_scheduler, constant_scheduler],
        milestones = [WARMUP_STEPS],
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    
    global_step = 0

    if accelerator.is_main_process:
        wandb.init(
            project="f5tts-lora",
            config={"epochs": EPOCHS, "lr": LR, "batch_frames": BATCH_FRAMES},
        )

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", disable=not accelerator.is_local_main_process):
            grad_norm = 0.0
            with accelerator.accumulate(model):
                mel         = batch["mel"]           # (B, 100, T)
                text        = batch["text"]          # list[str]
                mel_lengths = batch["mel_lengths"]   # (B,)

                loss, _, _ = model(mel.permute(0, 2, 1), text, lens=mel_lengths)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM))

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step  += 1
            epoch_loss   += loss.item()
            epoch_steps  += 1

            if accelerator.is_main_process:
                if global_step % LOG_EVERY == 0:
                    wandb.log({
                        "loss":      loss.item(),
                        "grad_norm": grad_norm,
                        "lr":        scheduler.get_last_lr()[0],
                        "step":      global_step,
                    })

                if global_step % SAVE_EVERY == 0:
                    os.makedirs(CKPT_DIR, exist_ok=True)
                    save_path = f"{CKPT_DIR}/step_{global_step}"
                    accelerator.unwrap_model(model).save_pretrained(save_path)
                    # keep only last KEEP_CKPTS step checkpoints
                    ckpts = sorted([
                        d for d in os.listdir(CKPT_DIR)
                        if d.startswith("step_") and os.path.isdir(f"{CKPT_DIR}/{d}")
                    ], key=lambda x: int(x.split("_")[1]))
                    for old in ckpts[:-KEEP_CKPTS]:
                        shutil.rmtree(f"{CKPT_DIR}/{old}")

        avg_epoch_loss = epoch_loss / epoch_steps

        if accelerator.is_main_process:
            wandb.log({
                "epoch_loss": avg_epoch_loss,
                "epoch":      epoch + 1,
                "step":       global_step,
            })



    if accelerator.is_main_process:
        os.makedirs(CKPT_DIR, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(f"{CKPT_DIR}/lora_final")

        from huggingface_hub import HfApi, create_repo
        create_repo(HF_REPO, exist_ok=True, repo_type="model")
        api = HfApi()
        api.upload_folder(
            folder_path = CKPT_DIR,
            repo_id     = HF_REPO,
            repo_type   = "model",
        )
        wandb.finish()


if __name__ == "__main__":
    main()