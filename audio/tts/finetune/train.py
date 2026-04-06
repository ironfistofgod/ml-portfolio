import os
import json
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
SAVE_EVERY    = 1000   # steps
LOG_EVERY     = 10

def main():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )

    # "custom" mode takes a direct path to vocab.txt, bypassing F5-TTS's internal data dir
    vocab_char_map, vocab_size = get_tokenizer(f"{DATA_DIR}/vocab.txt", "custom")

    # build model
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
        wandb.init(project="f5tts-ljspeech", config={
            "epochs": EPOCHS, "lr": LR, "batch_frames": BATCH_FRAMES,
        })

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                mel        = batch["mel"]           # (B, 100, T)
                text       = batch["text"]          # (B, text_len)
                mel_lengths = batch["mel_lengths"]  # (B,)

                loss, _, _ = model(mel.permute(0, 2, 1), text, lens=mel_lengths)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                if global_step % LOG_EVERY == 0:
                    wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % SAVE_EVERY == 0:
                    os.makedirs(CKPT_DIR, exist_ok=True)
                    accelerator.save_state(f"{CKPT_DIR}/step_{global_step}")

            
    if accelerator.is_main_process:
        os.makedirs(CKPT_DIR, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        accelerator.save(unwrapped.state_dict(), f"{CKPT_DIR}/final.pt")

        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path = CKPT_DIR,
            repo_id     = HF_REPO,
            repo_type   = "model",
        )
        wandb.finish()


if __name__ == "__main__":
    main()