import os
import sys
import argparse
import math
import torch
import wandb
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel
from huggingface_hub import upload_folder, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX style LoRA fine-tuning")

    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder with images + .txt captions")
    parser.add_argument("--output_dir", type=str, default="/workspace/flux-style-lora")
    parser.add_argument("--hf_repo", type=str, required=True, help="HuggingFace repo to push adapter")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="flux-style-lora")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    return parser.parse_args()


class StyleDataset(Dataset):
    def __init__(self, data_dir, resolution) -> None:
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        
        self.image_paths = sorted([
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ])
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption_path = image_path.with_suffix(".txt")
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        caption = caption_path.read_text().strip() if caption_path.exists() else "a painting in oil portrait style"
        
        return {"image": image, "caption": caption}
    

def encode_prompt(captions, tokenizer_clip, tokenizer_t5, text_encoder_clip, text_encoder_t5, device):
    clip_tokens = tokenizer_clip(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        pooled_embeds = text_encoder_clip(clip_tokens, output_hidden_states=False).pooler_output

    t5_tokens = tokenizer_t5(
        captions,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        t5_embeds = text_encoder_t5(t5_tokens).last_hidden_state

    return t5_embeds, pooled_embeds


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, config=vars(args))
        os.makedirs(args.output_dir, exist_ok=True)

    # Load all FLUX components
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)

    tokenizer_clip = pipe.tokenizer
    tokenizer_t5   = pipe.tokenizer_2
    text_encoder_clip = pipe.text_encoder.to(accelerator.device)
    text_encoder_t5   = pipe.text_encoder_2.to(accelerator.device)
    vae         = pipe.vae.to(accelerator.device)
    transformer = pipe.transformer.to(accelerator.device)

    vae.requires_grad_(False)
    text_encoder_clip.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # ── Pre-compute latents + embeddings once (huge speedup) ─────────────────
    dataset = StyleDataset(args.data_dir, args.resolution)
    precompute_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Pre-computing latents + embeddings for {len(dataset)} images...")
    all_latents, all_t5, all_pooled = [], [], []

    with torch.no_grad():
        for batch in tqdm(precompute_loader, desc="Pre-computing"):
            img = batch["image"].to(accelerator.device, dtype=vae.dtype)
            lat = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
            all_latents.append(lat.cpu())

            t5_emb, pool_emb = encode_prompt(
                batch["caption"],
                tokenizer_clip, tokenizer_t5,
                text_encoder_clip, text_encoder_t5,
                accelerator.device,
            )
            all_t5.append(t5_emb.cpu())
            all_pooled.append(pool_emb.cpu())

    all_latents = torch.cat(all_latents, dim=0)   # (N, 16, H, W)
    all_t5      = torch.cat(all_t5,      dim=0)   # (N, 512, 4096)
    all_pooled  = torch.cat(all_pooled,  dim=0)   # (N, 768)
    N = all_latents.shape[0]

    # Pre-compute constant tensors (same for every batch at this resolution)
    _, c, h_lat, w_lat = all_latents.shape
    img_ids = FluxPipeline._prepare_latent_image_ids(1, h_lat, w_lat, accelerator.device, torch.bfloat16)
    txt_ids = torch.zeros(512, 3, device=accelerator.device, dtype=torch.bfloat16)

    # Free text encoders + VAE from GPU — no longer needed
    text_encoder_clip.cpu()
    text_encoder_t5.cpu()
    vae.cpu()
    torch.cuda.empty_cache()
    print("Freed text encoders and VAE from GPU.")
    # ─────────────────────────────────────────────────────────────────────────

    # Attach LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "add_q_proj", "add_k_proj", "add_v_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    steps_per_epoch = math.ceil(N / args.train_batch_size)
    num_update_steps = math.ceil(steps_per_epoch / args.gradient_accumulation_steps) * args.num_train_epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=max(1, num_update_steps // 10),
        num_training_steps=num_update_steps,
    )

    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    global_step = 0
    is_tty = sys.stdout.isatty()
    epoch_bar = tqdm(range(args.num_train_epochs), desc="Epochs", ascii=True, dynamic_ncols=False, ncols=80)

    for epoch in epoch_bar:
        transformer.train()
        indices = torch.randperm(N)
        epoch_loss = 0.0
        num_batches = 0

        # Inner step bar only in real TTY — RunPod logs can't render ANSI cursor codes
        step_bar = tqdm(range(0, N, args.train_batch_size), desc=f"Ep {epoch+1}",
                        leave=False, ascii=True, ncols=80, disable=not is_tty)
        for i in step_bar:
            idx = indices[i:i + args.train_batch_size]

            with accelerator.accumulate(transformer):
                latents     = all_latents[idx].to(accelerator.device, dtype=torch.bfloat16)
                t5_embeds   = all_t5[idx].to(accelerator.device)
                pooled_embeds = all_pooled[idx].to(accelerator.device)

                noise = torch.randn_like(latents)
                bsz   = latents.shape[0]
                t     = torch.rand(bsz, device=accelerator.device, dtype=torch.bfloat16)

                packed_latents = FluxPipeline._pack_latents(latents, bsz, c, h_lat, w_lat)
                packed_noise   = FluxPipeline._pack_latents(noise,   bsz, c, h_lat, w_lat)

                noisy = (1 - t.view(-1,1,1)) * packed_latents + t.view(-1,1,1) * packed_noise
                target = packed_noise - packed_latents

                guidance = torch.full((bsz,), 3.5, device=accelerator.device, dtype=torch.bfloat16)
                pred = transformer(
                    hidden_states=noisy,
                    timestep=t,
                    encoder_hidden_states=t5_embeds,
                    pooled_projections=pooled_embeds,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )[0]

                loss = torch.nn.functional.mse_loss(pred.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1
            if is_tty:
                step_bar.set_postfix(loss=f"{loss.item():.4f}")

            if accelerator.is_main_process and global_step % 10 == 0:
                wandb.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step})
                if not is_tty:
                    print(f"  Step {global_step} | Loss {loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{args.num_train_epochs} | Avg Loss {avg_loss:.4f}")
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

    # Save + push
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(args.output_dir)

        # Fix adapter_config.json — PEFT leaves base_model_name_or_path and task_type as null
        adapter_cfg_path = Path(args.output_dir) / "adapter_config.json"
        if adapter_cfg_path.exists():
            import json
            cfg = json.loads(adapter_cfg_path.read_text())
            cfg["base_model_name_or_path"] = args.model_id
            cfg["task_type"] = "OTHER"
            adapter_cfg_path.write_text(json.dumps(cfg, indent=2))

        # Fix README.md — PEFT writes the local cache path as base_model, HF rejects it
        readme = Path(args.output_dir) / "README.md"
        if readme.exists():
            import re
            text = readme.read_text()
            text = re.sub(r'base_model:\s*\S+', f'base_model: {args.model_id}', text)
            readme.write_text(text)

        create_repo(args.hf_repo, exist_ok=True)
        upload_folder(
            repo_id=args.hf_repo,
            folder_path=args.output_dir,
            commit_message="FLUX style LoRA adapter",
        )

        wandb.finish()
        print(f"Adapter saved to {args.output_dir} and pushed to {args.hf_repo}")


if __name__ == "__main__":
    main()
