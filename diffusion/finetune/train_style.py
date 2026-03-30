import os
import argparse
import math
import torch
import wandb
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        
        caption = caption_path.read_text().strip() if caption_path.exists() else "a painting in ghibli style"
        
        return {"image": image, "caption": caption}
    
    
def encode_prompt(caption, tokenizer_clip, tokenizer_t5, text_encoder_clip, text_encoder_t5, device):
    # CLIP encoding — max 77 tokens
    clip_tokens = tokenizer_clip(
        caption,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        pooled_embeds = text_encoder_clip(clip_tokens, output_hidden_states=False).pooler_output

    # T5 encoding — max 512 tokens
    t5_tokens = tokenizer_t5(
        caption,
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
    scheduler   = pipe.scheduler

    # Freeze everything except the transformer
    vae.requires_grad_(False)
    text_encoder_clip.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)

    # Enable gradient checkpointing on transformer
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
        
    # Attach LoRA to the transformer's attention layers
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
        
    # Optimizer — only LoRA parameters, not the frozen weights
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # Dataset and dataloader
    dataset = StyleDataset(args.data_dir, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
    )

    # LR scheduler — cosine decay
    num_update_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps) * args.num_train_epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_update_steps,
    )

    # Hand everything to accelerator
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    global_step = 0

    for epoch in range(args.num_train_epochs):
        transformer.train()

        for batch in dataloader:
            with accelerator.accumulate(transformer):

                # 1. Encode images to latent space via VAE
                images = batch["image"].to(accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 2. Sample random noise and timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Cast t to bfloat16 to match latents dtype and avoid float32 promotion
                t = torch.rand(bsz, device=accelerator.device, dtype=latents.dtype)

                # 3. Pack latents for FLUX transformer: (B,16,H,W) -> (B, H/2*W/2, 64)
                _, c, h_lat, w_lat = latents.shape
                packed_latents = FluxPipeline._pack_latents(latents, bsz, c, h_lat, w_lat)
                packed_noise   = FluxPipeline._pack_latents(noise,   bsz, c, h_lat, w_lat)

                # 4. Flow matching on packed latents
                noisy_latents = (1 - t.view(-1,1,1)) * packed_latents + t.view(-1,1,1) * packed_noise
                target = packed_noise - packed_latents

                # 5. Encode captions
                t5_embeds, pooled_embeds = encode_prompt(
                    batch["caption"],
                    tokenizer_clip, tokenizer_t5,
                    text_encoder_clip, text_encoder_t5,
                    accelerator.device,
                )

                # 6. Build FLUX positional IDs
                img_ids = FluxPipeline._prepare_latent_image_ids(
                    bsz, h_lat, w_lat, accelerator.device, latents.dtype
                )
                txt_ids = torch.zeros(512, 3, device=accelerator.device, dtype=latents.dtype)

                # 7. Predict velocity (FLUX.1-dev requires guidance tensor)
                guidance = torch.full((bsz,), 3.5, device=accelerator.device, dtype=latents.dtype)
                pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=t,
                    encoder_hidden_states=t5_embeds,
                    pooled_projections=pooled_embeds,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )[0]

                # 6. Flow matching loss
                loss = torch.nn.functional.mse_loss(pred.float(), target.float())

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process and global_step % 50 == 0:
                wandb.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step})
                print(f"Epoch {epoch} | Step {global_step} | Loss {loss.item():.4f}")
                
    # Save LoRA adapter
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(args.output_dir)

        # Create HF repo if it doesn't exist, then push
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