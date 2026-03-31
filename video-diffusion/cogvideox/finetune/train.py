import os
import shutil
from pathlib import Path

import wandb
import torch
import torchvision.transforms as TT
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from transformers import AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params

from peft import LoraConfig, get_peft_model_state_dict
from huggingface_hub import upload_folder

CFG = {
    "model_id":           "THUDM/CogVideoX-2b",
    "data_root":          "/workspace/data/wan-dissolve",
    "caption_column":     "prompts.txt",
    "video_column":       "videos.txt",
    "output_dir":         "/workspace/cogvideox-dissolve-lora",
    "hub_model_id":       "chethan1988/cogvideox-dissolve-lora",

    "height":             480,
    "width":              720,
    "max_num_frames":     49,
    "fps":                8,
    "id_token":           "DISSOLVE",

    "rank":               64,
    "lora_alpha":         64,

    "train_batch_size":   1,
    "gradient_accumulation_steps": 1,
    "max_train_steps":    3000,
    "checkpointing_steps": 500,
    "mixed_precision":    "bf16",
    "gradient_checkpointing": True,
    "seed":               42,

    "lr":                 1.0,
    "prodigy_d_coef":     1.0,
    "prodigy_growth_rate": 1.02,
    "adam_beta1":         0.9,
    "adam_beta2":         0.99,
    "adam_weight_decay":  1e-4,
    "adam_epsilon":       1e-8,
    "max_grad_norm":      1.0,

    "wandb_project":      "cogvideox-dissolve",
}


class VideoDataset(Dataset):
    def __init__(self, data_root, caption_column, video_column,
                 height, width, max_num_frames, fps, id_token=""):
        self.data_root = Path(data_root)
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.id_token = id_token

        prompt_path = self.data_root / caption_column
        video_path  = self.data_root / video_column

        self.prompts = [l.strip() for l in prompt_path.read_text().splitlines() if l.strip()]
        self.video_paths = [self.data_root / l.strip() for l in video_path.read_text().splitlines() if l.strip()]

        assert len(self.prompts) == len(self.video_paths), \
            f"Mismatch: {len(self.prompts)} prompts vs {len(self.video_paths)} videos"

        self.videos = self._load_all_videos()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return {
            "prompt":  self.id_token + " " + self.prompts[idx],
            "video":   self.videos[idx],
        }

    def _load_all_videos(self):
        import decord
        decord.bridge.set_bridge("torch")
        videos = []
        for path in tqdm(self.video_paths, desc="Loading videos"):
            vr = decord.VideoReader(str(path))
            total = len(vr)
            step  = max(1, total // self.max_num_frames)
            indices = list(range(0, total, step))[:self.max_num_frames]

            frames = vr.get_batch(indices)          # [F, H, W, C]  uint8

            # CogVideoX VAE requires (F-1) % 4 == 0
            remainder = (3 + (len(frames) % 4)) % 4
            if remainder:
                frames = frames[:-remainder]

            frames = frames.permute(0, 3, 1, 2)     # [F, C, H, W]
            frames = (frames.float() - 127.5) / 127.5  # [-1, 1]
            frames = self._resize_crop(frames)
            videos.append(frames.contiguous())
        return videos

    def _resize_crop(self, frames):
        if frames.shape[3] / frames.shape[2] > self.width / self.height:
            frames = resize(frames, [self.height, int(frames.shape[3] * self.height / frames.shape[2])],
                            interpolation=InterpolationMode.BICUBIC)
        else:
            frames = resize(frames, [int(frames.shape[2] * self.width / frames.shape[3]), self.width],
                            interpolation=InterpolationMode.BICUBIC)
        h, w = frames.shape[2], frames.shape[3]
        top  = (h - self.height) // 2
        left = (w - self.width)  // 2
        return TT.functional.crop(frames, top, left, self.height, self.width)
    
def prepare_rotary_positional_embeddings(height, width, num_frames, vae_scale_factor_spatial,
                                         patch_size, attention_head_dim, device):
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width  = width  // (vae_scale_factor_spatial * patch_size)
    base_size_width  = 720 // (vae_scale_factor_spatial * patch_size)
    base_size_height = 480 // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid(
        (grid_height, grid_width), base_size_width, base_size_height
    )
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        device=device,
    )
    return freqs_cos, freqs_sin


def load_models(cfg):
    project_config = ProjectConfiguration(
        project_dir=cfg["output_dir"],
        logging_dir=os.path.join(cfg["output_dir"], "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=cfg["mixed_precision"],
        project_config=project_config,
    )
    set_seed(cfg["seed"])

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Text encoder — frozen, never trained
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(cfg["model_id"], subfolder="text_encoder")
    text_encoder.requires_grad_(False)

    # VAE — frozen, never trained
    vae = AutoencoderKLCogVideoX.from_pretrained(cfg["model_id"], subfolder="vae")
    vae.requires_grad_(False)

    # CogVideoX-2b weights are float16; 5b weights are bfloat16
    load_dtype = torch.bfloat16 if "5b" in cfg["model_id"].lower() else torch.float16

    # Transformer — this is what we train
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        cfg["model_id"], subfolder="transformer", torch_dtype=load_dtype
    )

    # Noise scheduler
    scheduler = CogVideoXDPMScheduler.from_pretrained(cfg["model_id"], subfolder="scheduler")

    # LoRA on the transformer attention layers
    lora_config = LoraConfig(
        r=cfg["rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights=True,
    )
    transformer.add_adapter(lora_config)

    if cfg["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()

    # Cast LoRA params to float32 for stable training, frozen weights stay in load_dtype
    cast_training_params([transformer], dtype=torch.float32)

    vae.enable_slicing()
    vae.enable_tiling()

    return accelerator, tokenizer, text_encoder, vae, transformer, scheduler

@torch.no_grad()
def pre_encode_videos(dataset, vae, device):
    """Encode all videos to latent distributions once before training starts."""
    vae.to(device)
    latent_dists = []
    for frames in tqdm(dataset.videos, desc="Pre-encoding videos"):
        # frames: [F, C, H, W] → unsqueeze → [1, F, C, H, W] → permute → [1, C, F, H, W]
        video = frames.unsqueeze(0).to(device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4)
        latent_dist = vae.encode(video).latent_dist
        latent_dists.append(latent_dist)
    dataset.videos = latent_dists


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompts, max_length=226):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = {k: v.to(text_encoder.device) for k, v in tokens.items()}
    prompt_embeds = text_encoder(**tokens).last_hidden_state
    return prompt_embeds   # [B, seq_len, hidden_dim]

def setup_optimizer(transformer, cfg):
    from prodigyopt import Prodigy

    # Only optimize LoRA params — frozen weights have no grad
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]

    optimizer = Prodigy(
        trainable_params,
        lr=cfg["lr"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        beta3=None,
        d_coef=cfg["prodigy_d_coef"],
        weight_decay=cfg["adam_weight_decay"],
        eps=cfg["adam_epsilon"],
        growth_rate=cfg["prodigy_growth_rate"],
        decouple=True,
        use_bias_correction=True,
        safeguard_warmup=True,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["max_train_steps"],
        eta_min=0.0,
    )

    return optimizer, lr_scheduler

def train(cfg):
    accelerator, tokenizer, text_encoder, vae, transformer, scheduler = load_models(cfg)
    optimizer, lr_scheduler = setup_optimizer(transformer, cfg)

    dataset = VideoDataset(
        data_root=cfg["data_root"],
        caption_column=cfg["caption_column"],
        video_column=cfg["video_column"],
        height=cfg["height"],
        width=cfg["width"],
        max_num_frames=cfg["max_num_frames"],
        fps=cfg["fps"],
        id_token=cfg["id_token"],
    )

    # Pre-encode all videos once — much faster than encoding every batch
    pre_encode_videos(dataset, vae, accelerator.device)

    scaling_factor = vae.config.scaling_factor

    # collate_fn defined here so it can capture scaling_factor
    def collate_fn_encoded(batch):
        prompts = [item["prompt"] for item in batch]
        # sample from latent distribution, scale, permute [B,C,F,H,W] → [B,F,C,H,W]
        latents = [item["video"].sample() * scaling_factor for item in batch]
        latents = torch.cat(latents).permute(0, 2, 1, 3, 4).float()
        return {"prompts": prompts, "latents": latents}

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn_encoded,
        num_workers=0,
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device)

    # Compute once before the loop — doesn't change per batch
    model_config = accelerator.unwrap_model(transformer).config
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    # Latent frame count after VAE temporal compression (factor of 4)
    num_latent_frames = (cfg["max_num_frames"] - 1) // 4 + 1

    image_rotary_emb = None
    if model_config.use_rotary_positional_embeddings:
        image_rotary_emb = prepare_rotary_positional_embeddings(
            height=cfg["height"],
            width=cfg["width"],
            num_frames=num_latent_frames,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_size=model_config.patch_size,
            attention_head_dim=model_config.attention_head_dim,
            device=accelerator.device,
        )

    if accelerator.is_main_process:
        wandb.init(project=cfg["wandb_project"], config=cfg, name=f"cogvideox-lora-{cfg['max_train_steps']}steps")

    total_steps = cfg["max_train_steps"]
    global_step = 0
    epoch = 0

    progress_bar = tqdm(range(total_steps), desc="Training")

    transformer.train()
    while global_step < total_steps:
        epoch += 1
        epoch_loss = 0.0
        epoch_batches = 0

        for batch in dataloader:
            if global_step >= total_steps:
                break

            with accelerator.accumulate(transformer):
                # latents already encoded: [B, F, C, H, W]
                model_input = batch["latents"].to(accelerator.device)
                prompts     = batch["prompts"]

                prompt_embeds = encode_prompt(tokenizer, text_encoder, prompts)

                bsz = model_input.shape[0]
                noise = torch.randn_like(model_input)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (bsz,), device=model_input.device,
                ).long()

                noisy_latents = scheduler.add_noise(model_input, noise, timesteps)

                with accelerator.autocast():
                    model_output = transformer(
                        hidden_states=noisy_latents,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]

                # Velocity prediction — same paradigm as FLUX
                model_pred = scheduler.get_velocity(model_output, noisy_latents, timesteps)

                # Move alphas_cumprod to GPU — scheduler keeps them on CPU by default
                alphas_cumprod = scheduler.alphas_cumprod.to(model_input.device)[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input  # clean latents, not noise
                loss = torch.mean(
                    (weights * (model_pred - target) ** 2).reshape(bsz, -1), dim=1
                ).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), cfg["max_grad_norm"])

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            epoch_batches += 1

            progress_bar.update(1)
            progress_bar.set_postfix({"epoch": epoch, "loss": f"{loss.item():.4f}", "step": global_step})

            # Per-step W&B logging — step= sets the x-axis so scale is correct
            if accelerator.is_main_process and global_step % 10 == 0:
                pg = optimizer.param_groups[0]
                effective_lr = pg.get("d", 1.0) * pg["lr"]  # Prodigy: d * base_lr
                wandb.log({
                    "train/loss":  loss.item(),
                    "train/lr":    effective_lr,
                    "train/epoch": epoch,
                }, step=global_step)

            if global_step % cfg["checkpointing_steps"] == 0:
                save_checkpoint(accelerator, transformer, cfg, global_step)

        # Per-epoch summary — logged at the last step of the epoch so x-axis aligns
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            wandb.log({"epoch/avg_loss": avg_epoch_loss, "epoch/num": epoch}, step=global_step)

    progress_bar.close()

    if accelerator.is_main_process:
        wandb.finish()

    return accelerator, transformer

def save_checkpoint(accelerator, transformer, cfg, step):
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(transformer)
    lora_layers = get_peft_model_state_dict(unwrapped)

    ckpt_dir = os.path.join(cfg["output_dir"], f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save in diffusers-compatible safetensors format
    CogVideoXPipeline.save_lora_weights(
        save_directory=ckpt_dir,
        transformer_lora_layers=lora_layers,
    )
    print(f"Saved checkpoint at step {step} → {ckpt_dir}")

    # Keep only last 3 checkpoints
    all_ckpts = sorted(Path(cfg["output_dir"]).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
    for old in all_ckpts[:-3]:
        shutil.rmtree(old)
        print(f"Removed old checkpoint: {old.name}")


def save_and_upload(accelerator, transformer, cfg):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    unwrapped = accelerator.unwrap_model(transformer)
    # Cast back to native dtype before saving
    save_dtype = torch.bfloat16 if "5b" in cfg["model_id"].lower() else torch.float16
    unwrapped = unwrapped.to(save_dtype)
    lora_layers = get_peft_model_state_dict(unwrapped)

    # Save in diffusers-compatible safetensors format — loadable with pipe.load_lora_weights()
    CogVideoXPipeline.save_lora_weights(
        save_directory=cfg["output_dir"],
        transformer_lora_layers=lora_layers,
    )
    print(f"Saved final LoRA weights → {cfg['output_dir']}")

    print(f"Uploading to HuggingFace: {cfg['hub_model_id']} ...")
    upload_folder(
        repo_id=cfg["hub_model_id"],
        folder_path=cfg["output_dir"],
        repo_type="model",
        commit_message=f"CogVideoX dissolve LoRA — {cfg['max_train_steps']} steps",
        ignore_patterns=["checkpoint-*"],
    )
    print("Upload complete.")


if __name__ == "__main__":
    accelerator, transformer = train(CFG)
    save_and_upload(accelerator, transformer, CFG)