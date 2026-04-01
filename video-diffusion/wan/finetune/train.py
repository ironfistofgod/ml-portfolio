"""
Wan2.1 T2V 1.3B LoRA fine-tuning launcher.
Calls finetrainers' train.py via torchrun.

Setup on RunPod before running:
  git clone https://github.com/huggingface/finetrainers /workspace/finetrainers
  cd /workspace/finetrainers && pip install -r requirements.txt
  pip install git+https://github.com/huggingface/diffusers

Then run from repo root:
  python video-diffusion/wan/finetune/train.py
"""

import os
import sys
import subprocess
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
NUM_GPUS       = 1
FINETRAINERS   = "/app/finetrainers/train.py"
DATASET_CONFIG = str(Path(__file__).parent / "training.json")
VALIDATION_FILE = str(Path(__file__).parent / "validation.json")
OUTPUT_DIR     = "/workspace/wan-dissolve-lora"
HUB_MODEL_ID   = "chethan1988/wan-dissolve-lora"

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["WANDB_MODE"]                = "online"
os.environ["NCCL_P2P_DISABLE"]          = "1"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
os.environ["FINETRAINERS_LOG_LEVEL"]    = "DEBUG"

# ── Command ───────────────────────────────────────────────────────────────────
cmd = [
    "torchrun",
    "--standalone",
    "--nnodes=1",
    f"--nproc_per_node={NUM_GPUS}",
    "--rdzv_backend", "c10d",
    "--rdzv_endpoint", "localhost:0",
    FINETRAINERS,

    # Parallel — single GPU: all degrees = 1
    "--parallel_backend", "ptd",
    "--pp_degree", "1",
    "--dp_degree", "1",
    "--dp_shards", "1",
    "--cp_degree", "1",
    "--tp_degree", "1",

    # Model
    "--model_name", "wan",
    "--pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",

    # Dataset — finetrainers downloads finetrainers/3dgs-dissolve from HuggingFace
    "--dataset_config", DATASET_CONFIG,
    "--dataset_shuffle_buffer_size", "10",
    "--enable_precomputation",
    "--precomputation_items", "101",  # encode all 101 videos (1 GPU) before training
    "--precomputation_once",         # reuse precomputed embeddings, don't redo

    # Dataloader
    "--dataloader_num_workers", "0",

    # Diffusion
    "--flow_weighting_scheme", "logit_normal",

    # Training
    "--training_type", "lora",
    "--seed", "42",
    "--batch_size", "1",
    "--train_steps", "3000",
    "--rank", "32",
    "--lora_alpha", "32",
    "--target_modules", r"blocks.*(to_q|to_k|to_v|to_out.0)",
    "--gradient_accumulation_steps", "4",
    "--gradient_checkpointing",
    "--checkpointing_steps", "500",
    "--checkpointing_limit", "2",
    "--enable_slicing",
    "--enable_tiling",

    # Optimizer
    "--optimizer", "adamw",
    "--lr", "5e-5",
    "--lr_scheduler", "constant_with_warmup",
    "--lr_warmup_steps", "75",   # 10% of 750 effective optimizer steps (3000 / grad_accum=4)
    "--beta1", "0.9",
    "--beta2", "0.99",
    "--weight_decay", "1e-4",
    "--epsilon", "1e-8",
    "--max_grad_norm", "1.0",

    # Validation — generates sample videos every 200 steps
    "--validation_dataset_file", VALIDATION_FILE,
    "--validation_steps", "500",

    # Misc
    "--tracker_name", "finetrainers-wan",
    "--output_dir", OUTPUT_DIR,
    "--hub_model_id", HUB_MODEL_ID,
    "--push_to_hub",
    "--init_timeout", "600",
    "--nccl_timeout", "600",
    "--report_to", "wandb",
]

if __name__ == "__main__":
    print("Launching Wan2.1 T2V LoRA training...")
    print(f"  GPUs:       {NUM_GPUS}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  HF repo:    {HUB_MODEL_ID}")
    print(f"  Steps:      3000 (grad_accum=4 → effective batch=4)")
    print()
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)
