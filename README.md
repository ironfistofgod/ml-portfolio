# ML Portfolio

End-to-end AI/ML engineering portfolio covering fine-tuning, inference optimization, and production serving of large generative models on Kubernetes. Each project includes a custom training pipeline, containerized serving layer, and a full CI/CD deployment to GKE.

---

## Projects

### 1. LLaMA-3.2 — Instruction Fine-Tuning + GKE Serving
`llama/`

Fine-tuned [Meta-Llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) on a custom instruction dataset using supervised fine-tuning (SFT) with DeepSpeed ZeRO-2.

**Training**
- Custom `train.py` with HuggingFace `Trainer` + DeepSpeed config
- Mixed precision (bf16), gradient checkpointing, gradient accumulation
- Dockerized training image pushed to GHCR, launched on RunPod A100

**Serving**
- FastAPI inference server (`serve/serve.py`) with tokenizer + model loaded at startup
- Deployed on **GKE** via Helm chart with PVC-backed model cache, HPA, PDB, RBAC
- GitHub Actions workflow: build Docker image → push GHCR → `helm upgrade` to GKE

---

### 2. FLUX.1-dev — DreamBooth + Style LoRA Fine-Tuning + GKE Serving
`diffusion/`

Fine-tuned [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) using two separate LoRA adapters — one for subject (DreamBooth) and one for artistic style — both trained with the **Prodigy optimizer** (learning-rate free).

**Training**
- `train_dreambooth.py` — subject-specific LoRA (rank 16) using prior-preservation loss
- `train_style.py` — style LoRA with curated image dataset
- Prodigy optimizer with `d_coef=1.0`, gradient accumulation ×4, bf16 mixed precision
- W&B loss logging with correct step-scale on x-axis
- 1500 steps; checkpoint saved every 250 steps to `/workspace` network volume
- LoRA weights uploaded to HuggingFace Hub after training: [`chethan1988/flux-dreambooth-lora-v4`](https://huggingface.co/chethan1988/flux-dreambooth-lora-v4)

**Serving**
- FastAPI app loads base FLUX pipeline + dynamically swaps LoRA adapter per request (`lora_type: "dreambooth" | "style"`)
- Deployed on **GKE A100 40GB** via Helm; GitHub Actions builds GHCR image and runs `helm upgrade` on push to `main`

```bash
curl -X POST https://flux-serve.<ip>.nip.io/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a photo of sks man hiking at golden hour", "lora_type": "dreambooth"}' \
  --output result.jpg
```

---

### 3. CogVideoX-2b — Text-to-Video LoRA Fine-Tuning + GKE Serving
`video-diffusion/cogvideox/`

Fine-tuned [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) on the [`finetrainers/3dgs-dissolve`](https://huggingface.co/datasets/finetrainers/3dgs-dissolve) dataset (101 videos of 3D Gaussian Splatting dissolve effects) using a **custom training loop written from scratch**.

**Training**
- `prepare_data.py` — downloads dataset, pre-encodes all 101 videos to VAE latents + T5 text embeddings, saves to disk to avoid re-encoding each epoch
- `train.py` — full training loop:
  - CogVideoX 3D DiT transformer with LoRA (rank 128, alpha 128, `peft`)
  - Flow matching loss (`v-prediction` target, not noise prediction)
  - Prodigy optimizer with `d_coef=1.0`, `weight_decay=1e-3`
  - bf16 mixed precision; LoRA params cast to float32 for numerical stability
  - Gradient accumulation ×4, gradient clipping
  - tqdm progress bar with epoch/step/loss display
  - W&B logging: `train/loss`, `train/epoch`, `train/lr` with correct step-scale
  - Checkpointing every 500 steps (last 3 retained); final LoRA uploaded to HuggingFace: [`chethan1988/cogvideox-dissolve-lora`](https://huggingface.co/chethan1988/cogvideox-dissolve-lora)
  - 3000 steps total (~3 hours on A100 80GB)

**Dataset**
| Property | Value |
|---|---|
| Dataset | `finetrainers/3dgs-dissolve` |
| Videos | 101 × 5s clips |
| Resolution | 480 × 720 |
| Trigger word | `DISSOLVE` |
| Pre-encoding | VAE latents + T5 embeddings cached to `/workspace/data` |

**Serving**
- FastAPI server with `CogVideoXPipeline` + LoRA loaded at startup; returns raw `.mp4` bytes
- Deployed on **GKE A100 40GB** via Helm; same GitHub Actions pattern as FLUX

```bash
curl -X POST https://cogvideox-serve.<ip>.nip.io/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "DISSOLVE a mountain landscape shattering into glowing particles", "num_frames": 49}' \
  --output output.mp4
```

---

### 4. Wan2.1 T2V 1.3B — Text-to-Video LoRA Fine-Tuning
`video-diffusion/wan/`

Fine-tuned [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) on the same `3dgs-dissolve` dataset using the [finetrainers](https://github.com/a-r-r-o-w/finetrainers) library as a launcher.

**Training**
- `train.py` — Python launcher that configures and calls `finetrainers` training CLI
- AdamW optimizer, lr `2e-4`, cosine schedule, warmup 75 steps
- Gradient accumulation ×4, LoRA rank 32 / alpha 32
- W&B logging with epoch-scaled x-axis
- Checkpointing every 500 steps; 3000 total steps

---

## Infrastructure

All models follow the same pattern: training on **RunPod** (A100 on-demand), weights pushed to **HuggingFace Hub**, and serving on **GKE** via Helm + GitHub Actions CI/CD. Each project has its own Helm chart, FastAPI inference server, and GHA workflow that builds a Docker image to GHCR and deploys on push to `main`.

---

## Tech Stack

| Category | Tools |
|---|---|
| Training | PyTorch, HuggingFace Diffusers, Transformers, Accelerate, PEFT, finetrainers |
| Optimizers | Prodigy (learning-rate free), AdamW |
| Mixed Precision | bf16 (training), float16 (serving) |
| Experiment Tracking | Weights & Biases |
| Model Registry | HuggingFace Hub |
| Containerization | Docker, GHCR |
| Training Infra | RunPod (A100 80GB on-demand) |
| Serving Infra | GKE (A100 40GB node pool), Helm, FastAPI, uvicorn |
| CI/CD | GitHub Actions |
| Cloud | GCP (GKE, GCE Load Balancer, Persistent Disk, Static IPs) |

---

## HuggingFace Models

| Model | Repo |
|---|---|
| FLUX DreamBooth LoRA | [chethan1988/flux-dreambooth-lora-v4](https://huggingface.co/chethan1988/flux-dreambooth-lora-v4) |
| CogVideoX Dissolve LoRA | [chethan1988/cogvideox-dissolve-lora](https://huggingface.co/chethan1988/cogvideox-dissolve-lora) |
