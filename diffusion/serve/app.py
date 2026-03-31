import os
import io
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import torch

MOCK       = os.environ.get("MOCK", "false").lower() == "true"
MODEL_ID   = os.environ.get("MODEL_ID")
STYLE_LORA = os.environ.get("STYLE_LORA_ID")
DREAM_LORA = os.environ.get("DREAM_LORA_ID")

LORA_REGISTRY = {
    "style":      STYLE_LORA,
    "dreambooth": DREAM_LORA,
}

pipe = None
lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    if not MOCK:
        missing = [k for k, v in {"MODEL_ID": MODEL_ID, "STYLE_LORA_ID": STYLE_LORA, "DREAM_LORA_ID": DREAM_LORA}.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    yield
    
app = FastAPI(title="FLUX LoRA Serve", lifespan=lifespan)

class GenerateRequest(BaseModel):
    prompt:         str
    lora_type:      str   = Field(default="style", pattern="^(style|dreambooth)$")
    lora_scale:     float = Field(default=0.85, ge=0.0, le=1.0)
    num_steps:      int   = Field(default=28,   ge=1,   le=50)
    guidance_scale: float = Field(default=3.5,  ge=1.0, le=10.0)

@app.get("/health")
def health():
    return {"status": "ok", "mock": MOCK}

@app.post("/generate")
def generate(req: GenerateRequest):
    if req.lora_type not in LORA_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown lora_type: {req.lora_type}")

    if MOCK:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(100, 149, 237))
    else:
        with lock:
            pipe.unload_lora_weights()
            pipe.load_lora_weights(LORA_REGISTRY[req.lora_type])
            pipe.fuse_lora(lora_scale=req.lora_scale)
            img = pipe(
                req.prompt,
                num_inference_steps=req.num_steps,
                guidance_scale=req.guidance_scale,
            ).images[0]
            pipe.unfuse_lora()

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg")