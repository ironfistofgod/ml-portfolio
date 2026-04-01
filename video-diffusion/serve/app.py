import io
import os
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

MOCK     = os.environ.get("MOCK", "false").lower() == "true"
MODEL_ID = os.environ.get("MODEL_ID", "THUDM/CogVideoX-2b")
LORA_ID  = os.environ.get("LORA_ID", "chethan1988/cogvideox-dissolve-lora")

pipe = None
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    if not MOCK:
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video  # noqa — ensure available at startup
        pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
        pipe.load_lora_weights(LORA_ID)
    yield


app = FastAPI(title="CogVideoX LoRA Serve", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt:              str
    lora_scale:          float = Field(default=1.0,  ge=0.0, le=2.0)
    num_frames:          int   = Field(default=49,   ge=1,   le=97)
    num_inference_steps: int   = Field(default=50,   ge=1,   le=100)
    guidance_scale:      float = Field(default=6.0,  ge=1.0, le=15.0)
    fps:                 int   = Field(default=8,    ge=1,   le=30)


@app.get("/health")
def health():
    return {"status": "ok", "mock": MOCK}


@app.post("/generate")
def generate(req: GenerateRequest):
    if MOCK:
        return Response(content=b"fake-video", media_type="video/mp4")

    from diffusers.utils import export_to_video

    with lock:
        video_frames = pipe(
            prompt=req.prompt,
            num_frames=req.num_frames,
            height=480,
            width=720,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        ).frames[0]

    tmp_path = "/tmp/output.mp4"
    export_to_video(video_frames, tmp_path, fps=req.fps)

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    return Response(content=video_bytes, media_type="video/mp4")
