import modal
from pydantic import BaseModel, Field

app = modal.App("cogvideox-serve")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch==2.4.0",
        "diffusers==0.31.0",
        "transformers==4.45.2",
        "accelerate",
        "peft",
        "imageio",
        "imageio-ffmpeg",
        "sentencepiece",
        "huggingface_hub",
    )
)

MODEL_ID = "THUDM/CogVideoX-2b"
LORA_ID  = "chethan1988/cogvideox-dissolve-lora"


class GenerateRequest(BaseModel):
    prompt:              str
    num_frames:          int   = Field(default=49,  ge=1,  le=97)
    num_inference_steps: int   = Field(default=50,  ge=1,  le=100)
    guidance_scale:      float = Field(default=6.0, ge=1.0, le=15.0)
    fps:                 int   = Field(default=8,   ge=1,  le=30)


@app.cls(gpu="A100", image=image, timeout=600)
class CogVideoXServe:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import CogVideoXPipeline

        self.pipe = CogVideoXPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.load_lora_weights(LORA_ID)

    @modal.fastapi_endpoint(method="POST")
    def generate(self, req: GenerateRequest):
        from diffusers.utils import export_to_video
        from fastapi.responses import Response

        video_frames = self.pipe(
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

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok"}
