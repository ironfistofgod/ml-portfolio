import os
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

app = FastAPI()

base_model_id = "unsloth/Meta-Llama-3.1-8B"
lora_model_id = os.environ["MODEL_ID"]

llm = LLM(
    model=base_model_id,
    dtype="bfloat16",
    max_model_len=1024,
    enable_lora=True,
)

# vLLM's LLM class is not thread-safe — serialize concurrent requests
lock = threading.Lock()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    with lock:
        outputs = llm.generate(
            [req.prompt],
            params,
            lora_request=LoRARequest("llama-coder", 1, lora_model_id)
        )
    return {"generated": outputs[0].outputs[0].text}
