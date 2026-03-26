import os
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

model_id = os.environ["MODEL_ID"]

llm = LLM(
    model=model_id,
    dtype="bfloat16",
    max_model_len=1024,
)

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
    outputs = llm.generate([req.prompt], params)
    return {"generated": outputs[0].outputs[0].text}
