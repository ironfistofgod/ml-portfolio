import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import urllib.request
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

if not os.path.exists('input.txt'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'input.txt'
    )

from model import BigramLanguageModel, vocab_size, encode, decode

app = FastAPI()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = hf_hub_download(repo_id="chethan1988/gpt-shakespeare", filename="model.pt")
model = BigramLanguageModel(vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = 200


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    context = torch.tensor([encode(req.prompt)], dtype=torch.long).to(device)
    output = model.generate(context, max_new_tokens=req.max_tokens)
    return {"text": decode(output[0].tolist())}
