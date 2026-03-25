import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

from model import BigramLanguageModel, vocab_size, encode, decode, block_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

wandb.init(
    project="gpt-shakespeare",
    config={
        "block_size": 256,
        "batch_size": 64,
        "n_embd": 384,
        "n_head": 6,
        "n_blocks": 6,
        "learning_rate": 1e-3,
        "steps": 5000,
        "device": device,
    }
)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Vocabulary size: {vocab_size}")

data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data shape: {data.shape}, dtype: {data.dtype}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

batch_size = 64

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:   i+block_size]   for i in ix]).to(device)
    y = torch.stack([data[i+1: i+block_size+1] for i in ix]).to(device)
    return x, y

xb, yb = get_batch('train')
print(f"Input shape: {xb.shape}, Target shape: {yb.shape}")

model = BigramLanguageModel(vocab_size).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

logits, loss = model(xb, yb)
print(f"Initial loss: {loss.item():.4f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(5000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        wandb.log({"loss": loss.item(), "step": steps})
        print(f"Step {steps}: loss {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")
wandb.log({"final_loss": loss.item()})

torch.save(model.state_dict(), 'model.pt')
print("Model saved to model.pt")

from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='model.pt',
    path_in_repo='model.pt',
    repo_id='chethan1988/gpt-shakespeare',
    repo_type='model',
    token=os.environ.get('HF_TOKEN')
)
print("Model uploaded to HuggingFace")

context = torch.zeros((1, 1), dtype=torch.long).to(device)
output = model.generate(context, max_new_tokens=200)
print(decode(output[0].tolist()))

wandb.finish()
