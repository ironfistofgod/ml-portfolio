import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read()

# print(f"Total chararters in dataset : {len(text)}")
# print("First 200 characters:")
# print(text[:200])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"All characters: {''.join(chars)}")


# character to integer
stoi = { ch:i for i,ch in enumerate(chars) }

itos = { i:ch for i,ch in enumerate(chars)}

#encoder / decoder 
encode = lambda s: [stoi[c] for c in s]

decode = lambda l: ''.join([itos[i] for i in l])

print(encode("First Citizen"))
print(decode(encode("First Citizen")))

data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data shape: {data.shape}, dtype: {data.dtype}")

# 90% train, 10% validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

block_size = 256
batch_size = 64

def get_batch(split):
    data = train_data if split =="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1: i+block_size+1] for i in ix]).to(device)
    return x,y

xb, yb = get_batch('train')
print(f"Input shape: {xb.shape}")
print(f"Target shape: {yb.shape}")
print(f"\nInput:\n{xb}")
print(f"\nTarget:\n{yb}")

n_embd = 384
class Head(nn.Module):
    
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks                   = nn.Sequential(*[Block(n_embd, n_head=6) for _ in range(6)])
        self.ln_f                     = nn.LayerNorm(n_embd)
        self.lm_head                  = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                      # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T))       # (T, n_embd)
        x = tok_emb + pos_emb                                          # (B, T, n_embd)
        x = self.blocks(x)                                             # (B, T, n_embd)
        x = self.ln_f(x)                                               # (B, T, n_embd)
        logits = self.lm_head(x)                                       # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = BigramLanguageModel(vocab_size).to(device)
logits, loss = model(xb, yb)
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item()}")


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for steps in range(5000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(f"Final loss: {loss.item()}")

context = torch.zeros((1, 1), dtype=torch.long).to(device)  # start with character 0 ('\n')

output = model.generate(context, max_new_tokens=200)
print(decode(output[0].tolist()))



        