import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import GPTConfig
from minigpt.model import MiniGPT
from minigpt.tokenizer import ByteTokenizer


config = GPTConfig()
tokenizer = ByteTokenizer()

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device:", device)

checkpoint_path = PROJECT_ROOT / "results" / "minigpt_checkpoint.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

model = MiniGPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "hello transformer."
input_ids = tokenizer.encode(prompt)

idx = torch.tensor(
    [input_ids],
    dtype=torch.long,
    device=device,
)

batch_size, seq_len = idx.shape

with torch.no_grad():
    token_emb = model.token_embedding(idx)

    pos = torch.arange(seq_len, device=device)
    pos_emb = model.position_embedding(pos)

    x = token_emb + pos_emb

    block = model.blocks[0]
    x_norm = block.ln1(x)

    _, attention_weights = block.attn(
        x_norm,
        return_attention=True,
    )

head_id = 0
attention_matrix = attention_weights[0, head_id].detach().cpu()

tokens = []
for token_id in idx[0].detach().cpu().tolist():
    token = tokenizer.decode([token_id])

    if token == " ":
        token = "space"
    elif token == "\n":
        token = "\\n"
    elif token == "\t":
        token = "\\t"

    tokens.append(token)

assets_dir = PROJECT_ROOT / "assets"
assets_dir.mkdir(exist_ok=True)

plt.figure(figsize=(8, 6))
plt.imshow(attention_matrix)
plt.colorbar(label="attention weight")
plt.xticks(range(seq_len), tokens, rotation=90)
plt.yticks(range(seq_len), tokens)
plt.xlabel("Key positions")
plt.ylabel("Query positions")
plt.title("Attention Heatmap: Layer 0, Head 0")
plt.tight_layout()

output_path = assets_dir / "attention_heatmap.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")

print("prompt:", prompt)
print("attention shape:", attention_matrix.shape)
print("saved figure to assets/attention_heatmap.png")