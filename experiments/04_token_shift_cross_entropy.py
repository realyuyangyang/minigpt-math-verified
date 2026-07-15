import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import GPTConfig
from minigpt.model import MiniGPT
from minigpt.tokenizer import ByteTokenizer


torch.manual_seed(42)

config = GPTConfig()
tokenizer = ByteTokenizer()

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device:", device)

text = "hello transformer."
token_ids = tokenizer.encode(text)

x_ids = token_ids[:-1]
y_ids = token_ids[1:]

x = torch.tensor([x_ids], dtype=torch.long, device=device)
y = torch.tensor([y_ids], dtype=torch.long, device=device)

model = MiniGPT(config).to(device)
model.eval()

with torch.no_grad():
    logits, model_loss = model(x, y)

batch_size, seq_len, vocab_size = logits.shape

manual_loss = F.cross_entropy(
    logits.view(batch_size * seq_len, vocab_size),
    y.view(batch_size * seq_len),
)

print("\nToken Shift Example")
print("original text:", text)
print("input ids :", x_ids)
print("target ids:", y_ids)

print("\nInput -> Target")
for i in range(len(x_ids)):
    input_token = tokenizer.decode([x_ids[i]])
    target_token = tokenizer.decode([y_ids[i]])

    if input_token == " ":
        input_token = "<space>"
    if target_token == " ":
        target_token = "<space>"

    print(f"position {i:02d}: {repr(input_token):>12s} -> {repr(target_token):>12s}")

print("\nLoss Check")
print("model loss :", model_loss.item())
print("manual loss:", manual_loss.item())
print("absolute difference:", abs(model_loss.item() - manual_loss.item()))

assert torch.allclose(model_loss, manual_loss)

assets_dir = PROJECT_ROOT / "assets"
assets_dir.mkdir(exist_ok=True)

positions = list(range(len(x_ids)))
input_labels = []
target_labels = []

for token_id in x_ids:
    token = tokenizer.decode([token_id])
    input_labels.append("<space>" if token == " " else token)

for token_id in y_ids:
    token = tokenizer.decode([token_id])
    target_labels.append("<space>" if token == " " else token)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

table_data = [
    ["position"] + [str(i) for i in positions],
    ["input x"] + input_labels,
    ["target y"] + target_labels,
]

table = ax.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title("Token Shift in Next-Token Prediction")
plt.savefig(
    assets_dir / "token_shift_cross_entropy.png",
    dpi=200,
    bbox_inches="tight",
)

print("saved figure to assets/token_shift_cross_entropy.png")
print("token shift and cross-entropy test passed")