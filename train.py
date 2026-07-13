import os

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import GPTConfig
from minigpt.dataset import TextDataset
from minigpt.model import MiniGPT


config = GPTConfig()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device:", device)

dataset = TextDataset(
    file_path="data/input.txt",
    block_size=config.block_size,
)

model = MiniGPT(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
)

train_losses = []
val_losses = []


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(config.eval_iters):
            x, y = dataset.get_batch(
                split=split,
                batch_size=config.batch_size,
                device=device,
            )
            _, loss = model(x, y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


model.train()

for step in tqdm(range(config.max_iters)):
    x, y = dataset.get_batch(
        split="train",
        batch_size=config.batch_size,
        device=device,
    )

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % config.eval_interval == 0 or step == config.max_iters - 1:
        losses = estimate_loss()

        train_losses.append(losses["train"])
        val_losses.append(losses["val"])

        print(
            f"step {step}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )


os.makedirs("results", exist_ok=True)
os.makedirs("assets", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    },
    "results/minigpt_checkpoint.pt",
)

plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("evaluation step")
plt.ylabel("loss")
plt.legend()
plt.title("MiniGPT Training Loss")
plt.savefig("assets/loss_curve.png", dpi=200, bbox_inches="tight")

print("saved checkpoint to results/minigpt_checkpoint.pt")
print("saved loss curve to assets/loss_curve.png")