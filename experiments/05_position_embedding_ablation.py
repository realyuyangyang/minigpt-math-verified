import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import GPTConfig
from minigpt.dataset import TextDataset
from minigpt.model import MiniGPT


torch.manual_seed(42)

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device:", device)


def make_config(use_position_embedding: bool):
    config = GPTConfig()

    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 128
    config.block_size = 64
    config.batch_size = 16
    config.max_iters = 500
    config.eval_interval = 50
    config.eval_iters = 20
    config.learning_rate = 3e-4
    config.dropout = 0.1
    config.use_position_embedding = use_position_embedding

    return config


@torch.no_grad()
def estimate_loss(model, dataset, config):
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


def train_variant(use_position_embedding: bool):
    config = make_config(use_position_embedding)

    label = (
        "with position embedding"
        if use_position_embedding
        else "without position embedding"
    )

    print(f"\nTraining variant: {label}")

    dataset = TextDataset(
        file_path=str(PROJECT_ROOT / "data" / "input.txt"),
        block_size=config.block_size,
    )

    model = MiniGPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    train_losses = []
    val_losses = []
    eval_steps = []

    model.train()

    for step in tqdm(range(config.max_iters)):
        x, y = dataset.get_batch(
            split="train",
            batch_size=config.batch_size,
            device=device,
        )

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config)

            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            eval_steps.append(step)

            print(
                f"{label} | "
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

    return {
        "label": label,
        "eval_steps": eval_steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


with_pos = train_variant(use_position_embedding=True)
without_pos = train_variant(use_position_embedding=False)

assets_dir = PROJECT_ROOT / "assets"
assets_dir.mkdir(exist_ok=True)

plt.figure()
plt.plot(
    with_pos["eval_steps"],
    with_pos["val_losses"],
    marker="o",
    label="with position embedding",
)
plt.plot(
    without_pos["eval_steps"],
    without_pos["val_losses"],
    marker="o",
    label="without position embedding",
)
plt.xlabel("training step")
plt.ylabel("validation loss")
plt.title("Position Embedding Ablation")
plt.legend()
plt.savefig(
    assets_dir / "position_embedding_ablation.png",
    dpi=200,
    bbox_inches="tight",
)

print("saved figure to assets/position_embedding_ablation.png")
print("position embedding ablation completed")