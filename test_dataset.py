from config import GPTConfig
from minigpt.dataset import TextDataset


config = GPTConfig()
dataset = TextDataset("data/input.txt", block_size=config.block_size)

x, y = dataset.get_batch(
    split="train",
    batch_size=4,
    device="cpu",
)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("x[0]:", x[0][:20])
print("y[0]:", y[0][:20])
