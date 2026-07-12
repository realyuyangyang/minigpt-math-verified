import torch

from config import GPTConfig
from minigpt.model import MiniGPT


config = GPTConfig()
model = MiniGPT(config)

x = torch.randint(
            low=0,
                high=config.vocab_size,
                    size=(4, config.block_size),
                    )

y = torch.randint(
            low=0,
                high=config.vocab_size,
                    size=(4, config.block_size),
                    )

logits, loss = model(x, y)

print("x shape:", x.shape)
print("logits shape:", logits.shape)
print("loss:", loss.item())

generated = model.generate(
            idx=x[:1, :10],
                max_new_tokens=20,
                )

print("generated shape:", generated.shape)
