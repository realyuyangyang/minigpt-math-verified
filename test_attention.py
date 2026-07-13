import torch

from config import GPTConfig
from minigpt.attention import CausalSelfAttention


config = GPTConfig()
attn = CausalSelfAttention(config)

x = torch.randn(2, 8, config.n_embd)

y, attention_weights = attn(x, return_attention=True)

print("input shape:", x.shape)
print("output shape:", y.shape)
print("attention shape:", attention_weights.shape)

future_attention = attention_weights[:, :, torch.triu(torch.ones(8, 8), diagonal=1) == 1]

print("max future attention:", future_attention.max().item())
