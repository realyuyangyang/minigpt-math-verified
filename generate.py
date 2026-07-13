import torch

from config import GPTConfig
from minigpt.model import MiniGPT
from minigpt.tokenizer import ByteTokenizer


config = GPTConfig()
tokenizer = ByteTokenizer()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device:", device)

checkpoint = torch.load(
    "results/minigpt_checkpoint.pt",
    map_location=device,
)

model = MiniGPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "hello"
input_ids = tokenizer.encode(prompt)

idx = torch.tensor([input_ids], dtype=torch.long).to(device)

generated = model.generate(
    idx=idx,
    max_new_tokens=config.max_new_tokens,
    temperature=config.temperature,
    top_k=config.top_k,
)

text = tokenizer.decode(generated[0].tolist())

print("\n=== Prompt ===")
print(prompt)

print("\n=== Generated Text ===")
print(text)