import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import GPTConfig
from minigpt.attention import CausalSelfAttention
from minigpt.model import MiniGPT


torch.manual_seed(42)

config = GPTConfig()
config.dropout = 0.0

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("device:", device)


def test_attention_weights():
    attn = CausalSelfAttention(config).to(device)
    attn.eval()

    batch_size = 2
    seq_len = 16

    x = torch.randn(
        batch_size,
        seq_len,
        config.n_embd,
        device=device,
    )

    _, attention_weights = attn(x, return_attention=True)

    future_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1,
    ).bool()

    future_attention = attention_weights[:, :, future_mask]
    max_future_attention = future_attention.max().item()

    print("max future attention:", max_future_attention)

    assert max_future_attention == 0.0

    return max_future_attention


def test_future_token_leakage():
    model = MiniGPT(config).to(device)
    model.eval()

    batch_size = 1
    seq_len = 16
    prefix_len = 8

    # Step 1: Create the first input sequence.
    x1 = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )

    # Step 2: Copy x1 to x2.
    x2 = x1.clone()

    # Step 3: Keep the prefix unchanged, but replace the future tokens.
    x2[:, prefix_len:] = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len - prefix_len),
        device=device,
    )

    print("\nTest 2: Future Token Perturbation")
    print("x1:", x1[0].tolist())
    print("x2:", x2[0].tolist())
    print("shared prefix x1:", x1[0, :prefix_len].tolist())
    print("shared prefix x2:", x2[0, :prefix_len].tolist())
    print("changed future x1:", x1[0, prefix_len:].tolist())
    print("changed future x2:", x2[0, prefix_len:].tolist())

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # Prefix logits should be identical.
    # If they differ, future tokens leaked into previous positions.
    prefix_diff = (
        logits1[:, :prefix_len, :] - logits2[:, :prefix_len, :]
    ).abs().max().item()

    # Suffix logits are allowed to differ because future tokens were changed.
    suffix_diff = (
        logits1[:, prefix_len:, :] - logits2[:, prefix_len:, :]
    ).abs().max().item()

    print("max prefix logits diff after changing future tokens:", prefix_diff)
    print("max suffix logits diff after changing future tokens:", suffix_diff)

    assert prefix_diff < 1e-6

    return prefix_diff, suffix_diff


def save_result_figure(max_future_attention, prefix_diff, suffix_diff):
    os.makedirs(PROJECT_ROOT / "assets", exist_ok=True)

    plt.figure()
    plt.bar(
        ["future attention", "prefix logit diff", "suffix logit diff"],
        [max_future_attention, prefix_diff, suffix_diff],
    )
    plt.ylabel("max absolute value")
    plt.title("Causal Mask Leakage Test")
    plt.savefig(
        PROJECT_ROOT / "assets" / "mask_leakage_test.png",
        dpi=200,
        bbox_inches="tight",
    )

    print("saved figure to assets/mask_leakage_test.png")


def main():
    max_future_attention = test_attention_weights()
    prefix_diff, suffix_diff = test_future_token_leakage()

    save_result_figure(
        max_future_attention,
        prefix_diff,
        suffix_diff,
    )

    print("causal mask leakage test passed")


if __name__ == "__main__":
    main()