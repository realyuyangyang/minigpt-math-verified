from pathlib import Path

import torch
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

torch.manual_seed(42)

d_k_values = [8, 16, 32, 64, 128, 256, 512, 1024]
num_samples = 20000

raw_variances = []
scaled_variances = []


def estimate_variance(d_k):
    q = torch.randn(num_samples, d_k)
    k = torch.randn(num_samples, d_k)

    raw_logits = (q * k).sum(dim=-1)
    scaled_logits = raw_logits / (d_k ** 0.5)

    raw_var = raw_logits.var(unbiased=False).item()
    scaled_var = scaled_logits.var(unbiased=False).item()

    return raw_var, scaled_var


for d_k in d_k_values:
    raw_var, scaled_var = estimate_variance(d_k)

    raw_variances.append(raw_var)
    scaled_variances.append(scaled_var)

    print(
        f"d_k={d_k:4d} | "
        f"Var(QK^T)={raw_var:8.4f} | "
        f"Var(QK^T / sqrt(d_k))={scaled_var:8.4f}"
    )


assets_dir = PROJECT_ROOT / "assets"
assets_dir.mkdir(exist_ok=True)

plt.figure()
plt.plot(d_k_values, raw_variances, marker="o", label="without scaling")
plt.plot(d_k_values, scaled_variances, marker="o", label="with 1/sqrt(d_k) scaling")
plt.xscale("log", base=2)
plt.xlabel("head dimension d_k")
plt.ylabel("variance of attention logits")
plt.title("Attention Scaling Variance Experiment")
plt.legend()
plt.savefig(
    assets_dir / "attention_scaling_variance.png",
    dpi=200,
    bbox_inches="tight",
)

print("saved figure to assets/attention_scaling_variance.png")