# Attention Scaling Variance Experiment

This experiment verifies why scaled dot-product attention divides the raw attention logits by `sqrt(d_k)`.

In Transformer self-attention, the attention score between a query vector `q` and a key vector `k` is computed as a dot product:

```text
q · k
```

For a full sequence, this is written as:

```text
QK^T
```

However, when the head dimension `d_k` becomes large, the variance of the raw dot-product logits increases. This can make the softmax distribution too sharp and lead to unstable gradients.

Scaled dot-product attention solves this by using:

```text
QK^T / sqrt(d_k)
```

---

## Mathematical Intuition

Assume each element of `q` and `k` is independently sampled from a standard normal distribution:

```text
q_i ~ N(0, 1)
k_i ~ N(0, 1)
```

The dot product is:

```text
q · k = q_1 k_1 + q_2 k_2 + ... + q_dk k_dk
```

Each term `q_i k_i` has approximately:

```text
E[q_i k_i] = 0
Var(q_i k_i) = 1
```

Since the dot product is the sum of `d_k` independent terms, its variance grows approximately linearly with `d_k`:

```text
Var(q · k) ≈ d_k
```

Therefore, as `d_k` increases, the raw attention logits become larger in magnitude.

To stabilize the variance, Transformer attention divides the dot product by `sqrt(d_k)`:

```text
(q · k) / sqrt(d_k)
```

The variance then becomes:

```text
Var((q · k) / sqrt(d_k)) ≈ Var(q · k) / d_k ≈ 1
```

So the scaling factor keeps the attention logits at a stable scale across different head dimensions.

---

## Experiment Design

We simulate random query and key vectors with different head dimensions:

```text
d_k = 8, 16, 32, 64, 128, 256, 512, 1024
```

For each `d_k`, we compute two quantities:

```text
raw logits    = q · k
scaled logits = (q · k) / sqrt(d_k)
```

Then we estimate the variance of both raw logits and scaled logits.

---

## Expected Result

Without scaling:

```text
Var(q · k) grows approximately with d_k
```

With `1 / sqrt(d_k)` scaling:

```text
Var((q · k) / sqrt(d_k)) remains close to 1
```

This demonstrates that the scaling factor prevents attention logits from growing too large as the head dimension increases.

---

## Observed Result

The experiment shows the expected pattern:

```text
without scaling: variance increases as d_k increases
with scaling: variance stays close to 1
```

This confirms the theoretical reason for using the `1 / sqrt(d_k)` factor in scaled dot-product attention.

---

## Why This Matters

If the attention logits become too large, the softmax function can become saturated.

For example, large logits may produce an attention distribution close to one-hot:

```text
[0.001, 0.002, 0.997]
```

This means the model may attend too strongly to a single token early in training, which can make optimization less stable.

By scaling the logits, the model keeps the softmax input in a more stable range, producing smoother attention distributions and healthier gradients.

---

## Conclusion

This experiment verifies that:

```text
QK^T has variance approximately proportional to d_k
```

while:

```text
QK^T / sqrt(d_k) has variance approximately close to 1
```

Therefore, the `1 / sqrt(d_k)` scaling factor is not an arbitrary trick. It is a variance stabilization mechanism that keeps attention logits numerically stable across different head dimensions.

This is one of the key mathematical details behind the stability of Transformer self-attention.

