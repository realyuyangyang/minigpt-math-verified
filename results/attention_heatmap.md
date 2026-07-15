# Attention Heatmap Visualization

This experiment visualizes the attention matrix from a trained MiniGPT model.

The goal is to inspect whether the causal self-attention module follows the autoregressive constraint: each token should only attend to itself and previous tokens, but not future tokens.

---

## Background

In a decoder-only Transformer, self-attention is causal.

For an input sequence:

```text
[x0, x1, x2, x3]
```

the allowed attention pattern is:

```text
x0 -> x0
x1 -> x0, x1
x2 -> x0, x1, x2
x3 -> x0, x1, x2, x3
```

This means the attention matrix should have a lower-triangular structure.

The upper-triangular region corresponds to attention from current positions to future positions, which should be masked out.

---

## Method

We load the trained MiniGPT checkpoint and pass a short prompt into the model:

```text
hello transformer.
```

Then we extract the attention weights from:

```text
Layer 0, Head 0
```

The attention matrix has shape:

```text
sequence_length × sequence_length
```

Rows represent query positions.

Columns represent key positions.

Each value shows how much a query token attends to a key token.

---

## Expected Pattern

With a correct causal mask, each token should only attend to itself and previous tokens.

Therefore, the heatmap should show a lower-triangular pattern:

```text
allowed region: lower triangle
masked region: upper triangle
```

The upper-triangular region should have zero or near-zero attention weight.

---

## Result

The generated heatmap is saved to:

```text
assets/attention_heatmap.png
```

The visualization shows that attention weights are concentrated in the lower-triangular region.

This confirms that future positions are masked and the model follows the causal attention constraint.

---

## Interpretation

The attention heatmap provides an intuitive visual check of the causal mask.

Compared with the causal mask leakage test, this experiment is more visual:

* The leakage test checks the numerical correctness of masking.
* The heatmap shows the attention structure directly.

Together, they verify that the MiniGPT implementation prevents future-token information leakage.

---

## Conclusion

This experiment confirms that the implemented causal self-attention module follows the autoregressive structure required by decoder-only language models.

Each token only attends to itself and previous tokens.

Future tokens are masked out, ensuring that next-token prediction does not use information from the future.

