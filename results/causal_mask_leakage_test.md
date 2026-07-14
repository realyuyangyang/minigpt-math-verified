# Causal Mask Leakage Test

This experiment verifies that the causal self-attention module prevents information leakage from future tokens.

In a decoder-only Transformer, each token position can only attend to itself and previous positions. It must not access future tokens during training or generation.

For a sequence:

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

This corresponds to the autoregressive factorization:

```text
P(x0, x1, x2, ..., xT) = Π P(xt | x<t)
```

The experiment includes two tests.

---

## Test 1: Future Attention Weight Test

This test directly checks the attention matrix.

For a sequence of length `T`, the upper-triangular part of the attention matrix represents attention to future tokens. With a correct causal mask, all values in this region should be exactly zero after softmax.

Result:

```text
max future attention: 0.0
```

This confirms that the attention module does not assign probability mass to future positions.

---

## Test 2: Future Token Perturbation Test

This test checks whether changing future tokens affects previous predictions.

We construct two input sequences with the same prefix but different future tokens:

```text
x1 = [same prefix, original future tokens]
x2 = [same prefix, changed future tokens]
```

For example:

```text
x1 = [A, B, C, D, E, F, G, H, 1, 2, 3, 4]
x2 = [A, B, C, D, E, F, G, H, 9, 8, 7, 6]
```

The first part of the sequence is identical, while the future tokens are changed.

If the causal mask is implemented correctly, the model output logits for the shared prefix should remain unchanged:

```text
logits(x1)[:, :prefix_len, :] == logits(x2)[:, :prefix_len, :]
```

Result:

```text
max prefix logits diff after changing future tokens: 0.0
```

This means that modifying future tokens does not affect predictions at previous positions.

The suffix logits are allowed to differ because the suffix tokens themselves were changed.

---

## Conclusion

Both tests confirm that the causal mask is implemented correctly.

The attention weights show no probability mass on future positions, and the model outputs for previous positions remain unchanged even when future tokens are modified.

Therefore, the MiniGPT implementation prevents future-token information leakage and correctly follows the autoregressive language modeling constraint.
0
