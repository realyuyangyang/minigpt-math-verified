# Causal Mask Leakage Test

This experiment verifies that the causal self-attention module does not leak information from future tokens.

## Test 1: Future Attention Weights

For a sequence of length T, each position should only attend to itself and previous positions. The upper-triangular part of the attention matrix should be exactly zero.

Result:

```text
max future attention: 0.0
