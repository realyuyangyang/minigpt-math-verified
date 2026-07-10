# MiniGPT Reproduction with Mathematical Verification

This project reproduces a decoder-only Transformer from scratch in PyTorch and verifies its core mechanisms through mathematical and controlled experiments.

## Goals

- Implement a minimal GPT-style decoder-only Transformer from scratch.
- Understand token embeddings, positional embeddings, causal self-attention, Pre-LN Transformer blocks, and next-token prediction.
- Verify key Transformer mechanisms through experiments, including attention scaling, causal masking, positional encoding, and token shifting.

## Planned Experiments

- Attention scaling variance experiment
- Causal mask leakage test
- Attention heatmap visualization
- Position embedding ablation
- Token shift and cross-entropy verification

## Status

- [x] Repository initialized
- [x] Project structure created
- [x] Basic configuration added
- [ ] Minimal tokenizer and dataset loader
- [ ] MiniGPT model implementation
- [ ] Training loop
- [ ] Text generation
- [ ] Mathematical verification experiments
