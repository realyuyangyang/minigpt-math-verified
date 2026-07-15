# MiniGPT Reproduction with Mathematical Verification

A from-scratch PyTorch reproduction of a decoder-only Transformer with mathematical verification experiments for attention scaling, causal masking, positional encoding, and next-token prediction.

This project is designed to demonstrate not only how a GPT-style model is implemented, but also why its key components are mathematically necessary.

---

## Project Motivation

The goal of this project is to reproduce a minimal GPT-style decoder-only Transformer from scratch and verify its core mechanisms through controlled experiments.

Instead of only calling high-level libraries, this project implements the main Transformer components directly in PyTorch:

* Byte-level tokenizer
* Token embedding
* Position embedding
* Scaled dot-product attention
* Multi-head causal self-attention
* Pre-LN Transformer block
* Feed-forward network
* Residual connection
* Next-token prediction loss
* Autoregressive text generation

The project also includes mathematical verification experiments to test whether the implementation follows the expected Transformer behavior.

---

## Model Architecture

MiniGPT follows the standard decoder-only Transformer architecture:

```text
input tokens
    ↓
token embedding + position embedding
    ↓
Pre-LN Transformer blocks
    ↓
final LayerNorm
    ↓
linear language modeling head
    ↓
next-token logits
```

Each Transformer block contains:

```text
x → LayerNorm → Causal Self-Attention → Residual
x → LayerNorm → Feed Forward Network → Residual
```

The model is trained with next-token prediction:

```text
input:  [t0, t1, t2, t3]
target: [t1, t2, t3, t4]
```

---

## Repository Structure

```text
minigpt-math-verified/
├── config.py
├── train.py
├── generate.py
├── requirements.txt
│
├── minigpt/
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── attention.py
│   ├── block.py
│   └── model.py
│
├── experiments/
│   ├── 01_attention_scaling_variance.py
│   ├── 02_causal_mask_leakage_test.py
│   ├── 03_attention_heatmap.py
│   ├── 04_token_shift_cross_entropy.py
│   └── 05_position_embedding_ablation.py
│
├── assets/
│   ├── loss_curve.png
│   ├── attention_scaling_variance.png
│   ├── mask_leakage_test.png
│   ├── attention_heatmap.png
│   ├── token_shift_cross_entropy.png
│   └── position_embedding_ablation.png
│
└── results/
    ├── generation_examples.md
    ├── attention_scaling_variance.md
    ├── causal_mask_leakage_test.md
    ├── attention_heatmap.md
    ├── token_shift_cross_entropy.md
    └── position_embedding_ablation.md
```

---

## Implemented Components

### Byte-level Tokenizer

The tokenizer maps UTF-8 bytes to integer token IDs.

```text
vocab_size = 256
```

This keeps the tokenizer simple and avoids relying on external tokenizer training.

---

### Causal Self-Attention

The attention module computes:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

A lower-triangular causal mask is applied so that each token can only attend to itself and previous tokens.

---

### Pre-LN Transformer Block

Each block uses Pre-LayerNorm:

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

This follows the common stable training design used in modern decoder-only Transformers.

---

### Next-token Prediction

The dataset constructs shifted input-target pairs:

```text
x = [t0, t1, t2, ..., tT-1]
y = [t1, t2, t3, ..., tT]
```

The model computes cross-entropy loss over the vocabulary at every sequence position.

---

## Training

Run training with:

```bash
python train.py
```

The script trains MiniGPT on `data/input.txt`, saves a local checkpoint to:

```text
results/minigpt_checkpoint.pt
```

and saves the loss curve to:

```text
assets/loss_curve.png
```

The checkpoint file is intentionally ignored by Git because model weights should not be committed to the repository.

---

## Text Generation

After training, generate text with:

```bash
python generate.py
```

The generation script loads the trained checkpoint and performs autoregressive generation from a prompt.

Generation results can be recorded in:

```text
results/generation_examples.md
```

---

## Mathematical Verification Experiments

### 1. Attention Scaling Variance Experiment

Script:

```bash
python experiments/01_attention_scaling_variance.py
```

This experiment verifies why scaled dot-product attention divides logits by `sqrt(d_k)`.

Without scaling:

```text
Var(QK^T) grows approximately with d_k
```

With scaling:

```text
Var(QK^T / sqrt(d_k)) remains close to 1
```

This confirms that the scaling factor stabilizes the variance of attention logits.

Result figure:

```text
assets/attention_scaling_variance.png
```

Detailed report:

```text
results/attention_scaling_variance.md
```

---

### 2. Causal Mask Leakage Test

Script:

```bash
python experiments/02_causal_mask_leakage_test.py
```

This experiment verifies that future tokens do not leak into previous positions.

It includes two tests:

```text
1. Future attention weights should be exactly zero.
2. Changing future tokens should not affect prefix logits.
```

The key expected result is:

```text
max future attention: 0.0
max prefix logits diff after changing future tokens: 0.0
```

Result figure:

```text
assets/mask_leakage_test.png
```

Detailed report:

```text
results/causal_mask_leakage_test.md
```

---

### 3. Attention Heatmap Visualization

Script:

```bash
python experiments/03_attention_heatmap.py
```

This experiment visualizes the attention matrix from a trained MiniGPT model.

The heatmap should show a lower-triangular attention pattern, confirming that each token only attends to itself and previous tokens.

Result figure:

```text
assets/attention_heatmap.png
```

Detailed report:

```text
results/attention_heatmap.md
```

---

### 4. Token Shift and Cross-Entropy Verification

Script:

```bash
python experiments/04_token_shift_cross_entropy.py
```

This experiment verifies the next-token prediction objective.

It checks that:

```text
input x  = token_ids[:-1]
target y = token_ids[1:]
```

and verifies that the model loss is equal to a manually computed cross-entropy loss.

Result figure:

```text
assets/token_shift_cross_entropy.png
```

Detailed report:

```text
results/token_shift_cross_entropy.md
```

---

### 5. Position Embedding Ablation

Script:

```bash
python experiments/05_position_embedding_ablation.py
```

This experiment compares two model variants:

```text
Model A: with position embedding
Model B: without position embedding
```

The goal is to show that positional information is important because self-attention without positional encoding is permutation-equivariant.

Result figure:

```text
assets/position_embedding_ablation.png
```

Detailed report:

```text
results/position_embedding_ablation.md
```

---

## Key Results

This project verifies several core Transformer mechanisms:

```text
1. Attention scaling stabilizes the variance of QK^T logits.
2. Causal masking prevents future-token information leakage.
3. Attention heatmaps show the lower-triangular causal structure.
4. Token shift correctly implements next-token prediction.
5. Position embedding provides necessary order information.
```

Together, these experiments show that the MiniGPT implementation is not only runnable, but also mathematically grounded.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Generate text:

```bash
python generate.py
```

Run experiments:

```bash
python experiments/01_attention_scaling_variance.py
python experiments/02_causal_mask_leakage_test.py
python experiments/03_attention_heatmap.py
python experiments/04_token_shift_cross_entropy.py
python experiments/05_position_embedding_ablation.py
```

---

## Status

* [x] Repository initialized
* [x] Project structure created
* [x] Byte-level tokenizer
* [x] Dataset loader with token shift
* [x] Decoder-only MiniGPT implementation
* [x] Training loop
* [x] Text generation
* [x] Attention scaling variance experiment
* [x] Causal mask leakage test
* [x] Attention heatmap visualization
* [x] Token shift and cross-entropy verification
* [x] Position embedding ablation
* [x] Experiment reports

---

## What This Project Demonstrates

This project demonstrates the ability to:

* Implement a decoder-only Transformer from scratch in PyTorch
* Understand the mathematical role of attention scaling
* Verify causal masking through both numerical and visual tests
* Explain next-token prediction and cross-entropy alignment
* Design ablation experiments for Transformer components
* Build a clean, reproducible GitHub project for model understanding

---

## Future Work

Possible extensions include:

* Add top-k and temperature comparison examples
* Add Pre-LN vs Post-LN comparison
* Add different context length experiments
* Add multi-head attention visualization across layers
* Train on a larger text corpus
* Add BPE tokenizer support

