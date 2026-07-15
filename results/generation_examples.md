# Generation Examples

This file records text generation examples from the trained MiniGPT model.

The goal is not to demonstrate high-quality language generation, but to verify that the full autoregressive generation pipeline works correctly:

```text
prompt -> tokenizer.encode -> model.generate -> tokenizer.decode -> generated text
```

---

## Example 1

### Prompt

```text
hello
```

### Generated Text

```text
hello transformer.
```

### Notes

This example verifies that the trained checkpoint can be loaded and used for autoregressive text generation.

---

## Example 2

### Prompt

```text
the model
```

### Generated Text

```text
the model is trained to predict the next token.
```

### Notes

The generated text may be imperfect because MiniGPT is trained on a very small dataset. The main purpose is to verify the training and generation workflow.

---

## Example 3

### Prompt

```text
attention
```

### Generated Text

```text
attention is computed from query, key, and value vectors.
```

### Notes

This example connects the generated output with the Transformer-related training corpus.

---

## Interpretation

The generated examples confirm that the MiniGPT pipeline is functional:

```text
1. The trained checkpoint can be loaded.
2. The prompt can be encoded into byte-level token IDs.
3. The model can autoregressively sample new tokens.
4. The generated token IDs can be decoded back into text.
```

Because this is a small from-scratch reproduction, generation quality is not the primary objective.

The main objective is to demonstrate a complete decoder-only Transformer workflow, including training, checkpoint loading, and autoregressive generation.
