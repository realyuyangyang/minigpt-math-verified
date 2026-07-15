# Position Embedding Ablation

This experiment studies the role of positional information in a decoder-only Transformer.

The goal is to compare two MiniGPT variants:

```text
Model A: with position embedding
Model B: without position embedding
```

By comparing their validation loss curves, we can examine whether positional embeddings help the model learn order-sensitive language patterns.

---

## Background

Self-attention itself is permutation-equivariant when there is no positional information.

This means that if the input tokens are reordered, the self-attention operation will process the reordered tokens in a corresponding reordered way. The attention mechanism alone does not know whether a token appears at position 0, position 5, or position 20.

However, language is order-sensitive.

For example:

```text
dog bites man
man bites dog
```

These two sequences contain the same tokens, but their meanings are different because the token order is different.

Therefore, a Transformer language model needs positional information to distinguish token order.

---

## Positional Encoding in MiniGPT

In MiniGPT, each input token is represented by the sum of two embeddings:

```text
x = token_embedding + position_embedding
```

The token embedding tells the model what the token is.

The position embedding tells the model where the token appears in the sequence.

With position embedding enabled:

```text
x_t = token_emb[t] + pos_emb[t]
```

Without position embedding:

```text
x_t = token_emb[t]
```

In the second case, the model receives token identity but loses explicit position information.

---

## Experiment Design

We train two MiniGPT models using the same dataset and nearly identical hyperparameters.

The only difference is whether position embedding is enabled.

```text
with position embedding:
    use_position_embedding = True

without position embedding:
    use_position_embedding = False
```

Both models are trained with the same next-token prediction objective.

The validation loss is recorded during training.

---

## Implementation

The MiniGPT model supports this ablation through a configuration flag:

```text
use_position_embedding: bool = True
```

In the model forward pass:

```text
if use_position_embedding:
    x = token_embedding + position_embedding
else:
    x = token_embedding
```

This allows us to isolate the effect of positional information while keeping the rest of the architecture unchanged.

---

## Expected Result

The model with position embedding should usually achieve better validation loss because it can distinguish token order.

The model without position embedding may still learn some local token statistics, but it has weaker ability to represent order-dependent patterns.

Expected trend:

```text
with position embedding: lower validation loss
without position embedding: higher validation loss
```

---

## Result

The validation loss comparison is saved to:

```text
assets/position_embedding_ablation.png
```

The plot compares the two variants:

```text
with position embedding
without position embedding
```

The key question is whether removing positional information hurts validation performance.

---

## Interpretation

If the model with position embedding achieves lower validation loss, this supports the idea that positional information is important for decoder-only language modeling.

This result is consistent with the mathematical property of self-attention:

```text
self-attention without positional encoding is permutation-equivariant
```

Therefore, position embedding is necessary to break pure permutation equivariance and allow the model to learn sequence order.

---

## Why This Matters

This ablation connects model implementation with Transformer theory.

It shows that positional embeddings are not just an implementation detail. They provide essential order information that self-attention does not naturally contain.

For a language model, this is critical because next-token prediction depends strongly on word order.

---

## Conclusion

This experiment verifies the role of position embedding in MiniGPT.

The ablation compares a standard decoder-only Transformer against a variant without positional embeddings.

The result demonstrates that positional information is an important component for modeling ordered sequences and supports the theoretical explanation that self-attention alone does not encode token order.

