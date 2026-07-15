# Token Shift and Cross-Entropy Verification

This experiment verifies the training objective of a decoder-only language model.

MiniGPT is trained with next-token prediction. Given an input sequence, the model predicts the next token at each position.

---

## Background

For a token sequence:

```text
[t0, t1, t2, t3, t4]
```

the model input is:

```text
[t0, t1, t2, t3]
```

and the training target is:

```text
[t1, t2, t3, t4]
```

This is called token shift.

The model does not learn to reconstruct the current token. Instead, it learns to predict the next token from previous context.

---

## Autoregressive Objective

A decoder-only language model factorizes the probability of a sequence as:

```text
P(x0, x1, ..., xT) = Π P(xt | x<t)
```

This means each token is predicted using only previous tokens.

For example:

```text
P("hello") = P("h") · P("e" | "h") · P("l" | "he") · P("l" | "hel") · P("o" | "hell")
```

In practice, the model predicts all next tokens in parallel during training using causal masking.

---

## Experiment Design

We use the text:

```text
hello transformer.
```

The tokenizer converts the text into byte-level token IDs.

Then we construct:

```text
input x  = token_ids[:-1]
target y = token_ids[1:]
```

For example:

```text
input:  h  e  l  l  o  <space>  t  r  a  n  s  f  o  r  m  e  r
target: e  l  l  o  <space>  t  r  a  n  s  f  o  r  m  e  r  .
```

This directly verifies that the target sequence is shifted one position to the left.

---

## Cross-Entropy Loss Check

The MiniGPT model returns a loss from its `forward()` function.

Inside the model, the logits have shape:

```text
[B, T, vocab_size]
```

The targets have shape:

```text
[B, T]
```

To compute cross-entropy, the logits and targets are reshaped as:

```text
logits  -> [B * T, vocab_size]
targets -> [B * T]
```

Then the loss is computed as:

```text
CrossEntropyLoss(logits, targets)
```

In this experiment, we compute the loss in two ways:

```text
1. model_loss  = loss returned by MiniGPT.forward()
2. manual_loss = manually computed cross-entropy loss
```

Then we check:

```text
model_loss == manual_loss
```

---

## Result

The experiment prints the input-target token pairs:

```text
position 00: 'h' -> 'e'
position 01: 'e' -> 'l'
position 02: 'l' -> 'l'
position 03: 'l' -> 'o'
...
```

This confirms that each input token is aligned with the next token as its target.

The experiment also verifies that the model loss equals the manually computed cross-entropy loss:

```text
model loss  == manual loss
absolute difference: 0.0
```

The visualization is saved to:

```text
assets/token_shift_cross_entropy.png
```

---

## Interpretation

This experiment confirms two important implementation details.

First, the dataset correctly constructs shifted input-target pairs:

```text
x = [t0, t1, t2, ..., tT-1]
y = [t1, t2, t3, ..., tT]
```

Second, the model correctly computes next-token prediction loss using cross-entropy over the vocabulary.

This verifies that MiniGPT is trained as an autoregressive language model rather than an autoencoder.

---

## Conclusion

The token shift and cross-entropy experiment verifies the training objective of MiniGPT.

The model receives previous tokens as input and learns to predict the next token at each position.

This matches the standard decoder-only Transformer language modeling objective and confirms that the implementation correctly aligns inputs, targets, logits, and loss.

