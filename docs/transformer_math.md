# Transformer Mathematical Notes

This document summarizes the mathematical formulas used in the first version of the MiniGPT reproduction project.

The goal is to connect the PyTorch implementation with the core mathematical mechanisms of a decoder-only Transformer.

---

## 1. Token and Position Embedding

Given a token sequence:

$$
x = [x_0, x_1, x_2, \dots, x_{T-1}]
$$

each token ID is mapped to a token embedding:

$$
e_t = E[x_t]
$$

where:

$$
E \in \mathbb{R}^{V \times d_{\text{model}}}
$$

* (V): vocabulary size
* (d_{\text{model}}): embedding dimension

A learnable position embedding is also added:

$$
p_t = P[t]
$$

where:

$$
P \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}
$$

The input representation at position (t) is:

$$
h_t^{(0)} = e_t + p_t
$$

In matrix form:

$$
H^{(0)} = E[x] + P
$$

This gives the model both token identity and positional information.

---

## 2. Query, Key, and Value Projection

For each Transformer layer, the hidden representation (H) is projected into query, key, and value matrices:

$$
Q = HW_Q
$$

$$
K = HW_K
$$

$$
V = HW_V
$$

where:

$$
W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$

* (Q): query matrix
* (K): key matrix
* (V): value matrix
* (d_k): head dimension

Intuitively:

```text
Query: what this token is looking for
Key: what each token contains
Value: what information each token provides
```

---

## 3. Scaled Dot-Product Attention

The raw attention logits are computed by:

$$
QK^T
$$

The scaled dot-product attention formula is:

$$
\text{Attention}(Q, K, V)
=========================

\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

The term:

$$
\frac{1}{\sqrt{d_k}}
$$

is used to stabilize the variance of attention logits.

---

## 4. Why Divide by (\sqrt{d_k})?

Assume each element of query vector (q) and key vector (k) is independently sampled from a standard normal distribution:

$$
q_i \sim \mathcal{N}(0, 1)
$$

$$
k_i \sim \mathcal{N}(0, 1)
$$

The dot product is:

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

Since:

$$
\mathbb{E}[q_i k_i] = 0
$$

and approximately:

$$
\text{Var}(q_i k_i) = 1
$$

the variance of the dot product grows with (d_k):

$$
\text{Var}(q \cdot k) \approx d_k
$$

After scaling:

$$
\frac{q \cdot k}{\sqrt{d_k}}
$$

the variance becomes:

$$
\text{Var}
\left(
\frac{q \cdot k}{\sqrt{d_k}}
\right)
=======

\frac{\text{Var}(q \cdot k)}{d_k}
\approx 1
$$

Therefore:

$$
\text{Var}(QK^T) \propto d_k
$$

but:

$$
\text{Var}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
\approx 1
$$

This is why attention scaling helps keep the softmax input numerically stable.

---

## 5. Causal Mask

In decoder-only language modeling, token position (t) is only allowed to attend to positions:

$$
0, 1, 2, \dots, t
$$

It must not attend to future positions:

$$
t+1, t+2, \dots, T-1
$$

The causal mask (M) is defined as:

$$
M_{ij}
======

\begin{cases}
1, & j \le i \
0, & j > i
\end{cases}
$$

where:

* (i): query position
* (j): key position

The attention logits are masked before softmax:

$$
A =
\frac{QK^T}{\sqrt{d_k}}
$$

For future positions:

$$
A_{ij} = -\infty \quad \text{if } j > i
$$

Then:

$$
\text{softmax}(A_{ij}) = 0 \quad \text{if } j > i
$$

This ensures that future tokens receive zero attention probability.

---

## 6. Autoregressive Factorization

A decoder-only language model factorizes the probability of a token sequence as:

$$
P(x_0, x_1, \dots, x_T)
=======================

\prod_{t=0}^{T}
P(x_t \mid x_{<t})
$$

where:

$$
x_{<t} = [x_0, x_1, \dots, x_{t-1}]
$$

This means each token is predicted using only previous tokens.

For example:

$$
P(x_0, x_1, x_2, x_3)
=====================

P(x_0)
P(x_1 \mid x_0)
P(x_2 \mid x_0, x_1)
P(x_3 \mid x_0, x_1, x_2)
$$

The causal mask enforces this constraint inside self-attention.

---

## 7. Token Shift for Next-Token Prediction

Given a token sequence:

$$
[x_0, x_1, x_2, x_3, x_4]
$$

the model input is:

$$
[x_0, x_1, x_2, x_3]
$$

and the target is:

$$
[x_1, x_2, x_3, x_4]
$$

In code:

```text
input  = token_ids[:-1]
target = token_ids[1:]
```

So the model learns:

$$
x_t \rightarrow x_{t+1}
$$

At each position, the model predicts the next token.

---

## 8. Cross-Entropy Loss

The model outputs logits:

$$
Z \in \mathbb{R}^{B \times T \times V}
$$

where:

* (B): batch size
* (T): sequence length
* (V): vocabulary size

The target tokens are:

$$
Y \in \mathbb{R}^{B \times T}
$$

For each position, the model predicts a probability distribution over the vocabulary:

$$
p_{t} = \text{softmax}(z_t)
$$

The cross-entropy loss for one token is:

$$
\mathcal{L}_t = -\log p_t(y_t)
$$

For the full batch and sequence:

$$
\mathcal{L}
===========

-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log P(y_{b,t} \mid x_{b,\le t})
$$

In implementation, logits are reshaped as:

$$
[B, T, V] \rightarrow [B \times T, V]
$$

and targets are reshaped as:

$$
[B, T] \rightarrow [B \times T]
$$

Then standard cross-entropy is applied.

---

## 9. Multi-Head Attention

Instead of using one attention head, the Transformer uses multiple heads.

For head (i):

$$
\text{head}_i
=============

\text{Attention}(Q_i, K_i, V_i)
$$

The outputs of all heads are concatenated:

$$
H =
\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)
$$

Then a final output projection is applied:

$$
\text{MHA}(Q, K, V)
===================

H W_O
$$

where:

$$
W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

Multi-head attention allows the model to attend to different types of relationships in parallel.

---

## 10. Pre-LN Transformer Block

MiniGPT uses a Pre-LayerNorm Transformer block.

The attention sublayer is:

$$
x' = x + \text{MHA}(\text{LN}(x))
$$

The feed-forward sublayer is:

$$
x'' = x' + \text{MLP}(\text{LN}(x'))
$$

So the full block is:

$$
x \rightarrow x + \text{Attention}(\text{LayerNorm}(x))
$$

$$
x \rightarrow x + \text{MLP}(\text{LayerNorm}(x))
$$

Pre-LN is commonly used because it improves training stability compared with placing LayerNorm after the residual connection.

---

## 11. Feed-Forward Network

The feed-forward network is applied independently at each token position.

It usually expands the hidden dimension and then projects it back:

$$
\text{MLP}(x)
=============

W_2 \sigma(W_1 x + b_1) + b_2
$$

where:

$$
W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}
$$

$$
W_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}
$$

MiniGPT uses GELU as the activation function:

$$
\sigma = \text{GELU}
$$

---

## 12. Position Embedding and Permutation Equivariance

Self-attention without positional information is permutation-equivariant.

This means that if the input tokens are permuted, the output representations are permuted in the same way.

Formally, for a permutation matrix (\Pi):

$$
\text{SelfAttention}(\Pi X)
===========================

\Pi \text{SelfAttention}(X)
$$

This means self-attention alone does not know the original order of tokens.

However, language is order-sensitive.

For example:

```text
dog bites man
man bites dog
```

These two sequences contain the same words but have different meanings.

Position embedding breaks this pure permutation equivariance by adding position-specific information:

$$
H^{(0)} = E[x] + P
$$

Therefore, the model can distinguish not only what token appears, but also where it appears.

---

## 13. Autoregressive Generation

During generation, the model repeatedly predicts one token at a time.

Given current context:

$$
[x_0, x_1, \dots, x_t]
$$

the model predicts the distribution for the next token:

$$
P(x_{t+1} \mid x_{\le t})
$$

Then a token is sampled:

$$
x_{t+1} \sim P(x_{t+1} \mid x_{\le t})
$$

The new token is appended to the context:

$$
[x_0, x_1, \dots, x_t, x_{t+1}]
$$

This process repeats until the desired number of tokens is generated.

---

## 14. Summary of Verified Mechanisms

The first version of this project verifies the following mathematical mechanisms:

```text
1. Scaled dot-product attention:
   QK^T / sqrt(d_k)

2. Attention scaling variance:
   Var(QK^T) ≈ d_k
   Var(QK^T / sqrt(d_k)) ≈ 1

3. Causal mask:
   M_ij = 1 if j <= i, else 0

4. Autoregressive factorization:
   P(x_0, ..., x_T) = Π P(x_t | x_<t)

5. Token shift:
   input  = token_ids[:-1]
   target = token_ids[1:]

6. Cross-entropy loss:
   L = -log P(next token | previous tokens)

7. Position embedding:
   hidden = token_embedding + position_embedding

8. Permutation equivariance:
   Self-attention without position information does not encode order.
```

---

## Conclusion

The MiniGPT implementation is not only a runnable decoder-only Transformer, but also a mathematically verified reproduction.

Each core component corresponds to a specific mathematical idea:

```text
attention scaling -> variance stabilization
causal mask -> autoregressive constraint
token shift -> next-token prediction
cross-entropy -> language modeling objective
position embedding -> order information
```

These formulas explain why the implementation works and how the experiments verify the key mechanisms of Transformer language models.

