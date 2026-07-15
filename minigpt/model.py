import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
        )

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size "
                f"{self.config.block_size}"
            )

        token_emb = self.token_embedding(idx)

        if getattr(self.config, "use_position_embedding", True):
            pos = torch.arange(seq_len, device=idx.device)
            pos_emb = self.position_embedding(pos)
            x = token_emb + pos_emb
        else:
            x = token_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape

            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, vocab_size),
                targets.reshape(batch_size * seq_len),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                [idx, next_idx],
                dim=1,
            )

        return idx