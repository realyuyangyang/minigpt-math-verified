from dataclasses import dataclass


@dataclass
class GPTConfig:
    # data
    vocab_size: int = 256
    block_size: int = 128

    # model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int =1000
    eval_interval: int = 100
    eval_iters: int = 100

    # generation
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int | None = None

    # system
    device: str = "cuda"
