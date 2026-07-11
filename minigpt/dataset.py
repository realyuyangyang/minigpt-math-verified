import torch

from minigpt.tokenizer import ByteTokenizer


class TextDataset:
    def __init__(self, file_path: str, block_size: int, train_split: float = 0.9):
        self.block_size = block_size
        self.tokenizer = ByteTokenizer()

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int, device: str):
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))

        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        x = x.to(device)
        y = y.to(device)

        return x, y
