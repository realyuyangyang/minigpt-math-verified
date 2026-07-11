class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        ids = [int(i) for i in ids]
        return bytes(ids).decode("utf-8", errors="ignore")
