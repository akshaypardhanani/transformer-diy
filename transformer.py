from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        num_encoders=6,
        num_decoders=6,
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        max_len=5000,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        self.encoder = ...
        self.decoder = ...

        self.pos_encoding = ...

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.output = nn.Linear(d_model, tgt_vocab_size)
