import math
import torch

from torch import nn
from torch.nn import functional as F
from encoder import Encoder
from decoder import Decoder
from positional_encoding import PositionalEncoding


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

        self.encoder = Encoder(d_model, num_heads, num_encoders)
        self.decoder = Decoder(d_model, num_heads, num_decoders)

        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.output = nn.Linear(d_model, tgt_vocab_size)

    def create_pad_mask(self, seq, pad_token):
        # return (seq != pad_token).unsqueeze(1).unsqueeze(2)
        mask = (seq != pad_token)   # [B, S]
        mask = mask.unsqueeze(1).expand(-1, seq.size(1), -1)  # [B, S, S]
        return mask 

    def create_subsequent_mask(self, size, batch_size):
        # mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        # return (~mask).unsqueeze(0).unsqueeze(1)
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool))  # [T,T]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)          # [B,T,T]
        return mask

    def forward(self, src, tgt, src_pad_token=0, tgt_pad_token=0):
        B, S = src.size()
        _, T = tgt.size()
        # Init masks
        src_mask = self.create_pad_mask(src, src_pad_token)
        tgt_mask = self.create_pad_mask(tgt, tgt_pad_token)
        # subsequent_mask = self.create_subsequent_mask(tgt.size(1)).to(tgt.device)
        subsequent_mask = self.create_subsequent_mask(T, B).to(tgt.device)
        tgt_mask = tgt_mask & subsequent_mask

        cross_mask = (src != src_pad_token).unsqueeze(1).expand(-1, T, -1)

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)

        encoder_output = self.encoder(src_emb, src_mask)
        decoder_output = self.decoder(tgt_emb, tgt_mask, encoder_output, cross_mask)

        output = self.output(decoder_output)
        return output

    def training_step(self, src, tgt_input, expected, pad_id=0):
        output = self.forward(src, tgt_input)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), expected.view(-1), ignore_index=pad_id)
        return loss
