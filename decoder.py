from multi_headed_attention import MultiHeadedAttention
from feed_forward import FeedForward
from torch import nn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1) -> None:
        super().__init__()

        self.masked_attention = MultiHeadedAttention(d_model, num_heads, dropout)

        self.masked_attention_norm = nn.LayerNorm(d_model)

        self.attention = MultiHeadedAttention(d_model, num_heads, dropout)

        self.attention_norm = nn.LayerNorm(d_model)

        self.feed_forward_network = FeedForward(d_model, d_ff, dropout)
        self.feed_forward_network_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, encoding, tgt_mask, encoding_mask):
        x = tgt
        x = x + self.masked_attention(x, x, x, tgt_mask)
        x = self.masked_attention_norm(x)
        x = x + self.attention(x, encoding, encoding, mask=encoding_mask)
        x = self.attention_norm(x)
        x = x + self.feed_forward_network(x)
        x = self.feed_forward_network_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_decoders) -> None:
        super().__init__()

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_decoders)]
        )

    def forward(self, tgt, tgt_mask, encoding, encoding_mask):
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, encoding, tgt_mask, encoding_mask)
        return output
